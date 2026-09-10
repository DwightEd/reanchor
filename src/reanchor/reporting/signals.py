"""Build readable token-level signals before outcome labels are joined."""

from __future__ import annotations

import json

import numpy as np

TRANSITION_SCORES = (
    "predictor_surprisal",
    "predictor_entropy",
    "reanchor_score",
    "sparse_score",
    "broad_score",
    "local_attention_mass",
    "attention_stability",
    "local_mass_drop",
    "positive_remote_gain",
    "remote_gain_focality",
    "remote_effective_sources",
    "remote_mean_distance",
    "evidence_gain_share",
    "other_prompt_gain_share",
    "response_history_gain_share",
)


def transition_signal_rows(
    sample,
    transitions,
    events,
    *,
    source_kind=None,
    predictor_logprob=None,
    predictor_entropy=None,
) -> list[dict]:
    """Collapse layer/head measurements into one label-free row per token."""

    positions = np.asarray(transitions["row_position"], dtype=int)
    eligible = np.asarray(transitions["eligible"], dtype=bool)
    token_text = np.asarray(transitions["token_text"]).astype(str)
    special = np.asarray(transitions["special_mask"], dtype=bool)
    evidence = np.asarray(transitions["evidence_mask"], dtype=bool)
    response_start = int(transitions["response_start"])
    if positions.shape != eligible.shape:
        raise ValueError("transition row and eligibility axes disagree")
    for name, values in (
        ("predictor_logprob", predictor_logprob),
        ("predictor_entropy", predictor_entropy),
    ):
        if values is not None and np.asarray(values).shape != positions.shape:
            raise ValueError(f"{name} must share the transition row axis")

    rows = []
    for index in np.flatnonzero(eligible):
        query = int(positions[index])
        target = query + 1
        if target >= len(token_text) or special[target]:
            continue
        positive = _site_values(transitions, "positive_remote_gain", index)
        prompt = _site_values(transitions, "prompt_positive_gain", index)
        evidence_gain = _site_values(transitions, "evidence_positive_gain", index)
        history = _site_values(transitions, "history_positive_gain", index)
        gains = {
            "evidence": float(evidence_gain.sum()),
            "other_prompt": float(np.maximum(prompt - evidence_gain, 0).sum()),
            "response_history": float(history.sum()),
        }
        total_gain = float(positive.sum())
        peak_source = _dominant_peak_source(transitions["peak_source"], index)
        rows.append(
            {
                "split": sample.split,
                "task": sample.task,
                "sample_id": sample.sample_id,
                "source_id": sample.source_id,
                "query_position": query,
                "target_position": target,
                "target_response_index": target - response_start,
                "query_token": token_text[query],
                "target_token": token_text[target],
                "remote_peak_position": peak_source,
                "remote_peak_token": token_text[peak_source] if peak_source >= 0 else "",
                "remote_peak_category": _source_category(
                    peak_source, response_start, evidence, special, source_kind
                ),
                "dominant_remote_route": max(gains, key=gains.get) if total_gain > 0 else "none",
                "reanchor_type": str(events["reanchor_type"][index]),
                "selected_transition": bool(events["significant"][index]),
                "anchor": bool(events["anchor"][index]),
                "predictor_surprisal": (
                    -float(predictor_logprob[index])
                    if predictor_logprob is not None
                    else np.nan
                ),
                "predictor_entropy": (
                    float(predictor_entropy[index])
                    if predictor_entropy is not None
                    else np.nan
                ),
                "reanchor_score": float(
                    max(events["sparse_score"][index], events["broad_score"][index])
                ),
                "sparse_score": float(events["sparse_score"][index]),
                "broad_score": float(events["broad_score"][index]),
                "local_attention_mass": _mean(transitions, "current_local_mass", index),
                "attention_stability": 1.0 - _mean(transitions, "time_tv", index),
                "local_mass_drop": _mean(transitions, "previous_local_mass", index)
                - _mean(transitions, "current_local_mass", index),
                "positive_remote_gain": _mean(transitions, "positive_remote_gain", index),
                "remote_gain_focality": _mean(transitions, "gain_focality", index),
                "remote_effective_sources": _mean(transitions, "effective_sources", index),
                "remote_mean_distance": _mean(transitions, "mean_distance", index),
                **{
                    f"{name}_gain_share": gain / total_gain if total_gain > 0 else np.nan
                    for name, gain in gains.items()
                },
            }
        )
    return rows


def response_record(sample, transitions) -> dict:
    """Keep the analyzed answer next to its token-level measurements."""

    token_text = np.asarray(transitions["token_text"]).astype(str)
    special = np.asarray(transitions["special_mask"], dtype=bool)
    pieces = token_text[sample.response_start :][~special[sample.response_start :]]
    return {
        "split": sample.split,
        "task": sample.task,
        "sample_id": sample.sample_id,
        "source_id": sample.source_id,
        "response_tokens": len(pieces),
        "response_text": "".join(pieces),
        "response_tokens_json": json.dumps(pieces.tolist(), ensure_ascii=False),
    }


def _site_values(transitions, name, index):
    values = np.asarray(transitions[name], dtype=float)
    if values.ndim != 3:
        raise ValueError(f"{name} must have layer/head/row axes")
    return np.nan_to_num(values[:, :, index], nan=0.0)


def _mean(transitions, name, index):
    values = np.asarray(transitions[name], dtype=float)[:, :, index]
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else np.nan


def _dominant_peak_source(peak_sources, index):
    values = np.asarray(peak_sources, dtype=int)[:, :, index].reshape(-1)
    values = values[values >= 0]
    if not len(values):
        return -1
    sources, counts = np.unique(values, return_counts=True)
    return int(sources[np.argmax(counts)])


def _source_category(position, response_start, evidence, special, source_kind):
    if position < 0:
        return "none"
    if special[position]:
        return "special"
    if position >= response_start:
        return "response_history"
    if source_kind is not None:
        kinds = np.asarray(source_kind).astype(str)
        if kinds.shape != evidence.shape:
            raise ValueError("source_kind must cover source tokens")
        if kinds[position] in ("constraint", "content", "other"):
            return str(kinds[position])
    return "evidence" if evidence[position] else "other_prompt"
