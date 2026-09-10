"""Join frozen scores with real span annotations, never generated pseudo labels."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from reanchor.calibration import verified_scores
from reanchor.io import digest, empty_directory, read_jsonl, write_json, write_jsonl
from reanchor.ragtruth import dataset_identity, source_content


def binary_metrics(rows, target: str, score: str, multiplicity=None):
    if not rows:
        return {"auroc": None, "ap": None, "prevalence": None, "tokens": 0, "positive": 0}
    counts = Counter(r["source_id"] for r in rows)
    y = np.asarray([r[target] for r in rows], dtype=float)
    scores = np.asarray([r[score] for r in rows], dtype=float)
    weights = np.asarray(
        [
            (1 if multiplicity is None else multiplicity.get(r["source_id"], 0))
            / counts[r["source_id"]]
            for r in rows
        ]
    )
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite scores")
    pos, neg = float((weights * y).sum()), float((weights * (1 - y)).sum())
    result = {
        "auroc": None,
        "ap": None,
        "prevalence": pos / (pos + neg) if pos + neg else None,
        "tokens": len(rows),
        "positive": int(y.sum()),
        "sources": len(counts),
    }
    if pos == 0 or neg == 0:
        return result
    order = np.argsort(-scores, kind="stable")
    y, weights, scores = y[order], weights[order], scores[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    tp = np.cumsum(weights * y)[ends]
    fp = np.cumsum(weights * (1 - y))[ends]
    recall = np.r_[0, tp / pos]
    fpr = np.r_[0, fp / neg]
    result["auroc"] = float(np.sum(np.diff(fpr) * (recall[1:] + recall[:-1]) / 2))
    result["ap"] = float(np.sum(np.diff(recall) * tp / np.maximum(tp + fp, 1e-30)))
    return result


def bootstrap_metrics(rows, target, score, repeats, seed):
    result = binary_metrics(rows, target, score)
    sources = sorted({r["source_id"] for r in rows})
    if repeats <= 0 or len(sources) < 2:
        return result
    rng = np.random.default_rng(seed)
    samples = defaultdict(list)
    for _ in range(repeats):
        multiplicity = Counter(rng.choice(sources, size=len(sources), replace=True).tolist())
        measured = binary_metrics(rows, target, score, multiplicity)
        for name in ("auroc", "ap"):
            if measured[name] is not None:
                samples[name].append(measured[name])
    for name, values in samples.items():
        result[name + "_ci95"] = np.quantile(values, [0.025, 0.975]).tolist()
        result[name + "_valid_bootstraps"] = len(values)
    return result


def join_annotations(rows, responses):
    by_response = defaultdict(list)
    for row in rows:
        by_response[row["response_id"]].append(row)
    joined = []
    for response_id, tokens in by_response.items():
        if response_id not in responses:
            raise ValueError(f"missing annotation response {response_id}")
        truth = responses[response_id]
        text = truth["response"]
        tokens = sorted(tokens, key=lambda r: r["token_index"])
        if [r["token_index"] for r in tokens] != list(range(tokens[0]["response_token_count"])):
            raise ValueError("evaluation requires complete all-token scores, including token zero")
        if any(
            r["response_digest"] != digest(text) or r["source_id"] != str(truth["source_id"])
            for r in tokens
        ):
            raise ValueError("scored text/source differs from annotation text/source")
        spans = []
        for span in truth["labels"]:
            start, end = span["start"], span["end"]
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or not 0 <= start < end <= len(text)
            ):
                raise ValueError("invalid annotation character interval")
            if span.get("text", text[start:end]) != text[start:end]:
                raise ValueError("annotation text does not match offsets")
            spans.append((start, end))
        onset_tokens = set()
        for start, end in spans:
            overlaps = [
                r["token_index"] for r in tokens if r["char_start"] < end and r["char_end"] > start
            ]
            if not overlaps:
                raise ValueError("annotation span has no overlapping scored token")
            onset_tokens.add(min(overlaps))
        seen_error = False
        for row in tokens:
            if not 0 <= row["char_start"] <= row["char_end"] <= len(text):
                raise ValueError("invalid scored token offset")
            error = any(row["char_start"] < end and row["char_end"] > start for start, end in spans)
            joined.append(
                {
                    **row,
                    "hallucinated": error,
                    "span_onset": row["token_index"] in onset_tokens,
                    "first_error": error and not seen_error,
                    "pre_first_error": not seen_error and not error,
                }
            )
            seen_error = seen_error or error
    return joined


def alarm_metrics(rows, threshold=0.99, max_delay=32):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["response_id"]].append(row)
    records = []
    for response_id, tokens in grouped.items():
        tokens = sorted(tokens, key=lambda r: r["token_index"])
        onset = next((r["token_index"] for r in tokens if r["first_error"]), None)
        alarms = [r["token_index"] for r in tokens if r["score"] >= threshold]
        in_window = [] if onset is None else [t for t in alarms if onset <= t <= onset + max_delay]
        records.append(
            {
                "response_id": response_id,
                "source_id": tokens[0]["source_id"],
                "onset": onset,
                "first_alarm": alarms[0] if alarms else None,
                "first_alarm_is_onset": onset is not None and bool(alarms) and alarms[0] == onset,
                "delay_after_onset": in_window[0] - onset if in_window else None,
                "miss_within_window": onset is not None and not in_window,
                "pre_error_tokens": sum(r["pre_first_error"] for r in tokens),
                "false_alarms": sum(
                    r["pre_first_error"] and r["score"] >= threshold for r in tokens
                ),
            }
        )
    error_responses = [r for r in records if r["onset"] is not None]
    by_source = defaultdict(list)
    for record in records:
        by_source[record["source_id"]].append(record)
    source_recall, source_false_alarm = [], []
    for source_records in by_source.values():
        source_errors = [r for r in source_records if r["onset"] is not None]
        if source_errors:
            source_recall.append(
                sum(r["first_alarm_is_onset"] for r in source_errors) / len(source_errors)
            )
        source_tokens = sum(r["pre_error_tokens"] for r in source_records)
        if source_tokens:
            source_false_alarm.append(
                1000 * sum(r["false_alarms"] for r in source_records) / source_tokens
            )
    denominator = sum(r["pre_error_tokens"] for r in records)
    summary = {
        "primary_alarm_metrics": [
            "source_balanced_exact_onset_recall",
            "source_balanced_false_alarms_per_1000",
        ],
        "unprefixed_alarm_metrics": "pooled response/token denominators; secondary",
        "threshold": threshold,
        "max_delay": max_delay,
        "false_alarms_per_1000_pre_error_tokens": 1000
        * sum(r["false_alarms"] for r in records)
        / denominator
        if denominator
        else None,
        "first_alarm_exact_onset_recall": sum(r["first_alarm_is_onset"] for r in error_responses)
        / len(error_responses)
        if error_responses
        else None,
        "misses_within_window": sum(r["miss_within_window"] for r in error_responses),
        "error_responses": len(error_responses),
        "source_balanced_exact_onset_recall": sum(source_recall) / len(source_recall)
        if source_recall
        else None,
        "source_balanced_false_alarms_per_1000": sum(source_false_alarm) / len(source_false_alarm)
        if source_false_alarm
        else None,
        "delay_note": "posthoc delay after onset; prior false alarms reported separately",
    }
    return records, summary


def evaluate_ragtruth(scores: Path, dataset: Path, output: Path, *, bootstrap=200, seed=42):
    rows, frozen = verified_scores(scores)  # Verify score freeze BEFORE opening labels.
    if frozen["scope"] != "all-response-tokens" or "reference_sources" not in frozen:
        raise ValueError("requires calibrated, complete natural-response scores")
    if any(row["split"] != "test" for row in rows):
        raise ValueError("evaluation accepts held-out test split only")
    if frozen.get("dataset_identity") != dataset_identity(dataset):
        raise ValueError("dataset contents changed since score preparation")
    sources = {}
    for item in read_jsonl(dataset / "source_info.jsonl"):
        key = str(item["source_id"])
        if key in sources:
            raise ValueError("duplicate annotation source")
        sources[key] = item
    for row in rows:
        item = sources.get(row["source_id"])
        if item is None:
            raise ValueError("missing annotation source")
        source, instruction = source_content(item)
        if (
            row["source_text_digest"] != digest(source)
            or row["instruction_digest"] != digest(instruction)
            or row["task"] != item["task_type"]
        ):
            raise ValueError("scored evidence/instruction differs from annotation source")
    wanted = {r["response_id"] for r in rows}
    truth = {}
    for row in read_jsonl(dataset / "response.jsonl"):
        if str(row["id"]) in wanted:
            if str(row["id"]) in truth:
                raise ValueError("duplicate annotation response")
            truth[str(row["id"])] = row
    joined = join_annotations(rows, truth)
    cohorts = {
        "first_error_full_stream": (joined, "first_error"),
        "span_onset_vs_normal": (
            [r for r in joined if r["span_onset"] or not r["hallucinated"]],
            "span_onset",
        ),
        "all_hallucinated_tokens": (joined, "hallucinated"),
    }
    metrics = {}
    for name, (cohort, target) in cohorts.items():
        metrics[name] = {
            score: bootstrap_metrics(cohort, target, score, bootstrap, seed)
            for score in ("score", "raw_score", "negative_margin", "margin_rank")
        }
    alarms, alarm_summary = alarm_metrics(joined, frozen.get("alarm_quantile", 0.99))
    empty_directory(output)
    write_jsonl(output / "joined.jsonl", joined)
    write_jsonl(output / "alarms.jsonl", alarms)
    result = {
        "schema": "reanchor/first-error-evaluation@1",
        "metrics": metrics,
        "alarms": alarm_summary,
        "bootstrap": bootstrap,
        "seed": seed,
        "frozen_scores_digest": frozen["scores_digest"],
        "dataset_identity": frozen["dataset_identity"],
        "span_onset_policy": "first overlap per official span; shared onset tokens deduplicated",
        "weighting": "equal total token weight per source within each metric cohort",
        "label_policy": "all official spans, including implicit_true; source-relative",
        "claim_boundary": "exploratory observer detection; not native causal mechanism",
    }
    write_json(output / "summary.json", result)
    return result
