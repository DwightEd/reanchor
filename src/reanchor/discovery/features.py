"""Measure local-to-remote attention transitions without selecting events."""

from __future__ import annotations

import numpy as np

SITE_METRICS = (
    "current_local_mass",
    "current_remote_mass",
    "previous_local_mass",
    "remote_gain",
    "positive_remote_gain",
    "time_tv",
    "gain_focality",
    "effective_sources",
    "mean_distance",
    "prompt_positive_gain",
    "evidence_positive_gain",
    "history_positive_gain",
)


def _normalize_ordinary(row: np.ndarray, ordinary: np.ndarray) -> np.ndarray:
    retained = row * ordinary
    total = retained.sum(-1, keepdims=True)
    return np.divide(
        retained,
        total,
        out=np.full_like(retained, np.nan, dtype=np.float32),
        where=total > 0,
    )


def measure_attention_transitions(
    attention: np.ndarray,
    *,
    rows: np.ndarray,
    response_start: int,
    special_mask: np.ndarray,
    evidence_mask: np.ndarray,
    window: int,
) -> dict[str, np.ndarray]:
    """Return per-head features for one layer of complete native attention.

    `attention` has `[head, query-row, source-token]` axes. Adjacent rows are
    compared under the current query's source partition so crossing the window
    boundary by age alone cannot create remote gain.
    """

    attention = np.asarray(attention, dtype=np.float32)
    rows = np.asarray(rows, dtype=np.int64)
    special = np.asarray(special_mask, dtype=bool)
    evidence = np.asarray(evidence_mask, dtype=bool)
    if attention.ndim != 3:
        raise ValueError("attention must have [head, query-row, source-token] axes")
    heads, row_count, token_count = attention.shape
    if rows.shape != (row_count,) or special.shape != (token_count,):
        raise ValueError("row positions and special mask must match attention axes")
    if evidence.shape != special.shape:
        raise ValueError("evidence mask must cover source tokens")
    if window < 1 or not 0 < response_start < token_count:
        raise ValueError("window and response_start are invalid")

    result = {name: np.full((heads, row_count), np.nan, dtype=np.float32) for name in SITE_METRICS}
    result["peak_source"] = np.full((heads, row_count), -1, dtype=np.int32)
    eligible = np.zeros(row_count, dtype=bool)
    source = np.arange(token_count)
    prompt = source < response_start
    history = ~prompt

    for index in range(1, row_count):
        query = int(rows[index])
        previous_query = int(rows[index - 1])
        legal_ordinary = (source <= query) & ~special
        remote = legal_ordinary & (query - source > window)
        local = legal_ordinary & ~remote
        current = _normalize_ordinary(attention[:, index], legal_ordinary)
        previous = _normalize_ordinary(attention[:, index - 1], legal_ordinary)
        gain = (current - previous) * remote
        positive = np.maximum(gain, 0)
        positive_total = positive.sum(-1)
        distribution = np.divide(
            positive,
            positive_total[:, None],
            out=np.zeros_like(positive),
            where=positive_total[:, None] > 0,
        )
        entropy = -np.where(
            distribution > 0,
            distribution * np.log(np.maximum(distribution, 1e-30)),
            0,
        ).sum(-1)
        best = gain.argmax(-1)
        best_gain = gain[np.arange(heads), best]

        result["current_local_mass"][:, index] = (current * local).sum(-1)
        result["current_remote_mass"][:, index] = (current * remote).sum(-1)
        result["previous_local_mass"][:, index] = (previous * local).sum(-1)
        result["remote_gain"][:, index] = gain.sum(-1)
        result["positive_remote_gain"][:, index] = positive_total
        result["time_tv"][:, index] = 0.5 * np.abs(current - previous).sum(-1)
        result["gain_focality"][:, index] = np.divide(
            positive.max(-1),
            positive_total,
            out=np.full(heads, np.nan, dtype=np.float32),
            where=positive_total > 0,
        )
        result["effective_sources"][:, index] = np.where(
            positive_total > 0, np.exp(entropy), np.nan
        )
        result["mean_distance"][:, index] = np.where(
            positive_total > 0,
            (distribution * (query - source)).sum(-1),
            np.nan,
        )
        result["prompt_positive_gain"][:, index] = positive[:, prompt].sum(-1)
        result["evidence_positive_gain"][:, index] = positive[:, evidence].sum(-1)
        result["history_positive_gain"][:, index] = positive[:, history].sum(-1)
        result["peak_source"][:, index] = np.where(best_gain > 0, best, -1)
        eligible[index] = (
            query >= response_start
            and query + 1 < token_count
            and not special[query]
            and not special[previous_query]
        )

    result["eligible"] = eligible
    return result
