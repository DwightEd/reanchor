"""Independent-source summaries and explicit-candidate reversal diagnostics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _interval(values: np.ndarray, bootstrap: int, seed: int) -> tuple[float, float]:
    if len(values) == 1 or bootstrap == 0:
        value = float(values.mean())
        return value, value
    random = np.random.default_rng(seed)
    draws = random.choice(values, size=(bootstrap, len(values)), replace=True).mean(1)
    lower, upper = np.quantile(draws, (0.025, 0.975))
    return float(lower), float(upper)


def source_ratio_summary(rows, numerator: str, *, bootstrap: int, seed: int) -> dict:
    """Average within-source ratios so prolific sources do not dominate."""

    by_source = defaultdict(list)
    for row in rows:
        by_source[str(row["source_id"])].append(bool(row[numerator]))
    if not by_source:
        return {"estimate": None, "ci95": [None, None], "sources": 0, "observations": 0}
    values = np.array([np.mean(items) for items in by_source.values()], dtype=float)
    lower, upper = _interval(values, bootstrap, seed)
    return {
        "estimate": float(values.mean()),
        "ci95": [lower, upper],
        "sources": len(values),
        "observations": sum(map(len, by_source.values())),
    }


def source_mean_summary(rows, value: str, *, bootstrap: int, seed: int) -> dict:
    by_source = defaultdict(list)
    for row in rows:
        number = float(row[value])
        if np.isfinite(number):
            by_source[str(row["source_id"])].append(number)
    if not by_source:
        return {"estimate": None, "ci95": [None, None], "sources": 0, "observations": 0}
    values = np.array([np.mean(items) for items in by_source.values()], dtype=float)
    lower, upper = _interval(values, bootstrap, seed)
    return {
        "estimate": float(values.mean()),
        "ci95": [lower, upper],
        "sources": len(values),
        "observations": sum(map(len, by_source.values())),
    }


def received_then_overridden(layer_response: np.ndarray) -> bool:
    """Detect a positive intermediate explicit-candidate effect ending nonpositive."""

    values = np.asarray(layer_response, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return False
    return bool(np.max(finite[:-1]) > 0 and finite[-1] <= 0)
