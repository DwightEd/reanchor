"""Collapse layer/head measurements into predeclared token statistics."""

from __future__ import annotations

import math

import numpy as np


def score_transitions(
    remote_gain: np.ndarray,
    previous_local_mass: np.ndarray,
    *,
    eligible: np.ndarray,
    broad_head_fraction: float,
    local_floor: float,
    site_gain_floor: float,
) -> dict[str, np.ndarray]:
    """Return sparse and broad token scores before null calibration."""

    gain = np.asarray(remote_gain, dtype=np.float32)
    previous_local = np.asarray(previous_local_mass, dtype=np.float32)
    eligible = np.asarray(eligible, dtype=bool)
    if gain.ndim != 3 or previous_local.shape != gain.shape:
        raise ValueError("gain arrays must have [layer, head, row] axes")
    if eligible.shape != (gain.shape[2],):
        raise ValueError("eligible must share the row axis")
    if not 0 < broad_head_fraction <= 1:
        raise ValueError("broad_head_fraction must be in (0, 1]")

    gate = (
        eligible[None, None]
        & np.isfinite(gain)
        & np.isfinite(previous_local)
        & (previous_local >= local_floor)
    )
    positive = np.where(gate, np.maximum(gain, 0), 0)
    sparse = positive.max((0, 1))
    broad_heads = max(1, math.ceil(gain.shape[1] * broad_head_fraction))
    top = np.sort(positive, axis=1)[:, -broad_heads:]
    broad = top.mean(1).max(0)
    support = (positive >= site_gain_floor).sum((0, 1), dtype=np.int32)
    return {
        "sparse_score": sparse.astype(np.float32),
        "broad_score": broad.astype(np.float32),
        "supporting_sites": support,
    }
