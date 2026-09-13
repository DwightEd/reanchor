"""Map frozen token anchors to the read sites that seed causal tracing."""

from __future__ import annotations

import numpy as np


def anchor_coordinates(
    remote_gain: np.ndarray,
    previous_local_mass: np.ndarray,
    anchor: np.ndarray,
    *,
    active_gain_floor: float,
    local_floor: float,
) -> np.ndarray:
    """Return sorted ``(layer, head, response-row)`` causal seed coordinates."""

    gain = np.asarray(remote_gain)
    previous = np.asarray(previous_local_mass)
    anchors = np.asarray(anchor, dtype=bool)
    if gain.ndim != 3 or previous.shape != gain.shape:
        raise ValueError("read-site arrays must have shape [layer, head, response-row]")
    if anchors.shape != (gain.shape[-1],):
        raise ValueError("anchor mask must have one value per response row")
    active = (
        np.isfinite(gain)
        & np.isfinite(previous)
        & (gain >= active_gain_floor)
        & (previous >= local_floor)
        & anchors[None, None, :]
    )
    coordinates = np.argwhere(active).astype(np.int32, copy=False)
    missing = np.flatnonzero(anchors & ~active.any(axis=(0, 1)))
    if len(missing):
        raise ValueError(f"anchor rows have no active read site: {missing.tolist()}")
    return coordinates
