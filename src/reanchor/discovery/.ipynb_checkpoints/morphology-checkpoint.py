"""Describe frozen reanchor anchors without changing event selection."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from .calibration import CalibratedSelection


@dataclass(frozen=True)
class MorphologyConfig:
    active_gain_floor: float = 0.05
    local_floor: float = 0.50
    broad_head_fraction: float = 0.50
    head_focality_threshold: float = 0.50
    focal_head_fraction: float = 0.50
    source_agreement: float = 0.50

    def __post_init__(self):
        values = tuple(getattr(self, field.name) for field in fields(self))
        if not all(0 <= value <= 1 for value in values):
            raise ValueError("morphology thresholds must be in [0, 1]")
        if min(self.broad_head_fraction, self.head_focality_threshold) == 0:
            raise ValueError("breadth and focality thresholds must be positive")


@dataclass(frozen=True)
class MorphologyProfile:
    reanchor_type: np.ndarray
    dominant_layer: np.ndarray
    active_head_fraction: np.ndarray
    all_layer_head_fraction: np.ndarray
    active_layer_fraction: np.ndarray
    head_focality: np.ndarray
    focal_head_fraction: np.ndarray
    effective_sources: np.ndarray
    source_agreement: np.ndarray
    gain_source_agreement: np.ndarray
    mean_distance: np.ndarray
    peak_prompt_fraction: np.ndarray
    peak_history_fraction: np.ndarray
    peak_evidence_fraction: np.ndarray
    dominant_source_position: np.ndarray
    dominant_source_token_id: np.ndarray
    dominant_source_token_text: np.ndarray

    def arrays(self) -> dict[str, np.ndarray]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid])) if valid.any() else np.nan


class MorphologyProfiler:
    def __init__(self, config: MorphologyConfig = MorphologyConfig()):
        self.config = config

    def run(self, features, selection: CalibratedSelection) -> MorphologyProfile:
        rows = len(features.eligible)
        kind = np.full(rows, "excluded", dtype="<U24")
        kind[features.eligible] = "none"
        kind[selection.significant] = "episode_member"

        def integer(value=-1):
            return np.full(rows, value, dtype=np.int32)

        def scalar():
            return np.full(rows, np.nan, dtype=np.float32)

        dominant_layer = integer()
        active_head_fraction = scalar()
        all_layer_head_fraction = scalar()
        active_layer_fraction = scalar()
        head_focality = scalar()
        focal_head_fraction = scalar()
        effective_sources = scalar()
        source_agreement = scalar()
        gain_source_agreement = scalar()
        mean_distance = scalar()
        peak_prompt_fraction = scalar()
        peak_history_fraction = scalar()
        peak_evidence_fraction = scalar()
        dominant_source_position = integer()

        active = (
            np.isfinite(features.remote_gain)
            & np.isfinite(features.previous_local_mass)
            & (features.remote_gain >= self.config.active_gain_floor)
            & (features.previous_local_mass >= self.config.local_floor)
        )
        for row in np.flatnonzero(selection.anchor):
            current = active[:, :, row]
            if not current.any():
                raise ValueError("a frozen anchor has no active read sites")
            gain = np.where(current, features.positive_remote_gain[:, :, row], 0)
            layer = int(gain.sum(1).argmax())
            selected = current[layer]
            weights = gain[layer, selected]
            peaks = features.peak_source[layer, selected, row]
            if np.any(peaks < 0) or features.special_mask[peaks].any():
                raise ValueError("active read sites require ordinary peak sources")

            dominant_layer[row] = layer
            active_head_fraction[row] = selected.mean()
            all_layer_head_fraction[row] = current.mean()
            active_layer_fraction[row] = current.any(1).mean()
            focality = features.gain_focality[layer, selected, row]
            head_focality[row] = _weighted_mean(focality, weights)
            focal_head_fraction[row] = np.mean(focality >= self.config.head_focality_threshold)
            effective_sources[row] = _weighted_mean(
                features.effective_sources[layer, selected, row], weights
            )
            mean_distance[row] = _weighted_mean(
                features.mean_distance[layer, selected, row], weights
            )
            votes = np.bincount(peaks, minlength=len(features.special_mask))
            gain_votes = np.bincount(peaks, weights=weights, minlength=len(features.special_mask))
            dominant_source_position[row] = int(votes.argmax())
            source_agreement[row] = votes.max() / votes.sum()
            gain_source_agreement[row] = gain_votes.max() / gain_votes.sum()
            peak_prompt_fraction[row] = votes[: features.response_start].sum() / votes.sum()
            peak_history_fraction[row] = votes[features.response_start :].sum() / votes.sum()
            peak_evidence_fraction[row] = votes[features.evidence_mask].sum() / votes.sum()
            kind[row] = self._type(
                active_head_fraction[row],
                focal_head_fraction[row],
                source_agreement[row],
            )

        source_id = np.full(rows, -1, dtype=np.int64)
        source_text = np.full(rows, "", dtype=features.token_text.dtype)
        present = dominant_source_position >= 0
        source_id[present] = features.token_ids[dominant_source_position[present]]
        source_text[present] = features.token_text[dominant_source_position[present]]
        return MorphologyProfile(
            kind,
            dominant_layer,
            active_head_fraction,
            all_layer_head_fraction,
            active_layer_fraction,
            head_focality,
            focal_head_fraction,
            effective_sources,
            source_agreement,
            gain_source_agreement,
            mean_distance,
            peak_prompt_fraction,
            peak_history_fraction,
            peak_evidence_fraction,
            dominant_source_position,
            source_id,
            source_text,
        )

    def _type(self, breadth: float, focal_fraction: float, agreement: float) -> str:
        broad = breadth >= self.config.broad_head_fraction
        focal = focal_fraction >= self.config.focal_head_fraction
        if not broad:
            return "sparse_focal" if focal else "sparse_diffuse"
        if not focal:
            return "broad_diffuse"
        if agreement >= self.config.source_agreement:
            return "broad_convergent"
        return "broad_diverse"
