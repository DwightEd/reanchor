"""Extract reusable sample transitions from complete captured attention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reanchor.capture.protocol import AuditSample

from .calibration import TokenScoreRecord
from .features import SITE_METRICS, measure_attention_transitions
from .scores import score_transitions


@dataclass(frozen=True)
class TransitionConfig:
    window: int = 10
    local_floor: float = 0.50
    site_gain_floor: float = 0.10
    broad_head_fraction: float = 0.25
    position_bins: int = 4

    def __post_init__(self):
        if self.window < 1 or self.position_bins < 1:
            raise ValueError("window and position_bins must be positive")
        fractions = (self.local_floor, self.site_gain_floor, self.broad_head_fraction)
        if not all(0 <= value <= 1 for value in fractions):
            raise ValueError("transition fractions must be in [0, 1]")
        if self.broad_head_fraction == 0:
            raise ValueError("broad_head_fraction must be positive")


@dataclass(frozen=True)
class TransitionFeatures:
    sample: AuditSample
    token_ids: np.ndarray
    token_text: np.ndarray
    row_position: np.ndarray
    response_start: int
    special_mask: np.ndarray
    evidence_mask: np.ndarray
    eligible: np.ndarray
    current_local_mass: np.ndarray
    current_remote_mass: np.ndarray
    previous_local_mass: np.ndarray
    remote_gain: np.ndarray
    positive_remote_gain: np.ndarray
    time_tv: np.ndarray
    gain_focality: np.ndarray
    effective_sources: np.ndarray
    mean_distance: np.ndarray
    prompt_positive_gain: np.ndarray
    evidence_positive_gain: np.ndarray
    history_positive_gain: np.ndarray
    peak_source: np.ndarray

    @classmethod
    def from_arrays(cls, sample: AuditSample, values: dict[str, np.ndarray]):
        metrics = {name: values[name] for name in SITE_METRICS}
        return cls(
            sample=sample,
            token_ids=values["token_ids"],
            token_text=values["token_text"],
            row_position=values["row_position"],
            response_start=int(values["response_start"]),
            special_mask=values["special_mask"],
            evidence_mask=values["evidence_mask"],
            eligible=values["eligible"],
            peak_source=values["peak_source"],
            **metrics,
        )

    def token_scores(self, config: TransitionConfig) -> TokenScoreRecord:
        values = score_transitions(
            self.remote_gain,
            self.previous_local_mass,
            eligible=self.eligible,
            broad_head_fraction=config.broad_head_fraction,
            local_floor=config.local_floor,
            site_gain_floor=config.site_gain_floor,
        )
        response_tokens = max(1, len(self.token_ids) - self.response_start)
        relative = np.maximum(self.row_position - self.response_start, 0)
        position_bin = np.minimum(
            relative * config.position_bins // response_tokens,
            config.position_bins - 1,
        ).astype(np.int16)
        return TokenScoreRecord(
            split=self.sample.split,
            task=self.sample.task,
            sample_id=self.sample.sample_id,
            source_id=self.sample.source_id,
            row_position=self.row_position,
            eligible=self.eligible,
            position_bin=position_bin,
            **values,
        )

    def arrays(self) -> dict[str, np.ndarray]:
        result = {
            "schema": np.array(1),
            "token_ids": self.token_ids,
            "token_text": self.token_text,
            "row_position": self.row_position,
            "response_start": np.array(self.response_start),
            "special_mask": self.special_mask,
            "evidence_mask": self.evidence_mask,
            "eligible": self.eligible,
            "peak_source": self.peak_source,
        }
        result.update({name: getattr(self, name) for name in SITE_METRICS})
        return result


class TransitionExtractor:
    """Convert streamed attention layers into one label-free sample artifact."""

    _METADATA = (
        "token_ids",
        "token_text",
        "row_position",
        "response_start",
        "special_mask",
        "evidence_mask",
    )

    def __init__(self, attention_reader, config: TransitionConfig = TransitionConfig()):
        self.reader = attention_reader
        self.config = config

    def run(self, sample: AuditSample) -> TransitionFeatures:
        metadata = self.reader.dataset.load_metadata(sample, *self._METADATA)
        layers = []
        for expected_layer, (layer, attention) in enumerate(self.reader.iter_layers(sample)):
            if layer != expected_layer:
                raise ValueError(f"{sample.key}: attention layers must be contiguous")
            layers.append(
                measure_attention_transitions(
                    attention,
                    rows=metadata["row_position"],
                    response_start=int(metadata["response_start"]),
                    special_mask=metadata["special_mask"],
                    evidence_mask=metadata["evidence_mask"],
                    window=self.config.window,
                )
            )
        if not layers:
            raise ValueError(f"{sample.key}: capture contains no attention layers")
        eligible = layers[0]["eligible"]
        if any(not np.array_equal(layer["eligible"], eligible) for layer in layers[1:]):
            raise ValueError(f"{sample.key}: layer eligibility is inconsistent")
        metrics = {name: np.stack([layer[name] for layer in layers]) for name in SITE_METRICS}
        return TransitionFeatures(
            sample=sample,
            token_ids=metadata["token_ids"],
            token_text=metadata["token_text"],
            row_position=metadata["row_position"],
            response_start=int(metadata["response_start"]),
            special_mask=metadata["special_mask"],
            evidence_mask=metadata["evidence_mask"],
            eligible=eligible,
            peak_source=np.stack([layer["peak_source"] for layer in layers]),
            **metrics,
        )
