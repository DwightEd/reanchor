"""Free-run generation, replay validation, and trajectory persistence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Protocol
from urllib.parse import quote

import numpy as np

from reanchor.artifacts.store import ArtifactStore

from .records import ChatMessage, SourceRecord


@dataclass(frozen=True)
class SamplingConfig:
    seed: int = 0
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 20
    trace_top_k: int = 20

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be nonnegative")
        if self.trace_top_k < 1:
            raise ValueError("trace_top_k must be positive")


@dataclass(frozen=True)
class CapturedGeneration:
    """Model outputs needed to reproduce and inspect one sampled trajectory."""

    token_ids: np.ndarray
    token_text: tuple[str, ...]
    special_mask: np.ndarray
    response_start: int
    response_text: str
    generation_selected_logits: np.ndarray
    replay_selected_logits: np.ndarray
    model_logprobs: np.ndarray
    sampling_logprobs: np.ndarray
    entropy: np.ndarray
    top_token_ids: np.ndarray
    top_logits: np.ndarray
    residual_states: np.ndarray
    attention_weights: np.ndarray
    replay_max_abs_logit_error: float
    stop_reason: str

    def __post_init__(self) -> None:
        token_count = len(self.token_ids)
        response_tokens = token_count - self.response_start
        if not 0 <= self.response_start < token_count:
            raise ValueError("response_start must point inside token_ids")
        if len(self.token_text) != token_count:
            raise ValueError("token_text and token_ids must have equal length")
        if self.special_mask.shape != (token_count,):
            raise ValueError("special_mask must match token_ids")
        vectors = (
            self.generation_selected_logits,
            self.replay_selected_logits,
            self.model_logprobs,
            self.sampling_logprobs,
            self.entropy,
        )
        if any(value.shape != (response_tokens,) for value in vectors):
            raise ValueError("per-token capture vectors must match the response length")
        if self.top_token_ids.shape != self.top_logits.shape:
            raise ValueError("top_token_ids and top_logits must have equal shape")
        if self.top_token_ids.ndim != 2 or self.top_token_ids.shape[0] != response_tokens:
            raise ValueError("top-token arrays must have shape [response, k]")
        if self.residual_states.ndim != 3 or self.residual_states.shape[1] != response_tokens:
            raise ValueError("residual_states must have shape [layers, response, hidden]")
        if self.attention_weights.ndim != 4 or self.attention_weights.shape[2] != response_tokens:
            raise ValueError(
                "attention_weights must have shape [layers, heads, response, sequence]"
            )
        if self.replay_max_abs_logit_error < 0:
            raise ValueError("replay error must be nonnegative")
        if self.stop_reason not in {"eos", "max_new_tokens"}:
            raise ValueError("stop_reason must be eos or max_new_tokens")


class GenerationBackend(Protocol):
    @property
    def metadata(self) -> Mapping[str, str]: ...

    def sample_and_replay(
        self, messages: tuple[ChatMessage, ...], sampling: SamplingConfig
    ) -> CapturedGeneration: ...


@dataclass(frozen=True)
class TrajectoryArtifact:
    sample_key: str
    response_tokens: int
    capture_path: Path
    metadata_path: Path


class ReplayMismatchError(RuntimeError):
    pass


class GenerationRecorder:
    """Create one validated, label-free trajectory artifact."""

    def __init__(
        self,
        backend: GenerationBackend,
        output: str | Path,
        *,
        replay_atol: float = 0.05,
    ):
        if replay_atol < 0:
            raise ValueError("replay_atol must be nonnegative")
        self.backend = backend
        self.store = ArtifactStore(output)
        self.replay_atol = replay_atol

    def run(self, record: SourceRecord, sampling: SamplingConfig) -> TrajectoryArtifact:
        capture = self.backend.sample_and_replay(record.messages, sampling)
        if capture.replay_max_abs_logit_error > self.replay_atol:
            raise ReplayMismatchError(
                f"{record.key}: replay max logit error "
                f"{capture.replay_max_abs_logit_error:.6g} exceeds {self.replay_atol:.6g}"
            )

        folder = self.store.root / "trajectories"
        for part in (record.split, record.task, record.sample_id, f"seed_{sampling.seed}"):
            folder /= quote(part, safe="")
        capture_path = folder / "capture.npz"
        metadata_path = folder / "trajectory.json"
        response_tokens = len(capture.token_ids) - capture.response_start
        row_position = np.arange(
            capture.response_start - 1,
            len(capture.token_ids) - 1,
            dtype=np.int64,
        )
        self.store.write_npz(
            capture_path,
            token_ids=np.asarray(capture.token_ids, dtype=np.int64),
            token_text=np.asarray(capture.token_text),
            special_mask=np.asarray(capture.special_mask, dtype=bool),
            response_start=np.asarray(capture.response_start, dtype=np.int64),
            row_position=row_position,
            generation_selected_logits=capture.generation_selected_logits,
            replay_selected_logits=capture.replay_selected_logits,
            model_logprobs=capture.model_logprobs,
            sampling_logprobs=capture.sampling_logprobs,
            entropy=capture.entropy,
            top_token_ids=capture.top_token_ids,
            top_logits=capture.top_logits,
            residual_states=capture.residual_states,
            attention_weights=capture.attention_weights,
        )
        identity = self._identity(record, sampling)
        self.store.write_json(
            metadata_path,
            {
                "schema": "constraint_control_trajectory_v1",
                "identity": identity,
                "sample_key": record.key,
                "source_id": record.source_id,
                "labels_used_for_capture": False,
                "messages": [asdict(message) for message in record.messages],
                "evidence_units": [asdict(unit) for unit in record.evidence_units],
                "sampling": asdict(sampling),
                "model": dict(self.backend.metadata),
                "response_text": capture.response_text,
                "response_tokens": response_tokens,
                "stop_reason": capture.stop_reason,
                "replay_max_abs_logit_error": capture.replay_max_abs_logit_error,
                "replay_atol": self.replay_atol,
                "capture": capture_path.name,
            },
        )
        return TrajectoryArtifact(record.key, response_tokens, capture_path, metadata_path)

    def _identity(self, record: SourceRecord, sampling: SamplingConfig) -> str:
        value = {
            "sample_key": record.key,
            "source_id": record.source_id,
            "messages": [asdict(message) for message in record.messages],
            "evidence_units": [asdict(unit) for unit in record.evidence_units],
            "sampling": asdict(sampling),
            "model": dict(self.backend.metadata),
        }
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
