"""Read immutable attention-audit v3 datasets through one validated interface."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AuditSample:
    split: str
    task: str
    sample_id: str
    source_id: str
    relative_path: Path
    response_tokens: int
    response_start: int

    @property
    def key(self) -> str:
        return f"{self.split}/{self.task}/{self.sample_id}"


@dataclass(frozen=True)
class CapturePaths:
    compact: Path
    qk: Path
    history: Path
    states: Path
    labels: Path
    attention: Path


class AuditDataset:
    """Validate a v3 capture once and expose only complete immutable samples."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        if self.manifest.get("audit_schema") != 3:
            raise ValueError("reanchor requires an attention_audit_v3 capture")
        if self.manifest.get("labels_used_for_capture") is not False:
            raise ValueError("capture must declare labels_used_for_capture=false")
        self.samples = tuple(self._sample(entry) for entry in self.manifest.get("samples", ()))
        if not self.samples:
            raise ValueError("capture manifest contains no samples")
        self._validate_samples()

    @property
    def model_path(self) -> Path:
        value = self.manifest.get("settings", {}).get("model")
        if not value:
            raise ValueError("capture manifest does not identify its checkpoint")
        return Path(value)

    def paths(self, sample: AuditSample) -> CapturePaths:
        compact = (self.root / sample.relative_path).resolve()
        self._inside_root(compact)
        return CapturePaths(
            compact=compact,
            qk=compact.with_suffix(".qk.npz"),
            history=compact.with_suffix(".history.npz"),
            states=compact.with_suffix(".states.npz"),
            labels=compact.with_suffix(".labels.npz"),
            attention=compact.with_suffix(".attention.npz"),
        )

    def completed_samples(
        self, *, require_states: bool
    ) -> tuple[tuple[AuditSample, ...], dict[str, int]]:
        completed = []
        for sample in self.samples:
            paths = self.paths(sample)
            required = [paths.compact, paths.qk, paths.history]
            if require_states:
                required.append(paths.states)
            if all(path.is_file() for path in required):
                completed.append(sample)
        coverage = {
            "planned": len(self.samples),
            "completed": len(completed),
            "skipped": len(self.samples) - len(completed),
        }
        return tuple(completed), coverage

    def load_metadata(self, sample: AuditSample, *fields: str) -> dict[str, np.ndarray]:
        with np.load(self.paths(sample).compact, allow_pickle=False) as archive:
            missing = [field for field in fields if field not in archive]
            if missing:
                raise ValueError(f"{sample.key}: missing capture fields {missing}")
            return {field: archive[field] for field in fields}

    def load_available_metadata(
        self, sample: AuditSample, *fields: str
    ) -> dict[str, np.ndarray]:
        """Return optional capture annotations without inventing fallback values."""

        with np.load(self.paths(sample).compact, allow_pickle=False) as archive:
            return {field: archive[field] for field in fields if field in archive}

    def _sample(self, entry: dict) -> AuditSample:
        relative = Path(str(entry["path"]))
        self._inside_root((self.root / relative).resolve())
        return AuditSample(
            split=str(entry["split"]),
            task=str(entry["task_type"]),
            sample_id=str(entry["sample_id"]),
            source_id=str(entry["source_id"]),
            relative_path=relative,
            response_tokens=int(entry["response_tokens"]),
            response_start=int(entry["response_start"]),
        )

    def _inside_root(self, path: Path) -> None:
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError(f"capture path is outside capture root: {path}") from error

    def _validate_samples(self) -> None:
        keys = [sample.key for sample in self.samples]
        if len(keys) != len(set(keys)):
            raise ValueError("capture manifest contains duplicate sample keys")
        splits_by_source: dict[str, set[str]] = {}
        for sample in self.samples:
            if not sample.source_id:
                raise ValueError(f"{sample.key}: source_id must be nonempty")
            splits_by_source.setdefault(sample.source_id, set()).add(sample.split)
        leaked = sorted(source for source, splits in splits_by_source.items() if len(splits) > 1)
        if leaked:
            raise ValueError(f"source_id appears in multiple splits: {leaked[:5]}")
