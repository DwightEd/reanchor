"""Own the on-disk layout and atomic publication of run artifacts."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import numpy as np

from reanchor.capture.protocol import AuditSample


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def sample_path(self, sample: AuditSample, name: str) -> Path:
        parts = (sample.split, sample.task, sample.sample_id)
        folder = self.root / "samples"
        for part in parts:
            folder /= quote(str(part), safe="")
        return folder / name

    def write_npz(self, path: Path, **arrays) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._temporary(path, ".tmp.npz") as temporary:
            np.savez_compressed(temporary, **arrays)

    def read_npz(self, path: Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}

    def write_json(self, path: Path, value) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._temporary(path, ".tmp") as temporary:
            temporary.write_text(
                json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )

    @contextmanager
    def _temporary(self, destination: Path, suffix: str):
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}{suffix}")
        try:
            yield temporary
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
