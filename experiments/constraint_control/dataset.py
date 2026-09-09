"""Read and validate source-grouped experiment inputs."""

from __future__ import annotations

import json
from pathlib import Path

from .records import ChatMessage, EvidenceUnit, SourceRecord


class SourceDataset:
    """Load immutable JSONL records through one validated interface."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        lines = self.path.read_text(encoding="utf-8").splitlines()
        self.records = tuple(
            self._parse(json.loads(line)) for line in lines if line.strip()
        )
        if not self.records:
            raise ValueError("source dataset contains no records")

    def _parse(self, value: dict) -> SourceRecord:
        messages = tuple(ChatMessage(**message) for message in value["messages"])
        evidence = tuple(
            EvidenceUnit(**unit) for unit in value.get("evidence_units", ())
        )
        return SourceRecord(
            sample_id=str(value["sample_id"]),
            source_id=str(value["source_id"]),
            split=str(value["split"]),
            task=str(value["task"]),
            messages=messages,
            evidence_units=evidence,
        )
