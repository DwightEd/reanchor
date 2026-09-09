"""Input records for free-running constraint-control experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError("message role must be system, user, or assistant")
        if not self.content:
            raise ValueError("message content must be nonempty")


@dataclass(frozen=True)
class EvidenceUnit:
    unit_id: str
    kind: str
    text: str
    message_index: int
    char_start: int
    char_end: int

    def __post_init__(self) -> None:
        if not self.unit_id or not self.text:
            raise ValueError("evidence unit_id and text must be nonempty")
        if self.kind not in {"constraint", "content", "other"}:
            raise ValueError("evidence kind must be constraint, content, or other")
        if self.message_index < 0:
            raise ValueError("evidence message_index must be nonnegative")
        if self.char_start < 0 or self.char_end <= self.char_start:
            raise ValueError("evidence character span must be nonempty")


@dataclass(frozen=True)
class SourceRecord:
    """One source-grouped prompt; outcome labels are deliberately absent."""

    sample_id: str
    source_id: str
    split: str
    task: str
    messages: tuple[ChatMessage, ...]
    evidence_units: tuple[EvidenceUnit, ...] = ()

    def __post_init__(self) -> None:
        values = (self.sample_id, self.source_id, self.split, self.task)
        if any(not value for value in values):
            raise ValueError("sample_id, source_id, split, and task must be nonempty")
        if not self.messages:
            raise ValueError("a source record must contain at least one message")
        unit_ids = [unit.unit_id for unit in self.evidence_units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("evidence unit_id values must be unique within a record")
        for unit in self.evidence_units:
            if unit.message_index >= len(self.messages):
                raise ValueError(f"evidence {unit.unit_id} refers to a missing message")
            content = self.messages[unit.message_index].content
            if content[unit.char_start : unit.char_end] != unit.text:
                raise ValueError(f"evidence {unit.unit_id} text does not match its character span")

    @property
    def key(self) -> str:
        return f"{self.split}/{self.task}/{self.sample_id}"
