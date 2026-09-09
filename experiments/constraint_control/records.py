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
class SourceRecord:
    """One source-grouped prompt; outcome labels are deliberately absent."""

    sample_id: str
    source_id: str
    split: str
    task: str
    messages: tuple[ChatMessage, ...]

    def __post_init__(self) -> None:
        values = (self.sample_id, self.source_id, self.split, self.task)
        if any(not value for value in values):
            raise ValueError("sample_id, source_id, split, and task must be nonempty")
        if not self.messages:
            raise ValueError("a source record must contain at least one message")

    @property
    def key(self) -> str:
        return f"{self.split}/{self.task}/{self.sample_id}"
