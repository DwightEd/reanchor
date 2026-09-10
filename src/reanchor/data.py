"""Input contract for paired relation-control events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

INPUT_SCHEMA = "reanchor/counterfactual-event@1"
INPUT_FIELDS = {
    "schema",
    "event_id",
    "source_id",
    "split",
    "relation",
    "prompt_a",
    "prompt_b",
    "answer_prefix",
    "option_a",
    "option_b",
    "followup",
}


@dataclass(frozen=True)
class CounterfactualEvent:
    """Two prompts that differ only in one relation value."""

    event_id: str
    source_id: str
    split: str
    relation: str
    prompt_a: str
    prompt_b: str
    answer_prefix: str
    option_a: str
    option_b: str
    followup: str

    def __post_init__(self) -> None:
        values = (
            self.event_id,
            self.source_id,
            self.split,
            self.relation,
            self.prompt_a,
            self.prompt_b,
            self.answer_prefix,
            self.option_a,
            self.option_b,
            self.followup,
        )
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError("event fields must be nonempty strings")
        if self.prompt_a == self.prompt_b:
            raise ValueError("prompt_a and prompt_b must be counterfactual worlds")
        if self.option_a == self.option_b:
            raise ValueError("option_a and option_b must differ")


def load_events(path: Path) -> tuple[CounterfactualEvent, ...]:
    """Load and validate the experiment JSONL once at the file boundary."""

    if not path.is_file():
        raise FileNotFoundError(f"event JSONL does not exist: {path}")

    events = []
    seen_ids = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            unexpected = set(record).difference(INPUT_FIELDS)
            missing = INPUT_FIELDS.difference(record)
            if unexpected or missing:
                raise ValueError(
                    f"{path}:{line_number} has unexpected fields {sorted(unexpected)} "
                    f"or missing fields {sorted(missing)}"
                )
            if record["schema"] != INPUT_SCHEMA:
                raise ValueError(f"{path}:{line_number} has an unsupported schema")
            event = CounterfactualEvent(
                **{name: value for name, value in record.items() if name != "schema"}
            )
            if event.event_id in seen_ids:
                raise ValueError(f"duplicate event id: {event.event_id}")
            seen_ids.add(event.event_id)
            events.append(event)

    if not events:
        raise ValueError(f"event JSONL is empty: {path}")
    return tuple(events)
