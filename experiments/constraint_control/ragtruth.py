"""Label-free adapter from the original RAGTruth files to capture records."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .records import ChatMessage, EvidenceUnit, SourceRecord

SYSTEM_PROMPT = "You are a helpful assistant."
TASKS = ("QA", "Summary", "Data2txt")


class RagTruthDataset:
    """Read RAGTruth prompts while keeping annotated responses out of capture."""

    def __init__(self, root: str | Path, *, task: str = "QA", split: str = "all"):
        if task not in (*TASKS, "all"):
            raise ValueError(f"task must be one of {(*TASKS, 'all')}")
        if split not in {"train", "test", "all"}:
            raise ValueError("split must be train, test, or all")
        self.root = Path(root)
        source_path = self.root / "source_info.jsonl"
        response_path = self.root / "response.jsonl"
        if not source_path.is_file():
            raise FileNotFoundError(f"RAGTruth source file does not exist: {source_path}")
        if not response_path.is_file():
            raise FileNotFoundError(f"RAGTruth response file does not exist: {response_path}")

        splits = _load_source_splits(response_path)
        records = []
        source_ids = set()
        for source in _read_jsonl(source_path):
            source_task = _task_type(source.get("task_type"))
            if task != "all" and source_task != task:
                continue
            source_id = str(source["source_id"])
            if source_id in source_ids:
                raise ValueError(f"duplicate RAGTruth source_id: {source_id}")
            source_ids.add(source_id)
            if source_id not in splits:
                raise ValueError(f"RAGTruth source has no official split: {source_id}")
            source_split = splits[source_id]
            if split != "all" and source_split != split:
                continue
            records.append(_source_record(source, source_task, source_split))
        if not records:
            raise ValueError(f"RAGTruth contains no records for task={task}, split={split}")
        self.records = tuple(records)


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield value


def _load_source_splits(path: Path) -> dict[str, str]:
    """Read only source membership; response text and annotations are discarded."""

    splits: dict[str, str] = {}
    for response in _read_jsonl(path):
        source_id = str(response["source_id"])
        split = str(response["split"])
        if split not in {"train", "test"}:
            raise ValueError(f"unsupported RAGTruth split: {split}")
        previous = splits.setdefault(source_id, split)
        if previous != split:
            raise ValueError(f"source_id {source_id} appears in multiple RAGTruth splits")
    return splits


def _task_type(value: Any) -> str:
    for task in TASKS:
        if str(value).casefold() == task.casefold():
            return task
    raise ValueError(f"unsupported RAGTruth task_type: {value}")


def _source_record(source: dict[str, Any], task: str, split: str) -> SourceRecord:
    prompt = str(source["prompt"])
    return SourceRecord(
        sample_id=str(source["source_id"]),
        source_id=str(source["source_id"]),
        split=split,
        task=task,
        messages=(
            ChatMessage("system", SYSTEM_PROMPT),
            ChatMessage("user", prompt),
        ),
        evidence_units=_evidence_units(source, task, prompt),
    )


def _evidence_units(source: dict[str, Any], task: str, prompt: str) -> tuple[EvidenceUnit, ...]:
    information = source["source_info"]
    if task == "QA":
        question = str(information["question"])
        passages = str(information["passages"]).rstrip()
        content_start = _unique_start(prompt, passages, "QA passages")
        content_spans = _paragraph_spans(prompt, content_start, passages)
        question_start = _unique_start(prompt[:content_start], question, "QA question")
        constraints = [("question", question_start, question_start + len(question))]
    elif task == "Summary":
        article = str(information).rstrip()
        content_start = _unique_start(prompt, article, "summary article")
        content_spans = [("article", content_start, content_start + len(article))]
        constraints = []
    else:
        content_start, content_end = _structured_data_span(prompt)
        content_spans = [("structured_data", content_start, content_end)]
        constraints = []

    content_end = max(end for _name, _start, end in content_spans)
    occupied = sorted(
        [(start, end) for _name, start, end in content_spans]
        + [(start, end) for _name, start, end in constraints]
    )
    constraint_spans = [
        *constraints,
        *(
            (f"instruction_{index}", start, end)
            for index, (start, end) in enumerate(
                _complement_spans(prompt, occupied, content_start, content_end),
                start=1,
            )
        ),
    ]
    descriptors = [
        (f"constraint:{name}", "constraint", start, end) for name, start, end in constraint_spans
    ]
    descriptors.extend(
        (f"content:{name}", "content", start, end) for name, start, end in content_spans
    )
    return tuple(
        _unit(unit_id, kind, prompt, start, end)
        for unit_id, kind, start, end in sorted(descriptors, key=lambda value: value[2])
    )


def _unique_start(text: str, value: str, name: str) -> int:
    start = text.find(value)
    if start < 0:
        raise ValueError(f"RAGTruth {name} is not an exact substring of prompt")
    if text.find(value, start + 1) >= 0:
        raise ValueError(f"RAGTruth {name} is not unique in prompt")
    return start


def _paragraph_spans(prompt: str, start: int, passages: str):
    spans = []
    for index, match in enumerate(re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", passages, re.S), 1):
        spans.append((f"passage_{index}", start + match.start(), start + match.end()))
    if not spans:
        raise ValueError("RAGTruth QA passages contain no nonempty passage")
    return spans


def _structured_data_span(prompt: str) -> tuple[int, int]:
    marker = "Structured data:"
    marker_start = prompt.find(marker)
    if marker_start < 0:
        raise ValueError("RAGTruth Data2txt prompt has no Structured data marker")
    start = marker_start + len(marker)
    while start < len(prompt) and prompt[start].isspace():
        start += 1
    end = prompt.rfind("\nOverview:")
    if end <= start:
        raise ValueError("RAGTruth Data2txt prompt has no Overview boundary")
    return start, end


def _complement_spans(
    prompt: str,
    occupied: list[tuple[int, int]],
    content_start: int,
    content_end: int,
) -> list[tuple[int, int]]:
    """Return non-content prompt instructions around the selected evidence."""

    boundaries = [0, *[value for span in occupied for value in span], len(prompt)]
    spans = []
    for start, end in zip(boundaries[::2], boundaries[1::2], strict=True):
        if start >= content_start and end <= content_end:
            continue
        trimmed = _trim_span(prompt, start, end)
        if trimmed is not None:
            spans.append(trimmed)
    return spans


def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _unit(unit_id: str, kind: str, prompt: str, start: int, end: int) -> EvidenceUnit:
    return EvidenceUnit(
        unit_id=unit_id,
        kind=kind,
        text=prompt[start:end],
        message_index=1,
        char_start=start,
        char_end=end,
    )
