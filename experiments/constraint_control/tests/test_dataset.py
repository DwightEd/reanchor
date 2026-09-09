import json

import pytest

from experiments.constraint_control.dataset import SourceDataset


def test_dataset_reads_messages_and_typed_evidence_units(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps(
            {
                "sample_id": "q1",
                "source_id": "document-1",
                "split": "discovery",
                "task": "QA",
                "messages": [
                    {
                        "role": "user",
                        "content": "Use the 2020 rule. Ada won in 2020. Who won?",
                    }
                ],
                "evidence_units": [
                    {
                        "unit_id": "c1",
                        "kind": "constraint",
                        "text": "Use the 2020 rule.",
                        "message_index": 0,
                        "char_start": 0,
                        "char_end": 18,
                    },
                    {
                        "unit_id": "f1",
                        "kind": "content",
                        "text": "Ada won in 2020.",
                        "message_index": 0,
                        "char_start": 19,
                        "char_end": 35,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = SourceDataset(path)

    assert len(dataset.records) == 1
    record = dataset.records[0]
    assert record.key == "discovery/QA/q1"
    assert record.messages[0].content.endswith("Who won?")
    assert [(unit.unit_id, unit.kind) for unit in record.evidence_units] == [
        ("c1", "constraint"),
        ("f1", "content"),
    ]


def test_dataset_rejects_the_same_source_across_experimental_splits(tmp_path):
    path = tmp_path / "questions.jsonl"
    records = [
        {
            "sample_id": sample_id,
            "source_id": "shared-document",
            "split": split,
            "task": "QA",
            "messages": [{"role": "user", "content": "Question"}],
        }
        for sample_id, split in (("q1", "discovery"), ("q2", "confirmation"))
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source_id appears in multiple splits"):
        SourceDataset(path)
