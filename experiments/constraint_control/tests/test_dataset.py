import json

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
                "messages": [{"role": "user", "content": "Who won under the 2020 rule?"}],
                "evidence_units": [
                    {"unit_id": "c1", "kind": "constraint", "text": "Use the 2020 rule."},
                    {"unit_id": "f1", "kind": "content", "text": "Ada won in 2020."},
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
    assert record.messages[0].content == "Who won under the 2020 rule?"
    assert [(unit.unit_id, unit.kind) for unit in record.evidence_units] == [
        ("c1", "constraint"),
        ("f1", "content"),
    ]
