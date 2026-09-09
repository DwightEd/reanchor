import json

import pytest

from experiments.constraint_control.ragtruth import RagTruthDataset


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_ragtruth_dataset_builds_label_free_qa_records(tmp_path):
    question = "Which year applies?"
    passages = "passage 1:The 2020 rule applies.\n\npassage 2:The 2021 rule expired.\n\n"
    prompt = (
        "Briefly answer the following question:\n"
        f"{question}\n"
        "Use only the following passages:\n"
        f"{passages}"
        "If the answer is absent, say unable to answer.\n"
        "output:"
    )
    _write_jsonl(
        tmp_path / "source_info.jsonl",
        [
            {
                "source_id": "source-1",
                "task_type": "QA",
                "source": "MARCO",
                "source_info": {"question": question, "passages": passages},
                "prompt": prompt,
            }
        ],
    )
    _write_jsonl(
        tmp_path / "response.jsonl",
        [
            {
                "id": "old-response",
                "source_id": "source-1",
                "split": "test",
                "response": "POISON OLD ANSWER",
                "labels": [{"text": "POISON LABEL", "start": 0, "end": 6}],
            }
        ],
    )

    record = RagTruthDataset(tmp_path, task="QA").records[0]

    assert record.sample_id == "source-1"
    assert record.split == "test"
    assert record.task == "QA"
    assert [message.role for message in record.messages] == ["system", "user"]
    assert record.messages[1].content == prompt
    assert "POISON" not in repr(record)
    assert any(
        unit.kind == "constraint" and unit.text == question for unit in record.evidence_units
    )
    content = [unit for unit in record.evidence_units if unit.kind == "content"]
    assert [unit.text for unit in content] == [
        "passage 1:The 2020 rule applies.",
        "passage 2:The 2021 rule expired.",
    ]


def test_ragtruth_dataset_filters_task_and_official_split_before_capture(tmp_path):
    _write_jsonl(
        tmp_path / "source_info.jsonl",
        [
            {
                "source_id": "summary-1",
                "task_type": "Summary",
                "source": "CNN/DM",
                "source_info": "Long article.",
                "prompt": "Summarize:\nLong article.\noutput:",
            },
            {
                "source_id": "qa-1",
                "task_type": "QA",
                "source": "MARCO",
                "source_info": {"question": "Q?", "passages": "passage 1:A\n\n"},
                "prompt": "Question: Q?\npassage 1:A\n\noutput:",
            },
            {
                "source_id": "qa-2",
                "task_type": "QA",
                "source": "MARCO",
                "source_info": {"question": "Q2?", "passages": "passage 1:B\n\n"},
                "prompt": "Question: Q2?\npassage 1:B\n\noutput:",
            },
        ],
    )
    _write_jsonl(
        tmp_path / "response.jsonl",
        [
            {"source_id": "summary-1", "split": "train"},
            {"source_id": "qa-1", "split": "test"},
            {"source_id": "qa-2", "split": "train"},
        ],
    )

    records = RagTruthDataset(tmp_path, task="QA", split="test").records

    assert [record.sample_id for record in records] == ["qa-1"]


def test_ragtruth_dataset_requires_consistent_official_split(tmp_path):
    _write_jsonl(
        tmp_path / "source_info.jsonl",
        [
            {
                "source_id": "source-1",
                "task_type": "Summary",
                "source": "CNN/DM",
                "source_info": "Article.",
                "prompt": "Summarize:\nArticle.\noutput:",
            }
        ],
    )
    _write_jsonl(
        tmp_path / "response.jsonl",
        [
            {"source_id": "source-1", "split": "train"},
            {"source_id": "source-1", "split": "test"},
        ],
    )

    with pytest.raises(ValueError, match="multiple RAGTruth splits"):
        RagTruthDataset(tmp_path, task="Summary")
