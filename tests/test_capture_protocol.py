import json

import numpy as np
import pytest

from reanchor.capture.protocol import AuditDataset


def write_capture(root, relative, *, states=True):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        token_ids=np.arange(8),
        token_text=np.array(list("abcdefgh")),
        row_position=np.arange(2, 8),
        response_start=np.array(3),
        special_mask=np.zeros(8, dtype=bool),
        evidence_mask=np.array([False, True, True, False, False, False, False, False]),
        source_unit_id=np.full(8, -1),
        predictor_logprob=np.zeros(6),
        head_margin=np.zeros((2, 2, 6)),
        readout_runner_id=np.zeros(6, dtype=np.int64),
    )
    np.savez_compressed(path.with_suffix(".qk.npz"), query_0=np.zeros((2, 6, 4)))
    np.savez_compressed(path.with_suffix(".history.npz"), L0=np.zeros((2, 6, 6)))
    if states:
        np.savez_compressed(path.with_suffix(".states.npz"), final_residual=np.zeros((6, 8)))


def test_dataset_exposes_only_complete_v3_samples_and_reports_coverage(tmp_path):
    write_capture(tmp_path, "train/QA/complete.npz")
    write_capture(tmp_path, "train/QA/incomplete.npz", states=False)
    manifest = {
        "audit_schema": 3,
        "labels_used_for_capture": False,
        "settings": {"model": "/models/llama", "save_states": True},
        "samples": [
            {
                "split": "train",
                "task_type": "QA",
                "sample_id": "complete",
                "source_id": "source-a",
                "path": "train/QA/complete.npz",
                "response_tokens": 5,
                "response_start": 3,
            },
            {
                "split": "train",
                "task_type": "QA",
                "sample_id": "incomplete",
                "source_id": "source-b",
                "path": "train/QA/incomplete.npz",
                "response_tokens": 5,
                "response_start": 3,
            },
        ],
    }
    (tmp_path / "index.json").write_text(json.dumps(manifest))

    dataset = AuditDataset(tmp_path)
    samples, coverage = dataset.completed_samples(require_states=True)

    assert [sample.key for sample in samples] == ["train/QA/complete"]
    assert coverage == {"planned": 2, "completed": 1, "skipped": 1}
    assert dataset.paths(samples[0]).qk.name == "complete.qk.npz"


def test_dataset_rejects_capture_paths_outside_the_root(tmp_path):
    manifest = {
        "audit_schema": 3,
        "labels_used_for_capture": False,
        "settings": {},
        "samples": [
            {
                "split": "train",
                "task_type": "QA",
                "sample_id": "bad",
                "source_id": "source",
                "path": "../outside.npz",
                "response_tokens": 1,
                "response_start": 1,
            }
        ],
    }
    (tmp_path / "index.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="outside capture root"):
        AuditDataset(tmp_path)


def test_dataset_rejects_source_leakage_across_splits(tmp_path):
    sample = {
        "task_type": "QA",
        "sample_id": "a",
        "source_id": "shared-source",
        "path": "a.npz",
        "response_tokens": 1,
        "response_start": 1,
    }
    manifest = {
        "audit_schema": 3,
        "labels_used_for_capture": False,
        "settings": {},
        "samples": [dict(sample, split="train"), dict(sample, split="test", sample_id="b")],
    }
    (tmp_path / "index.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="appears in multiple splits"):
        AuditDataset(tmp_path)
