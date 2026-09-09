import json

import numpy as np

from reanchor.capture.attention import AttentionReader
from reanchor.capture.protocol import AuditDataset


def test_reader_reconstructs_causal_rows_and_preserves_native_history(tmp_path):
    relative = "train/QA/sample.npz"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    rows = np.arange(1, 5)
    np.savez_compressed(
        path,
        token_ids=np.arange(5),
        token_text=np.array(list("abcde")),
        row_position=rows,
        response_start=np.array(2),
        special_mask=np.zeros(5, dtype=bool),
        evidence_mask=np.zeros(5, dtype=bool),
    )
    np.savez_compressed(
        path.with_suffix(".qk.npz"),
        query_0=np.zeros((1, 4, 2), dtype=np.float32),
        key_0=np.zeros((1, 5, 2), dtype=np.float32),
        scale_0=np.array(1.0),
        dtype_0=np.array("float32"),
    )
    native_history = np.zeros((1, 4, 4), dtype=np.float32)
    for index, query in enumerate(rows):
        native_history[0, index, : index + 1] = 1 / (query + 1)
    np.savez_compressed(path.with_suffix(".history.npz"), L0=native_history)
    (tmp_path / "index.json").write_text(
        json.dumps(
            {
                "audit_schema": 3,
                "labels_used_for_capture": False,
                "settings": {"model": "/models/llama"},
                "samples": [
                    {
                        "split": "train",
                        "task_type": "QA",
                        "sample_id": "sample",
                        "source_id": "source",
                        "path": relative,
                        "response_tokens": 3,
                        "response_start": 2,
                    }
                ],
            }
        )
    )
    dataset = AuditDataset(tmp_path)

    reader = AttentionReader(dataset, device="cpu", query_chunk=2)
    layers = list(reader.iter_layers(dataset.samples[0]))

    assert len(layers) == 1
    layer, attention = layers[0]
    assert layer == 0
    assert attention.shape == (1, 4, 5)
    np.testing.assert_allclose(attention.sum(-1), 1)
    np.testing.assert_allclose(attention[..., rows], native_history)
    for index, query in enumerate(rows):
        assert not attention[0, index, query + 1 :].any()
