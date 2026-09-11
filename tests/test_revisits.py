import csv
import json

import numpy as np
import pytest

from main import main


def capture(path, attention):
    steps, length = attention.shape[2:]
    prompt = length - steps
    text = np.array([f" p{i}" for i in range(prompt)] + list("abcdef")[:steps])
    np.savez(
        path / "00000.npz",
        attention=attention,
        prompt_length=prompt,
        token_ids=np.arange(length),
        token_text=text,
        special_mask=np.zeros(length, dtype=bool),
        top_logits=np.tile([3.0, 2.0], (steps, 1)),
        log_normalizer=np.full(steps, 4.0),
    )
    (path / "samples.jsonl").write_text(
        json.dumps({"source_id": "q", "seed": 0, "trace": "00000.npz"}) + "\n",
        encoding="utf-8",
    )


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def analyze(path, name="analysis"):
    output = path / name
    main(
        [
            "revisits",
            "--samples",
            str(path),
            "--output",
            str(output),
            "--window",
            "2",
            "--context",
            "1",
        ]
    )
    return output / "00000"


def test_revisits_distinguish_split_heads_from_shared_diffuse_reading(tmp_path):
    attention = np.zeros((1, 2, 3, 23), dtype=np.float32)
    attention[0, 0, 0, 2] = attention[0, 1, 0, 12] = 1
    attention[0, :, 1, [2, 12]] = 0.5
    attention[0, 0, 2, 12] = attention[0, 1, 2, 2] = 1
    capture(tmp_path, attention)
    output = analyze(tmp_path)
    layers = rows(output / "layers.csv")
    assert [float(r["dispersion"]) for r in layers] == pytest.approx([0, 1, 0])
    assert [float(r["disagreement"]) for r in layers] == pytest.approx([1, 0, 1])
    assert [float(r["effective_rank"]) for r in layers] == pytest.approx([2, 1, 2])
    tokens = rows(output / "tokens.csv")
    assert [r["token"] for r in tokens] == list("abc")
    assert all(r["logit_entropy"] == "" for r in tokens)
    assert all(float(r["top1_probability"]) == pytest.approx(0.3678794412) for r in tokens)
    assert int(tokens[0]["query_position"]) == 19


def test_revisits_ignore_adjacent_copy_and_follow_short_history_reads(tmp_path):
    attention = np.zeros((2, 2, 6, 26), dtype=np.float32)
    for t, key in enumerate([1, 2, 3, 4, 18, 18]):
        attention[:, :, t, key] = 1
    attention[1, :, 4:, :] = 0
    attention[1, :, 4:, 20] = 1
    capture(tmp_path, attention)
    output = analyze(tmp_path)
    tokens = rows(output / "tokens.csv")
    assert [int(r["event"]) for r in tokens] == [0, 0, 0, 0, 1, 0]
    assert tokens[0]["shift"] == ""
    assert [float(r["shift"]) for r in tokens[1:]] == pytest.approx([1, 1, 1, 15, 0])
    assert float(tokens[4]["revisit"]) == pytest.approx(14)
    reads = rows(output / "reads.csv")
    assert {int(r["step"]) for r in reads} == {3, 4, 5}
    relay = next(r for r in reads if r["step"] == "4" and r["layer"] == "1")
    assert relay["key_region"] == "history"
    assert relay["key_token"] == "a"
    assert relay["distance"] == "3"
    # Generated token 0's own query is row 1; row 0 would incorrectly return key 1.
    assert relay["relay_position"] == "2"
    assert float(relay["path_weight"]) == 1
    attention[:, :, 5, :] = 0
    attention[:, :, 5, 3] = 1
    capture(tmp_path, attention)
    changed = analyze(tmp_path, "changed_future")
    for filename in ("tokens.csv", "layers.csv"):
        assert [r for r in rows(output / filename) if int(r["step"]) < 5] == [
            r for r in rows(changed / filename) if int(r["step"]) < 5
        ]


@pytest.mark.parametrize("problem", ["future", "zero", "negative"])
def test_revisits_reject_invalid_attention_instead_of_scoring_it(tmp_path, problem):
    attention = np.zeros((1, 2, 3, 23), dtype=np.float32)
    attention[:, :, :, 2] = 1
    if problem == "future":
        attention[0, 0, 0, 22] = 0.5
    elif problem == "zero":
        attention[0, 0, 0] = 0
    else:
        attention[0, 0, 0, 3] = -0.5
    capture(tmp_path, attention)
    with pytest.raises(ValueError, match="invalid attention"):
        analyze(tmp_path)
