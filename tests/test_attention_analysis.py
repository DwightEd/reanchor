import csv
import json

import numpy as np
import pytest

from main import main


@pytest.fixture
def attention_trace(tmp_path):
    attention = np.zeros((1, 1, 4, 9), dtype=np.float32)
    attention[0, 0, 0, [1, 2]] = [0.1, 0.9]
    attention[0, 0, 1, [0, 1, 2, 5]] = [0.05, 0.6, 0.15, 0.2]
    attention[0, 0, 2, [0, 1, 2, 5, 6]] = [0.025, 0.3, 0.075, 0.1, 0.5]
    attention[0, 0, 3, 3] = 1
    np.savez(
        tmp_path / "00000.npz",
        attention=attention,
        prompt_length=5,
        token_ids=np.arange(9),
        special_mask=[True] + [False] * 8,
        token_text=["<s>", " Alice", " Bob", " fact", ":", " yes", " therefore", " wrong", "."],
        token_pieces=["<s>", "Alice", "Bob", "fact", ":", "yes", "therefore", "wrong", "."],
        top_ids=np.tile([1, 2], (4, 1)),
        top_logits=np.tile([3.0, 2.0], (4, 1)),
        log_normalizer=np.full(4, 4.0),
        chosen_logit=np.full(4, 1.0),
        top_text=np.tile([" Alice", " Bob"], (4, 1)),
    )
    (tmp_path / "samples.jsonl").write_text(
        json.dumps(
            {"source_id": "q", "seed": 3, "trace": "00000.npz", "response": "yes therefore wrong."}
        )
        + "\n"
    )
    return tmp_path


def test_inspect_reads_same_past_keys_before_the_annotated_error(attention_trace):
    tmp_path = attention_trace
    output = tmp_path / "before error"
    main(
        [
            "inspect",
            "--samples",
            str(tmp_path),
            "--output",
            str(output),
            "--trace",
            "00000.npz",
            "--first-error",
            "2",
            "--before",
            "1",
            "--after",
            "0",
            "--min-distance",
            "2",
        ]
    )
    with (output / "attention.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["step"]) for row in rows] == [1, 2]
    assert [int(row["error_offset"]) for row in rows] == [-1, 0]
    assert [int(row["query_position"]) for row in rows] == [5, 6]
    assert rows[0]["query"] == " yes"
    assert rows[0]["previous_peak"] == " Bob"
    assert rows[0]["read_token"] == " Alice"
    assert float(rows[0]["shift"]) == pytest.approx(0.7)
    assert float(rows[0]["previous_attention"]) == pytest.approx(0.1)
    assert float(rows[0]["attention"]) == pytest.approx(0.6)
    assert float(rows[0]["gain"]) == pytest.approx(0.5)
    # A new self key and a changing sink must not manufacture a past-route switch.
    assert float(rows[1]["shift"]) == pytest.approx(0, abs=1e-6)
    assert "wrong" not in rows[0]["read_context"]
    with (output / "tokens.csv").open(encoding="utf-8") as stream:
        tokens = list(csv.DictReader(stream))
    assert [row["token"] for row in tokens] == [" therefore", " wrong"]
    assert float(tokens[0]["top2_margin"]) == 1
    assert float(tokens[0]["chosen_margin"]) == -2
    assert float(tokens[0]["probability"]) == pytest.approx(np.exp(-3))


def test_first_generated_error_has_no_invented_previous_reading(attention_trace):
    output = attention_trace / "first"
    main(
        [
            "inspect",
            "--samples",
            str(attention_trace),
            "--output",
            str(output),
            "--trace",
            "00000.npz",
            "--first-error",
            "0",
            "--min-distance",
            "1",
        ]
    )
    with (output / "attention.csv").open(encoding="utf-8") as stream:
        (row,) = csv.DictReader(stream)
    assert row["step"] == "0" and row["error_offset"] == "0"
    assert row["query"] == ":"
    assert row["shift"] == row["previous_attention"] == row["gain"] == ""
    assert row["read_token"] == " Bob"


def test_future_decoding_does_not_change_pre_error_reading(attention_trace):
    command = [
        "inspect",
        "--samples",
        str(attention_trace),
        "--trace",
        "00000.npz",
        "--first-error",
        "2",
        "--min-distance",
        "2",
    ]
    main(command + ["--output", str(attention_trace / "original")])
    with np.load(attention_trace / "00000.npz") as trace:
        arrays = dict(trace)
    arrays["attention"][:, :, 3, :] = 0
    arrays["attention"][:, :, 3, 0] = 1
    arrays["token_text"][7:] = ["future", "changed"]
    np.savez(attention_trace / "00000.npz", **arrays)
    main(command + ["--output", str(attention_trace / "changed")])
    assert (attention_trace / "original/attention.csv").read_bytes() == (
        attention_trace / "changed/attention.csv"
    ).read_bytes()
