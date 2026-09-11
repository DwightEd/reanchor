import csv
import json

import numpy as np

from reanchor.cli import main


def test_attention_csv_keeps_prediction_and_future_reading_positions_distinct(tmp_path):
    attention = np.zeros((1, 1, 3, 5), dtype=np.float16)
    attention[0, 0, 0, 0] = 1
    attention[0, 0, 1, [0, 2]] = 0.5
    attention[0, 0, 2, 2] = 1
    np.savez(tmp_path / "00000.npz", attention=attention, prompt_length=2, token_ids=np.arange(5))
    (tmp_path / "samples.jsonl").write_text(
        json.dumps({"source_id": "q", "seed": 3, "trace": "00000.npz"}) + "\n"
    )
    main(
        [
            "analyze-attention",
            "--samples",
            str(tmp_path),
            "--output",
            str(tmp_path / "attention.csv"),
            "--window",
            "2",
            "--future-range",
            "1",
            "2",
        ]
    )
    with (tmp_path / "attention.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["query_position"]) for row in rows] == [1, 2, 3]
    assert [float(row["waad"]) for row in rows] == [1, 1, 1]
    assert float(rows[0]["fai"]) == 1
    assert [int(row["fai_queries"]) for row in rows] == [1, 0, 0]
    assert rows[1]["fai"] == "" and rows[2]["fai"] == ""
