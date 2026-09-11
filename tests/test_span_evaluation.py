import csv

import pytest

from reanchor.cli import main


def test_span_evaluation_exposes_a_detector_that_only_recognizes_continuations(tmp_path):
    path = tmp_path / "tokens.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["source_id", "response_id", "token_index", "token_count", "label", "score"]
        )
        for t, (label, score) in enumerate(
            zip([0, 1, 1, 1, 0, 1, 1], [0.5, 0.1, 0.9, 0.9, 0.5, 0.1, 0.9])
        ):
            writer.writerow(["source", "response", t, 7, label, score])
    output = tmp_path / "metrics.csv"
    main(["evaluate-spans", "--input", str(path), "--output", str(output), "--bootstrap", "0"])
    with output.open() as stream:
        result = {row["subset"]: row for row in csv.DictReader(stream)}
    assert float(result["all"]["auroc"]) == pytest.approx(0.6)
    assert float(result["onsets"]["auroc"]) == 0
    assert float(result["continuations"]["auroc"]) == 1
    assert int(result["first_error"]["positive"]) == 1
    assert int(result["onsets"]["positive"]) == 2
    assert int(result["continuations"]["positive"]) == 3


def test_span_evaluation_rejects_a_missing_tail(tmp_path):
    path = tmp_path / "tokens.csv"
    path.write_text("source_id,response_id,token_index,token_count,label,score\ns,r,0,2,0,.5\n")
    with pytest.raises(ValueError, match="complete"):
        main(["evaluate-spans", "--input", str(path), "--output", str(tmp_path / "out.csv")])
