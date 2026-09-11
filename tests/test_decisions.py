import csv
import json

import numpy as np
import pytest

from main import main


@pytest.fixture
def decision_input(tmp_path):
    weights = np.zeros((2, 2, 4, 7), dtype=np.float32)
    weights[:, :, 0, :3] = [0.1, 0.8, 0.1]
    weights[:, :, 1, :4] = [0.1, 0.45, 0.15, 0.3]
    weights[:, :, 2, :5] = [0.03, 0.5, 0.05, 0.02, 0.4]
    weights[1, 1, 2, 3] = 0
    weights[1, 1, 2, 1] += 0.02
    weights[:, :, 3, :6] = [0.01, 0.5, 0.02, 0.02, 0.15, 0.3]
    text = np.array(["<s>", " fact", ":", " onions", " for", "10", " minutes"])
    np.savez(
        tmp_path / "00000.npz",
        attention=weights,
        prompt_length=3,
        token_ids=np.arange(7),
        token_text=text,
        top_ids=np.tile([1, 2, 0, 6, 5], (4, 1)),
        top_logits=np.tile(np.log([0.4, 0.2, 0.1, 0.08, 0.04]) + 4, (4, 1)),
        top_text=np.tile(text[[1, 2, 0, 6, 5]], (4, 1)),
        chosen_logit=np.log([0.03, 0.03, 0.04, 0.08]) + 4,
        log_normalizer=np.full(4, 4.0),
    )
    (tmp_path / "samples.jsonl").write_text(
        json.dumps({"source_id": "q", "seed": 0, "trace": "00000.npz"}) + "\n"
    )
    (tmp_path / "settings.json").write_text(
        json.dumps({"model": "synthetic", "dtype": "float32", "temperature": 0.7, "top_p": 0.9})
    )
    (tmp_path / "prompts.jsonl").write_text('{"source_id":"q","prompt":" fact"}\n')
    with (tmp_path / "cases.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            "case source_id seed start stop text history_steps prompt_positions".split()
        )
        writer.writerow(["time", "q", 0, 1, 4, " for10 minutes", "0 1 2 3", "1"])
    return tmp_path


def run_decisions(path, output="inspection"):
    main(
        [
            "decisions",
            "--samples",
            str(path),
            "--cases",
            str(path / "cases.csv"),
            "--output",
            str(path / output),
        ]
    )
    return path / output


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_decisions_export_native_candidates_and_actual_choice(decision_input):
    output = run_decisions(decision_input)
    candidates = read_csv(output / "candidates.csv")
    assert len(candidates) == 15
    first = [r for r in candidates if r["step"] == "1"]
    assert [int(r["candidate_id"]) for r in first] == [1, 2, 0, 6, 5]
    assert [float(r["probability"]) for r in first] == pytest.approx([0.4, 0.2, 0.1, 0.08, 0.04])
    assert sum(float(r["probability"]) for r in first) == pytest.approx(0.82)
    assert all(r["is_chosen"] == "0" for r in first)
    chosen = read_csv(output / "tokens.csv")
    assert chosen[0]["token"] == " for"
    assert float(chosen[0]["probability"]) == pytest.approx(0.03)
    assert float(chosen[0]["chosen_margin"]) == pytest.approx(-2.5902671654)
    assert float(chosen[0]["top2_margin"]) == pytest.approx(0.6931471806)
    assert chosen[0]["logit_entropy"] == ""
    assert chosen[0]["chosen_rank"] == ""
    assert chosen[1]["chosen_rank"] == "5"


def test_decisions_keep_nonpeak_edges_all_heads_and_only_visible_keys(decision_input):
    output = run_decisions(decision_input)
    rows = read_csv(output / "reads.csv")
    assert len(rows) == 36
    assert {(r["layer"], r["head"]) for r in rows} == {
        ("0", "0"),
        ("0", "1"),
        ("1", "0"),
        ("1", "1"),
    }
    assert all(int(r["key_position"]) <= int(r["query_position"]) for r in rows)
    assert not any(r["key_region"] == "history" and r["key_step"] == "3" for r in rows)
    first = next(r for r in rows if r["step"] == "1" and r["key_region"] == "history")
    assert first["key_token"] == " onions" and first["distance"] == "0"
    assert first["previous_attention"] == first["gain"] == ""
    later = next(
        r
        for r in rows
        if r["step"] == "2" and r["key_step"] == "0" and r["layer"] == "0" and r["head"] == "0"
    )
    assert float(later["attention"]) == pytest.approx(0.02)
    assert float(later["previous_attention"]) == pytest.approx(0.3)
    assert float(later["gain"]) == pytest.approx(-0.28)
    zero = next(
        r
        for r in rows
        if r["step"] == "2" and r["key_step"] == "0" and r["layer"] == "1" and r["head"] == "1"
    )
    assert float(zero["attention"]) == 0


def test_prefix_audit_includes_the_branch_decision_but_not_later_states(decision_input):
    with np.load(decision_input / "00000.npz") as saved:
        second = dict(saved)
    second["token_ids"][4] = 8
    second["chosen_logit"][1] -= 1
    second["top_logits"][1, 0] += 0.01
    second["log_normalizer"][1] += 0.003
    np.savez(decision_input / "00001.npz", **second)
    with (decision_input / "samples.jsonl").open("a") as f:
        f.write(json.dumps({"source_id": "q", "seed": 1, "trace": "00001.npz"}) + "\n")
    output = run_decisions(decision_input)
    checks = read_csv(output / "same_prefix.csv")
    assert [r["step"] for r in checks] == ["0", "1"]
    assert checks[1]["next_id_a"] == "4" and checks[1]["next_id_b"] == "8"
    assert checks[1]["candidate_ids_equal"] == "1"
    assert float(checks[0]["max_top_logit_error"]) == 0
    assert float(checks[1]["max_top_logit_error"]) == pytest.approx(0.01)
    assert float(checks[1]["log_normalizer_error"]) == pytest.approx(0.003)


@pytest.mark.parametrize(
    "field,value", [("text", "another answer"), ("history_steps", "4"), ("prompt_positions", "3")]
)
def test_decisions_reject_wrong_capture_positions(decision_input, field, value):
    path = decision_input / "cases.csv"
    cases = read_csv(path)
    cases[0][field] = value
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    with pytest.raises(ValueError, match="window text|history step|prompt key"):
        run_decisions(decision_input)
