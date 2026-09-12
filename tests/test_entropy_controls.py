import hashlib
import json

from decoding.entropy_controls import EntropyControls, match_onsets


def test_same_token_matching_does_not_compare_different_words_or_move_onset():
    rows = [
        dict(
            response_id="a",
            source_id="a",
            token_index=0,
            token_id=7,
            entropy=1.0,
            label=False,
            first_error=False,
            boundary=True,
            clean_clause=True,
        ),
        dict(
            response_id="a",
            source_id="a",
            token_index=1,
            token_id=9,
            entropy=100.0,
            label=False,
            first_error=False,
            boundary=False,
            clean_clause=True,
        ),
        dict(
            response_id="a",
            source_id="a",
            token_index=2,
            token_id=7,
            entropy=3.0,
            label=True,
            first_error=True,
            boundary=False,
            clean_clause=False,
        ),
        dict(
            response_id="b",
            source_id="b",
            token_index=0,
            token_id=7,
            entropy=2.0,
            label=False,
            first_error=False,
            boundary=True,
            clean_clause=True,
        ),
    ]
    pairs = match_onsets(rows)
    same_source = [p for p in pairs if p["control"] == "same_response_token"]
    assert len(same_source) == 1
    assert same_source[0]["normal_entropy"] == 1.0
    assert same_source[0]["difference"] == 2.0
    assert not any(p["control"] == "same_token_clause_start" for p in pairs)


def test_run_keeps_only_complete_answers_and_exact_clause_starts(tmp_path):
    text = "The sun. The moon."
    label = dict(id="a", source_id="s", response=text, labels=[dict(start=9, end=17)])
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps(label) + "\n")
    rows = []
    for t, (span, token, entropy) in enumerate(
        [([0, 3], 7, 1.0), ([3, 8], 8, 0.5), ([9, 12], 7, 3.0), ([12, 17], 9, 0.5)]
    ):
        rows.append(
            dict(
                schema="route-graph/features@1",
                response_id="a",
                source_id="s",
                split="test",
                token_index=t,
                token_count=4,
                char_span=span,
                token_id=token,
                entropy=entropy,
                response_sha256=hashlib.sha256(text.encode()).hexdigest(),
            )
        )
    rows.append({**rows[0], "response_id": "unfinished"})
    features = tmp_path / "features.jsonl"
    features.write_text("".join(json.dumps(r) + "\n" for r in rows))
    output = tmp_path / "matched"
    result = EntropyControls(features, labels, output).run()
    assert result["first_errors"] == 1
    assert result["incomplete_responses"] == 1
    assert result["controls"]["same_token_clause_start"]["mean_difference"] == 2.0
