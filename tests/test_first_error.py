import pytest

from reanchor.evaluation import alarm_metrics, binary_metrics, join_annotations
from reanchor.io import digest


def token_rows(text, source="s", response="r"):
    return [
        {
            "response_id": response,
            "source_id": source,
            "token_index": index,
            "char_start": index,
            "char_end": index + 1,
            "response_digest": digest(text),
            "response_token_count": len(text),
            "score": 0.1,
            "raw_score": 0.1,
        }
        for index in range(len(text))
    ]


def test_first_token_and_later_span_are_not_conflated():
    rows = token_rows("abcdef")
    truth = {
        "r": {
            "response": "abcdef",
            "source_id": "s",
            "labels": [{"start": 0, "end": 2, "text": "ab"}, {"start": 4, "end": 6, "text": "ef"}],
        }
    }
    joined = join_annotations(rows, truth)
    assert [r["token_index"] for r in joined if r["first_error"]] == [0]
    assert [r["token_index"] for r in joined if r["span_onset"]] == [0, 4]
    assert sum(r["hallucinated"] for r in joined) == 4
    with pytest.raises(ValueError, match="complete"):
        join_annotations(rows[1:], truth)


def test_wrong_text_or_source_is_rejected():
    truth = {"r": {"response": "abd", "source_id": "s", "labels": []}}
    with pytest.raises(ValueError, match="differs"):
        join_annotations(token_rows("abc"), truth)


@pytest.mark.parametrize(
    "spans,onsets",
    [([(0, 2), (2, 4)], [0, 2]), ([(0, 3), (1, 4)], [0, 1]), ([(0, 2), (0, 3)], [0])],
)
def test_official_span_onsets_are_not_merged(spans, onsets):
    truth = {
        "r": {
            "response": "abcd",
            "source_id": "s",
            "labels": [{"start": a, "end": b} for a, b in spans],
        }
    }
    joined = join_annotations(token_rows("abcd"), truth)
    assert [r["token_index"] for r in joined if r["span_onset"]] == onsets
    assert [r["token_index"] for r in joined if r["first_error"]] == [0]


def test_metrics_handle_ties_and_class_absence():
    rows = [
        {"source_id": "s", "label": True, "score": 0.5},
        {"source_id": "s", "label": False, "score": 0.5},
    ]
    result = binary_metrics(rows, "label", "score")
    assert result["auroc"] == pytest.approx(0.5)
    assert result["ap"] == pytest.approx(0.5)
    assert binary_metrics(rows[:1], "label", "score")["auroc"] is None
    rows[0]["score"] = 1
    assert binary_metrics(rows, "label", "score")["auroc"] == 1


def test_early_false_alarm_is_not_exact_onset_detection():
    rows = token_rows("abcd")
    rows[0]["score"] = rows[2]["score"] = 1.0
    truth = {
        "r": {
            "response": "abcd",
            "source_id": "s",
            "labels": [{"start": 2, "end": 4, "text": "cd"}],
        }
    }
    records, result = alarm_metrics(join_annotations(rows, truth))
    assert result["first_alarm_exact_onset_recall"] == 0
    assert records[0]["delay_after_onset"] == 0
    assert records[0]["false_alarms"] == 1
