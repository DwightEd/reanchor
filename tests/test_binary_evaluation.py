import numpy as np

from reanchor.evaluation import BinaryEvaluator, DetectionEvaluator


def test_binary_evaluator_reports_source_balanced_auroc_and_aupr():
    rows = [
        {"source_id": "a", "label": 0, "score": 0.1},
        {"source_id": "a", "label": 1, "score": 0.9},
        {"source_id": "b", "label": 0, "score": 0.2},
        {"source_id": "b", "label": 1, "score": 0.8},
    ]

    result = BinaryEvaluator(bootstrap=20, seed=3).evaluate(rows, "score")

    assert result["auroc"] == 1.0
    assert result["aupr"] == 1.0
    assert result["sources"] == 2
    assert result["observations"] == 4
    assert result["evaluation_units"] == 4
    assert result["source_balanced"] is True
    assert result["cluster_bootstrap"] == "source_id"


def test_binary_evaluator_handles_ties_and_a_missing_class():
    tied = [
        {"source_id": "a", "label": 0, "score": 1.0},
        {"source_id": "b", "label": 1, "score": 1.0},
    ]
    one_class = [{"source_id": "a", "label": 0, "score": 0.1}]

    assert BinaryEvaluator(bootstrap=0).evaluate(tied, "score")["auroc"] == 0.5
    result = BinaryEvaluator(bootstrap=0).evaluate(one_class, "score")

    assert result["auroc"] is None
    assert result["aupr"] is None


def test_repeated_tokens_from_one_source_do_not_change_source_balanced_auroc():
    source_a = [
        {"source_id": "a", "label": 0, "score": 0.9},
        {"source_id": "a", "label": 1, "score": 0.1},
    ]
    source_b = [
        {"source_id": "b", "label": 0, "score": 0.2},
        {"source_id": "b", "label": 1, "score": 0.8},
    ]
    evaluator = BinaryEvaluator(bootstrap=0)

    base = evaluator.evaluate(source_a + source_b, "score")
    repeated = evaluator.evaluate(source_a * 10 + source_b, "score")

    assert base["auroc"] == 0.25
    assert repeated["auroc"] == base["auroc"]


def test_detection_evaluator_keeps_onset_and_continuing_as_separate_tasks():
    rows = [
        {
            "split": "test",
            "task": "QA",
            "source_id": "a",
            "phase": "normal",
            "label": 0,
            "score": 0.1,
        },
        {
            "split": "test",
            "task": "QA",
            "source_id": "a",
            "phase": "onset",
            "label": 1,
            "score": 0.9,
        },
        {
            "split": "test",
            "task": "QA",
            "source_id": "b",
            "phase": "normal",
            "label": 0,
            "score": 0.2,
        },
        {
            "split": "test",
            "task": "QA",
            "source_id": "b",
            "phase": "continuing",
            "label": 1,
            "score": 0.8,
        },
    ]

    result = DetectionEvaluator(bootstrap=0).evaluate(rows, ("score",))

    assert result["overall"]["hallucination"]["score"]["auroc"] == 1.0
    assert result["overall"]["onset"]["score"]["observations"] == 3
    assert result["overall"]["continuing"]["score"]["observations"] == 3
    assert result["by_split_task"]["test/QA"]["onset"]["score"]["aupr"] == 1.0
    assert not np.isnan(result["overall"]["hallucination"]["score"]["aupr"])
