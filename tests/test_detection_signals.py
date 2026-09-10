from types import SimpleNamespace

import numpy as np

from reanchor.reporting.records import join_outcomes
from reanchor.reporting.signals import transition_signal_rows


def test_transition_signals_expose_local_remote_shape_and_source_category():
    sample = SimpleNamespace(
        split="test", task="QA", sample_id="7", source_id="source-7", response_start=3
    )
    shape = (1, 2, 3)
    transitions = {
        "token_text": np.array(["question", "evidence", "prompt", "A", "B", "C"]),
        "row_position": np.array([3, 4, 5]),
        "response_start": np.array(3),
        "special_mask": np.zeros(6, dtype=bool),
        "evidence_mask": np.array([False, True, False, False, False, False]),
        "eligible": np.array([True, True, False]),
        "current_local_mass": np.full(shape, 0.4),
        "previous_local_mass": np.full(shape, 0.8),
        "positive_remote_gain": np.array([[[0.4, 0.2, 0.0], [0.2, 0.2, 0.0]]]),
        "time_tv": np.full(shape, 0.25),
        "gain_focality": np.full(shape, 0.5),
        "effective_sources": np.full(shape, 2.0),
        "mean_distance": np.full(shape, 4.0),
        "prompt_positive_gain": np.array([[[0.4, 0.05, 0.0], [0.2, 0.05, 0.0]]]),
        "evidence_positive_gain": np.array([[[0.3, 0.0, 0.0], [0.2, 0.0, 0.0]]]),
        "history_positive_gain": np.array([[[0.0, 0.15, 0.0], [0.0, 0.15, 0.0]]]),
        "peak_source": np.array([[[1, 3, -1], [1, 3, -1]]]),
    }
    events = {
        "sparse_score": np.array([0.6, 0.2, 0.0]),
        "broad_score": np.array([0.3, 0.4, 0.0]),
        "significant": np.array([True, False, False]),
        "anchor": np.array([True, False, False]),
        "reanchor_type": np.array(["sparse_focal", "none", "none"]),
    }

    rows = transition_signal_rows(
        sample,
        transitions,
        events,
        predictor_logprob=np.array([-0.1, -0.2, np.nan]),
        predictor_entropy=np.array([0.5, 0.6, np.nan]),
    )

    assert len(rows) == 2
    assert rows[0]["query_token"] == "A"
    assert rows[0]["target_token"] == "B"
    assert rows[0]["remote_peak_token"] == "evidence"
    assert rows[0]["dominant_remote_route"] == "evidence"
    assert rows[0]["predictor_surprisal"] == 0.1
    assert rows[0]["predictor_entropy"] == 0.5
    assert rows[0]["local_mass_drop"] == 0.4
    assert rows[0]["attention_stability"] == 0.75
    assert np.isclose(rows[0]["evidence_gain_share"], 5 / 6)
    assert rows[1]["dominant_remote_route"] == "response_history"

    typed = transition_signal_rows(
        sample,
        transitions,
        events,
        source_kind=np.array(["other", "content", "other", "", "", ""]),
    )
    assert typed[0]["remote_peak_category"] == "content"


def test_outcomes_are_joined_after_signal_extraction_and_identify_span_phase():
    rows = [
        {"target_response_index": 0},
        {"target_response_index": 1},
        {"target_response_index": 2},
        {"target_response_index": 3},
    ]

    joined = join_outcomes(rows, np.array([0, 1, 1, -1]))

    assert [row["phase"] for row in joined] == ["normal", "onset", "continuing", ""]
    assert [row["label"] for row in joined] == [0, 1, 1, ""]
