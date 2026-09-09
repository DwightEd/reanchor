import numpy as np

from reanchor.discovery.features import measure_attention_transitions


def causal_rows(rows, sources):
    attention = np.zeros((1, len(rows), sources), dtype=np.float32)
    for index, query in enumerate(rows):
        attention[0, index, query] = 1.0
    return attention


def test_current_partition_prevents_window_aging_from_becoming_remote_gain():
    rows = np.arange(2, 8)
    attention = causal_rows(rows, 8)
    # Source 2 moves from distance 2 to distance 3, but its attention is unchanged.
    attention[0, 2] = 0
    attention[0, 2, 2] = 1
    attention[0, 3] = 0
    attention[0, 3, 2] = 1

    result = measure_attention_transitions(
        attention,
        rows=rows,
        response_start=3,
        special_mask=np.zeros(8, dtype=bool),
        evidence_mask=np.zeros(8, dtype=bool),
        window=2,
    )

    assert result["remote_gain"][0, 3] == 0
    assert result["previous_local_mass"][0, 3] == 0


def test_transition_features_separate_evidence_history_and_special_sources():
    rows = np.arange(2, 8)
    attention = causal_rows(rows, 8)
    # At q=4 attention is local. At q=5 it moves to prompt evidence at source 2.
    attention[0, 2] = 0
    attention[0, 2, 4] = 1
    attention[0, 3] = 0
    attention[0, 3, 2] = 0.8
    attention[0, 3, 5] = 0.2
    special = np.zeros(8, dtype=bool)
    special[0] = True
    evidence = np.zeros(8, dtype=bool)
    evidence[2] = True

    result = measure_attention_transitions(
        attention,
        rows=rows,
        response_start=3,
        special_mask=special,
        evidence_mask=evidence,
        window=2,
    )

    assert result["eligible"][3]
    assert result["remote_gain"][0, 3] == np.float32(0.8)
    assert result["positive_remote_gain"][0, 3] == np.float32(0.8)
    assert result["evidence_positive_gain"][0, 3] == np.float32(0.8)
    assert result["prompt_positive_gain"][0, 3] == np.float32(0.8)
    assert result["history_positive_gain"][0, 3] == 0
    assert result["gain_focality"][0, 3] == 1
    assert result["effective_sources"][0, 3] == 1
    assert result["mean_distance"][0, 3] == 3

    # Moving attention to a remote special token is not a reanchor feature.
    attention[0, 3] = 0
    attention[0, 3, 0] = 0.8
    attention[0, 3, 5] = 0.2
    excluded = measure_attention_transitions(
        attention,
        rows=rows,
        response_start=3,
        special_mask=special,
        evidence_mask=evidence,
        window=2,
    )
    assert excluded["remote_gain"][0, 3] == 0
    assert excluded["positive_remote_gain"][0, 3] == 0
