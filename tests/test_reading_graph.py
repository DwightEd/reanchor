import numpy as np
import pytest

from decoding.reading_graph import ReadingGraph, RevisitSignal


def test_content_events_ignore_sink_changes_and_detect_reading_a_distant_source():
    weights = np.zeros((1, 2, 7, 27))
    for t, (key, sink) in enumerate(zip([2, 3, 4, 5, 6, 17, 18], [0.8, 0.2] * 3 + [0.8])):
        weights[:, :, t, 0] = sink
        weights[:, :, t, key] = 1 - sink
    source = np.zeros(20, dtype=bool)
    source[1:19] = True
    special = np.zeros(27, dtype=bool)
    special[0] = True
    values = RevisitSignal(weights, source, special, window=2, quantile=0.95).run()
    assert values["event"].tolist() == [False, False, False, False, False, True, False]
    assert values["shift"][1:].ravel() == pytest.approx([1, 1, 1, 1, 11, 1])
    assert values["revisit"][5] > values["threshold"][5]
    assert np.all(values["dispersion"] == pytest.approx(0))


def test_content_events_never_use_future_attention():
    weights = np.zeros((1, 1, 6, 26))
    for t, key in enumerate([2, 3, 4, 5, 17, 18]):
        weights[:, :, t, key] = 1
    source = np.ones(20, dtype=bool)
    special = np.zeros(26, dtype=bool)
    first = RevisitSignal(weights, source, special, 2, 0.95).run()
    weights[:, :, 5] = 0
    weights[:, :, 5, 1] = 1
    second = RevisitSignal(weights, source, special, 2, 0.95).run()
    for field in first:
        np.testing.assert_equal(first[field][:5], second[field][:5])


def test_three_hop_paths_preserve_both_branches_and_use_history_input_rows():
    weights = np.zeros((3, 2, 4, 8))
    weights[:, :, :, 0] = 1
    weights[0, :, 1] = 0
    weights[0, :, 1, 0] = 0.25
    weights[0, :, 1, 1] = 0.75
    weights[1, :, 2] = 0
    weights[1, :, 2, 4] = 1
    weights[2, :, 3] = 0
    weights[2, :, 3, 5] = 1
    graph = ReadingGraph(weights, np.array([1, 1, 0, 0], dtype=bool), hops=3).run()
    assert graph["paths"][2, 2, 3] == pytest.approx([0.25, 0.75])
    assert graph["paths"][1, 1, 2] == pytest.approx([0.25, 0.75])
    assert graph["paths"][0, 2, 3].sum() == 0
    assert graph["history"][2, 3, 1] == 1
    assert graph["source_positions"].tolist() == [0, 1]
