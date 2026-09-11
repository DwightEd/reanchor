import numpy as np
import pytest

from decoding.reading_graph import ReadingGraph
from decoding.route_readout import RouteReadout


def test_relation_residual_changes_when_equal_entropy_routes_are_rebound():
    weights = np.zeros((1, 2, 3, 5), dtype=np.float32)
    weights[:, :, :, 0] = 1
    weights[0, 1, 2] = 0
    weights[0, 1, 2, 2] = 1
    hidden = np.zeros((2, 4, 2), dtype=np.float32)
    hidden[1] = [[1, 0], [-1, 0], [1, 0], [1, 0]]
    source = np.array([True, True])
    good = RouteReadout(ReadingGraph(weights, source, 1).run(), hidden, 2).run()
    weights[0, 0, 2] = 0
    weights[0, 0, 2, 1] = 1
    bad = RouteReadout(ReadingGraph(weights, source, 1).run(), hidden, 2).run()
    assert good["relation_residual"][0, 0, 2] == pytest.approx(0)
    assert bad["relation_residual"][0, 0, 2] == pytest.approx(2)
    assert bad["relation_coverage"][0, 0, 2] == 1
    assert np.isnan(bad["relation_residual"][0, 0, :2]).all()


def test_no_source_path_is_unknown_rather_than_zero_residual():
    graph = dict(
        paths=np.zeros((1, 1, 2, 1)), history=np.zeros((1, 2, 2)), source_positions=np.array([0])
    )
    output = RouteReadout(graph, np.ones((2, 3, 2)), 2).run()
    assert np.isnan(output["relation_residual"]).all()
    assert not output["relation_coverage"].any()


def test_degenerate_source_representations_are_not_relation_evidence():
    graph = dict(
        paths=np.ones((1, 1, 3, 2)), history=np.zeros((1, 3, 3)), source_positions=np.array([0, 1])
    )
    graph["history"][0, 2, 0] = 1
    hidden = np.ones((2, 4, 2), dtype=np.float32)
    hidden[1, 2:] = [2, 0]
    result = RouteReadout(graph, hidden, 2).run()
    assert np.isnan(result["relation_residual"]).all()


def test_future_generated_states_do_not_change_earlier_relation_readouts():
    rng = np.random.default_rng(8)
    weights = np.zeros((2, 2, 5, 9), dtype=np.float32)
    for t in range(5):
        row = rng.uniform(size=(2, 2, 4 + t))
        weights[:, :, t, : 4 + t] = row / row.sum(-1, keepdims=True)
    graph = ReadingGraph(weights, np.ones(4, dtype=bool), 2).run()
    hidden = rng.normal(size=(3, 8, 4)).astype(np.float32)
    first = RouteReadout(graph, hidden, 4).run()
    hidden[:, -1] = rng.normal(size=(3, 4))
    weights[:, :, -1] = 0
    weights[:, :, -1, 0] = 1
    changed_graph = ReadingGraph(weights, np.ones(4, dtype=bool), 2).run()
    second = RouteReadout(changed_graph, hidden, 4).run()
    for field in first:
        np.testing.assert_allclose(first[field][..., :-1], second[field][..., :-1])
