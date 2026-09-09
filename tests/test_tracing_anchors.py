import numpy as np

from reanchor.tracing.anchors import anchor_coordinates


def test_only_frozen_anchor_rows_produce_read_sites():
    remote_gain = np.array(
        [
            [[0.0, 0.20, 0.30, 0.0], [0.0, 0.04, 0.06, 0.0]],
            [[0.0, 0.08, 0.07, 0.0], [0.0, 0.09, 0.05, 0.0]],
        ]
    )
    previous_local = np.ones_like(remote_gain)
    anchor = np.array([False, False, True, False])

    coordinates = anchor_coordinates(
        remote_gain,
        previous_local,
        anchor,
        active_gain_floor=0.05,
        local_floor=0.5,
    )

    np.testing.assert_array_equal(
        coordinates,
        np.array([[0, 0, 2], [0, 1, 2], [1, 0, 2], [1, 1, 2]]),
    )


def test_anchor_without_an_active_site_is_rejected():
    values = np.zeros((1, 1, 3))

    try:
        anchor_coordinates(
            values,
            np.ones_like(values),
            np.array([False, True, False]),
            active_gain_floor=0.1,
            local_floor=0.5,
        )
    except ValueError as error:
        assert "no active read site" in str(error)
    else:
        raise AssertionError("an anchor must have at least one causal seed")
