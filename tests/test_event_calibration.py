import numpy as np
import pytest

from reanchor.discovery.calibration import (
    MaxNullCalibrator,
    SelectionConfig,
    TokenScoreRecord,
)
from reanchor.discovery.scores import score_transitions


def score_record(source_id, sparse, *, broad=None, support=None, split="train"):
    sparse = np.asarray(sparse, dtype=np.float32)
    count = len(sparse)
    broad = np.zeros(count, dtype=np.float32) if broad is None else np.asarray(broad)
    support = np.ones(count, dtype=np.int32) if support is None else np.asarray(support)
    return TokenScoreRecord(
        split=split,
        task="QA",
        sample_id=source_id,
        source_id=source_id,
        row_position=np.arange(10, 10 + count),
        eligible=np.ones(count, dtype=bool),
        position_bin=np.minimum(np.arange(count) * 2 // count, 1),
        sparse_score=sparse,
        broad_score=broad,
        supporting_sites=support,
    )


def test_source_max_calibration_rejects_any_head_overselection_and_collapses_episode():
    config = SelectionConfig(
        position_bins=2,
        family_alpha=0.05,
        min_calibration_sources=64,
        sparse_gain_floor=0.1,
        broad_gain_floor=0.05,
        episode_gap=1,
    )
    calibration = []
    for index in range(128):
        values = np.full(20, 0.2, dtype=np.float32)
        values[4] = 0.25 + index / 10_000
        values[14] = 0.25 + index / 10_000
        calibration.append(score_record(f"source-{index}", values))

    target_values = np.full(20, 0.2, dtype=np.float32)
    target_values[8:10] = [0.95, 0.90]
    target = score_record("held-out", target_values, split="test")
    selected = MaxNullCalibrator(config).fit(calibration).select(target)

    # The old any-site rule would select all 20 raw scores above 0.1.
    assert np.flatnonzero(selected.significant).tolist() == [8, 9]
    assert np.flatnonzero(selected.anchor).tolist() == [8]
    assert selected.episode_id[8] == selected.episode_id[9] == 0
    assert selected.family_p_value[8] < selected.family_p_value[0]
    assert selected.channel[8] == "sparse"


def test_calibration_requires_enough_independent_sources():
    config = SelectionConfig(position_bins=2, min_calibration_sources=4)
    records = [score_record(f"source-{index}", np.full(8, 0.2)) for index in range(3)]

    with pytest.raises(ValueError, match="independent calibration sources"):
        MaxNullCalibrator(config).fit(records)


def test_broad_channel_does_not_depend_on_the_sparse_site_count():
    config = SelectionConfig(position_bins=1, min_calibration_sources=64)
    calibration = [
        score_record(f"source-{index}", np.zeros(8), broad=np.full(8, 0.02)) for index in range(128)
    ]
    target = score_record(
        "held-out",
        np.zeros(8),
        broad=np.array([0, 0, 0, 0.8, 0, 0, 0, 0]),
        support=np.zeros(8, dtype=np.int32),
        split="test",
    )

    selected = MaxNullCalibrator(config).fit(calibration).select(target)

    assert np.flatnonzero(selected.anchor).tolist() == [3]
    assert selected.channel[3] == "broad"


def test_token_scores_keep_sparse_and_broad_evidence_distinct():
    remote_gain = np.zeros((2, 4, 3), dtype=np.float32)
    previous_local = np.ones_like(remote_gain)
    remote_gain[0, 0, 1] = 0.8
    remote_gain[1, :, 2] = 0.3

    result = score_transitions(
        remote_gain,
        previous_local,
        eligible=np.array([False, True, True]),
        broad_head_fraction=0.5,
        local_floor=0.5,
        site_gain_floor=0.1,
    )

    assert result["sparse_score"].tolist() == pytest.approx([0, 0.8, 0.3])
    assert result["broad_score"].tolist() == pytest.approx([0, 0.4, 0.3])
    assert result["supporting_sites"].tolist() == [0, 1, 4]
