from types import SimpleNamespace

import numpy as np
import pytest

from reanchor.discovery.calibration import CalibratedSelection
from reanchor.discovery.morphology import MorphologyProfiler


def test_morphology_describes_frozen_anchor_without_reselecting_it():
    shape = (1, 4, 2)
    remote_gain = np.zeros(shape, dtype=np.float32)
    remote_gain[0, :, 1] = 0.2
    previous_local = np.ones(shape, dtype=np.float32)
    positive_gain = remote_gain.copy()
    focality = np.ones(shape, dtype=np.float32)
    effective = np.ones(shape, dtype=np.float32)
    distance = np.full(shape, 7, dtype=np.float32)
    peak = np.full(shape, -1, dtype=np.int32)
    peak[0, :, 1] = [2, 2, 2, 3]
    evidence = np.zeros(10, dtype=bool)
    evidence[2] = True
    features = SimpleNamespace(
        eligible=np.array([False, True]),
        remote_gain=remote_gain,
        previous_local_mass=previous_local,
        positive_remote_gain=positive_gain,
        gain_focality=focality,
        effective_sources=effective,
        mean_distance=distance,
        peak_source=peak,
        response_start=4,
        special_mask=np.zeros(10, dtype=bool),
        evidence_mask=evidence,
        token_ids=np.arange(10),
        token_text=np.array([f"t{index}" for index in range(10)]),
    )
    selection = CalibratedSelection(
        family_p_value=np.array([1.0, 0.01]),
        channel=np.array(["none", "broad"]),
        significant=np.array([False, True]),
        episode_id=np.array([-1, 0]),
        anchor=np.array([False, True]),
    )

    profile = MorphologyProfiler().run(features, selection)

    assert profile.reanchor_type.tolist() == ["excluded", "broad_convergent"]
    assert profile.dominant_layer[1] == 0
    assert profile.active_head_fraction[1] == 1
    assert profile.source_agreement[1] == pytest.approx(0.75)
    assert profile.peak_prompt_fraction[1] == 1
    assert profile.peak_evidence_fraction[1] == pytest.approx(0.75)
    assert profile.dominant_source_position[1] == 2
    assert profile.dominant_source_token_text[1] == "t2"
