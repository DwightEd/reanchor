from pathlib import Path

import numpy as np

from reanchor.capture.protocol import AuditSample
from reanchor.discovery.events import DiscoveryConfig, EventDiscovery
from reanchor.discovery.extractor import TransitionFeatures


class SampleDataset:
    def __init__(self, samples):
        self.samples = tuple(samples)

    def completed_samples(self, *, require_states):
        assert not require_states
        coverage = {"planned": len(self.samples), "completed": len(self.samples), "skipped": 0}
        return self.samples, coverage


class FeatureExtractor:
    def __init__(self):
        self.fail = False

    def run(self, sample):
        if self.fail:
            raise AssertionError("committed transitions should be resumed")
        rows = np.arange(1, 5)
        shape = (1, 2, len(rows))
        remote_gain = np.zeros(shape, dtype=np.float32)
        remote_gain[:, :, 1:3] = 0.2
        if sample.split == "test":
            remote_gain[:, :, 2] = 0.9
        positive = np.maximum(remote_gain, 0)
        peak = np.full(shape, -1, dtype=np.int32)
        peak[positive > 0] = 0
        finite = np.ones(shape, dtype=np.float32)
        return TransitionFeatures(
            sample=sample,
            token_ids=np.arange(5),
            token_text=np.array(list("abcde")),
            row_position=rows,
            response_start=2,
            special_mask=np.zeros(5, dtype=bool),
            evidence_mask=np.array([True, False, False, False, False]),
            eligible=np.array([False, True, True, False]),
            current_local_mass=finite,
            current_remote_mass=remote_gain,
            previous_local_mass=finite,
            remote_gain=remote_gain,
            positive_remote_gain=positive,
            time_tv=remote_gain,
            gain_focality=np.where(positive > 0, 1, np.nan),
            effective_sources=np.where(positive > 0, 1, np.nan),
            mean_distance=np.where(positive > 0, 3, np.nan),
            prompt_positive_gain=positive,
            evidence_positive_gain=positive,
            history_positive_gain=np.zeros(shape, dtype=np.float32),
            peak_source=peak,
        )


def test_discovery_calibrates_on_train_and_writes_one_held_out_anchor(tmp_path):
    samples = [
        AuditSample("train", "QA", str(index), f"source-{index}", Path(f"{index}.npz"), 3, 2)
        for index in range(40)
    ]
    samples.append(AuditSample("test", "QA", "held", "held", Path("held.npz"), 3, 2))
    config = DiscoveryConfig(
        window=2,
        position_bins=1,
        min_calibration_sources=32,
        broad_head_fraction=0.5,
    )

    extractor = FeatureExtractor()
    workflow = EventDiscovery(config, extractor=extractor)
    summary = workflow.run(SampleDataset(samples), tmp_path / "run")

    assert summary["samples"] == 41
    assert summary["significant_transitions"] == 1
    assert summary["anchors"] == 1
    assert summary["calibration_sources"] == 40
    event_path = tmp_path / "run/samples/test/QA/held/events.npz"
    with np.load(event_path, allow_pickle=False) as events:
        assert np.flatnonzero(events["anchor"]).tolist() == [2]
        assert str(events["reanchor_type"][2]) == "broad_convergent"
        assert not bool(events["labels_used"])
    assert (tmp_path / "run/index.json").is_file()
    assert (tmp_path / "run/calibration.json").is_file()

    extractor.fail = True
    resumed = workflow.run(SampleDataset(samples), tmp_path / "run")
    assert resumed["resumed_transitions"] == 41
