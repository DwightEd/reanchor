import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from reanchor.artifacts import ArtifactStore
from reanchor.capture.protocol import AuditSample
from reanchor.tracing.tracer import CausalTracer, TraceConfig


class FakeDataset:
    def __init__(self, sample):
        self.sample = sample
        self.model_path = Path("checkpoint")

    def completed_samples(self, *, require_states):
        assert require_states
        return (self.sample,), {"planned": 1, "completed": 1, "skipped": 0}

    def paths(self, sample):
        assert sample == self.sample
        return SimpleNamespace(compact=Path("capture.npz"))


def test_tracer_consumes_only_frozen_anchors_and_publishes_one_trace(tmp_path, monkeypatch):
    sample = AuditSample("test", "QA", "7", "source-7", Path("7.npz"), 4, 2)
    contrast_path = tmp_path / "contrasts.json"
    contrast_path.write_text(
        json.dumps({sample.key: [{"target": 6, "positive_id": 11, "negative_id": 12}]})
    )
    run = tmp_path / "run"
    store = ArtifactStore(run)
    sample_folder = store.sample_path(sample, "events.npz").parent
    store.write_npz(
        sample_folder / "transitions.npz",
        remote_gain=np.array([[[0.0, 0.20, 0.30, 0.0]]]),
        previous_local_mass=np.ones((1, 1, 4)),
    )
    store.write_npz(
        sample_folder / "events.npz",
        anchor=np.array([False, False, True, False]),
        row_position=np.arange(3, 7),
    )
    store.write_json(
        run / "index.json",
        {
            "method_schema": "reanchor/max-null-episode@1",
            "settings": {
                "window": 2,
                "site_gain_floor": 0.1,
                "broad_gain_floor": 0.05,
                "local_floor": 0.5,
            },
            "sample_artifacts": [
                {
                    "key": sample.key,
                    "transitions": str((sample_folder / "transitions.npz").relative_to(run)),
                    "events": str((sample_folder / "events.npz").relative_to(run)),
                }
            ],
        },
    )
    seen = {"calls": 0}

    class FakeCache:
        def __init__(self, paths, weights):
            self.trace = {"row_position": np.arange(3, 7)}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    def fake_trace(cache, coordinates, **options):
        seen["calls"] += 1
        if seen["calls"] > 1:
            raise AssertionError("committed traces should be resumed")
        seen["coordinates"] = coordinates.copy()
        assert options["window"] == 2
        assert options["contrasts"] == [{"target": 6, "positive_id": 11, "negative_id": 12}]
        return [
            {
                "event_row": np.array(2),
                "event_position": np.array(5),
                "margin_response": np.zeros((3, 3, 3)),
                "labels_used": np.array(False),
            }
        ]

    import reanchor.tracing.tracer as module

    monkeypatch.setattr(module, "CheckpointWeights", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "NativeCache", FakeCache)
    monkeypatch.setattr(module, "trace_events", fake_trace)
    monkeypatch.setattr(module, "prepare_local_readout", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module, "open_cut_resources", lambda *args, **kwargs: nullcontext((None, None))
    )

    config = TraceConfig(device="cpu", save_edges=False, contrast_file=str(contrast_path))
    summary = CausalTracer(config).run(FakeDataset(sample), run)

    np.testing.assert_array_equal(seen["coordinates"], np.array([[0, 0, 2]]))
    assert summary["anchors_traced"] == 1
    assert summary["samples_considered"] == 1
    assert summary["samples_with_anchors"] == 1
    assert summary["samples_without_anchors"] == 0
    trace_path = sample_folder / "traces/event_5.npz"
    with np.load(trace_path, allow_pickle=False) as trace:
        assert not bool(trace["labels_used"])
    assert json.loads((run / "tracing.json").read_text())["labels_used_for_tracing"] is False

    resumed = CausalTracer(config).run(FakeDataset(sample), run)
    assert resumed["anchors_resumed"] == 1
