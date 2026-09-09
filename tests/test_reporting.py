import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from reanchor.artifacts import ArtifactStore
from reanchor.capture.protocol import AuditSample
from reanchor.reporting.evaluation import (
    ReportBuilder,
    ReportConfig,
    received_then_overridden,
    source_ratio_summary,
)


def test_source_ratio_does_not_overweight_a_source_with_more_tokens():
    rows = [
        {"source_id": "large", "selected": True},
        *({"source_id": "large", "selected": False} for _ in range(9)),
        {"source_id": "small", "selected": True},
    ]

    summary = source_ratio_summary(rows, "selected", bootstrap=100, seed=1)

    assert summary["estimate"] == 0.55
    assert summary["sources"] == 2
    assert summary["observations"] == 11


def test_override_requires_positive_intermediate_and_nonpositive_final_effect():
    assert received_then_overridden(np.array([0.0, 0.4, 0.2, -0.1]))
    assert not received_then_overridden(np.array([0.0, -0.2, -0.1]))
    assert not received_then_overridden(np.array([0.0, 0.2, 0.1]))


class FakeDataset:
    def __init__(self, sample, labels):
        self.sample = sample
        self.samples = (sample,)
        self.label_path = labels

    def paths(self, sample):
        assert sample == self.sample
        return SimpleNamespace(labels=self.label_path)

    def load_metadata(self, sample, *fields):
        assert sample == self.sample
        return {"special_mask": np.zeros(6, dtype=bool)}


def test_report_is_the_first_stage_that_joins_labels(tmp_path):
    sample = AuditSample("test", "QA", "7", "source-7", Path("7.npz"), 4, 2)
    labels_path = tmp_path / "7.labels.npz"
    np.savez_compressed(labels_path, labels=np.array([0, 0, 1, 0]))
    run = tmp_path / "run"
    store = ArtifactStore(run)
    folder = store.sample_path(sample, "events.npz").parent
    store.write_npz(
        folder / "transitions.npz",
        response_start=np.array(2),
        row_position=np.arange(2, 6),
    )
    store.write_npz(
        folder / "events.npz",
        row_position=np.arange(2, 6),
        eligible=np.array([False, True, True, False]),
        significant=np.array([False, True, True, False]),
        anchor=np.array([False, True, False, False]),
        episode_id=np.array([-1, 0, 0, -1]),
        channel=np.array(["none", "broad", "broad", "none"]),
        family_p_value=np.array([1.0, 0.01, 0.02, 1.0]),
        sparse_score=np.array([0.0, 0.2, 0.2, 0.0]),
        broad_score=np.array([0.0, 0.3, 0.2, 0.0]),
        reanchor_type=np.array(["none", "broad_convergent", "episode_member", "none"]),
    )
    index = {
        "method_schema": "reanchor/max-null-episode@1",
        "sample_artifacts": [
            {
                "key": sample.key,
                "source_id": sample.source_id,
                "transitions": str((folder / "transitions.npz").relative_to(run)),
                "events": str((folder / "events.npz").relative_to(run)),
            }
        ],
    }
    store.write_json(run / "index.json", index)
    layer_response = np.zeros((3, 3, 4, 3))
    layer_response[0, 0, :, 1] = [0.0, 0.4, 0.2, -0.1]
    trace_path = folder / "traces/event_3.npz"
    store.write_npz(
        trace_path,
        event_position=np.array(3),
        target_position=np.array([3, 4, 5]),
        margin_response=np.zeros((3, 3, 3)),
        layer_margin_response=layer_response,
        explicit_contrast=np.array([False, True, False]),
    )
    store.write_json(
        run / "tracing.json",
        {"sample_artifacts": [{"key": sample.key, "traces": [str(trace_path.relative_to(run))]}]},
    )

    summary = ReportBuilder(ReportConfig(bootstrap=100)).run(FakeDataset(sample, labels_path), run)

    assert summary["labels_joined_only_in_reporting"] is True
    assert summary["groups"]["test/QA"]["event_incidence"]["estimate"] == 1.0
    assert summary["groups"]["test/QA"]["event_incidence_hallucinated"]["estimate"] == 1.0
    assert summary["groups"]["test/QA"]["anchor_target_hallucination"]["estimate"] == 1.0
    outcomes = summary["groups"]["test/QA"]["morphology_outcomes"]["broad_convergent"]
    assert outcomes["future_hallucination"]["1-4"]["estimate"] == 0.0
    causal = summary["causal_response"]
    assert causal["explicit_candidate_received_then_overridden"]["estimate"] == 1.0
    assert causal["observed_runner_margin_effect_hallucinated"]["observations"] == 0
    assert causal["explicit_candidate_margin_effect_hallucinated"]["observations"] == 1
    assert json.loads((run / "index.json").read_text()) == index
    with (run / "reports/events.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["label"] for row in rows] == ["1", "0"]
