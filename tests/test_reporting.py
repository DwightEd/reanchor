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
    np.savez_compressed(labels_path, labels=np.array([0, 0, 1, 1]))
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
        explicit_contrast=np.array([False, True, True]),
    )
    store.write_json(
        run / "tracing.json",
        {"sample_artifacts": [{"key": sample.key, "traces": [str(trace_path.relative_to(run))]}]},
    )
    source_margin = np.zeros((4, 3, 3, 3), dtype=np.float32)
    source_margin[3, 0, 0, 1] = -0.5
    source_margin[3, 0, 2, 2] = -0.4
    source_layers = np.zeros((4, 3, 3, 4, 3), dtype=np.float32)
    source_layers[3, 0, 0, -1, 1] = -0.5
    source_roots = np.zeros((4, 6), dtype=np.float32)
    source_roots[1, 1] = 0.4
    source_roots[3, 2] = 0.6
    mechanism_path = folder / "mechanisms/event_3.npz"
    store.write_npz(
        mechanism_path,
        event_position=np.array(3),
        target_position=np.array([3, 4, 5]),
        source_group_names=np.array(["constraint", "content", "other_prompt", "response_history"]),
        source_group_margin_response=source_margin,
        source_group_layer_margin_response=source_layers,
        source_group_root_coefficient=source_roots,
        source_group_seed_norm=np.array([0.0, 0.2, 0.0, 0.4]),
        source_unit_ids=np.array([3]),
        source_unit_roles=np.array(["content"]),
        source_unit_margin_response=source_margin[1:2],
        source_unit_root_coefficient=source_roots[1:2],
        source_unit_seed_norm=np.array([0.2]),
        transition_margin_response=source_margin.sum(0),
        current_remote_margin_response=source_margin.sum(0),
        baseline_margin=np.array([0.0, -1.0, 0.0]),
        positive_id=np.array([1, 2, 3]),
        negative_id=np.array([4, 5, 6]),
        explicit_contrast=np.array([False, True, True]),
        semantic_source_roles_available=np.array(True),
        closure_pass=np.array(True),
        closure_margin_max_abs=np.array(0.0),
        closure_layer_max_abs=np.array(0.0),
        closure_root_max_abs=np.array(0.0),
    )
    store.write_json(
        run / "mechanism.json",
        {
            "mechanism_schema": "reanchor/transition-mechanism-audit@2",
            "sample_artifacts": [
                {
                    "key": sample.key,
                    "mechanisms": [str(mechanism_path.relative_to(run))],
                }
            ],
        },
    )

    summary = ReportBuilder(ReportConfig(bootstrap=100)).run(FakeDataset(sample, labels_path), run)

    assert summary["labels_joined_only_in_reporting"] is True
    assert summary["groups"]["test/QA"]["event_incidence"]["estimate"] == 1.0
    assert summary["groups"]["test/QA"]["event_incidence_hallucinated"]["estimate"] == 1.0
    assert summary["groups"]["test/QA"]["anchor_target_hallucination"]["estimate"] == 1.0
    outcomes = summary["groups"]["test/QA"]["morphology_outcomes"]["broad_convergent"]
    assert outcomes["future_hallucination"]["1-4"]["estimate"] == 1.0
    causal = summary["causal_response"]
    assert causal["explicit_candidate_received_then_overridden"]["estimate"] == 0.5
    assert causal["observed_runner_margin_effect_hallucinated"]["observations"] == 0
    assert causal["explicit_candidate_margin_effect_hallucinated"]["observations"] == 2
    mechanism = summary["mechanism_audit"]
    assert mechanism["hallucinated_targets"] == 2
    assert mechanism["closure_failures"] == 0
    assert mechanism["phase_effects"]["onset"]["transition_remote_effect"]["estimate"] == -0.5
    chain = mechanism["same_event_onset_to_rollout"]
    assert chain["candidate_chain_rate"]["estimate"] == 1.0
    assert chain["claim_supported"] is False
    assert mechanism["binding_identified"] is False
    with (run / "reports/mechanisms.csv").open(newline="", encoding="utf-8") as handle:
        mechanism_rows = list(csv.DictReader(handle))
    assert mechanism_rows[0]["target_phase"] == "onset"
    assert mechanism_rows[1]["target_phase"] == "continuing"
    with (run / "reports/mechanism_source_units.csv").open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    assert [row["source_unit_id"] for row in source_rows] == ["3", "3"]
    assert json.loads((run / "index.json").read_text()) == index
    with (run / "reports/events.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["label"] for row in rows] == ["1", "1"]
