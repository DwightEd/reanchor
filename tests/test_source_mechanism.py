import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from reanchor.artifacts import ArtifactStore
from reanchor.capture.protocol import AuditSample
from reanchor.reporting.mechanisms import mechanism_target_row, source_unit_target_rows
from reanchor.tracing.contrasts import read_contrasts
from reanchor.tracing.jacobian import DifferentialLayer
from reanchor.tracing.mechanism import MechanismAuditor, MechanismConfig
from reanchor.tracing.source_groups import (
    SOURCE_GROUPS,
    source_partition,
    source_unit_partition,
)


def _semantic_trace():
    return {
        "token_ids": np.arange(8),
        "special_mask": np.array([True, False, False, False, False, False, False, True]),
        "response_start": np.array(5),
        "evidence_mask": np.array([False, False, True, False, False, False, False, False]),
        "source_kind": np.array(["", "constraint", "content", "other", "", "", "", ""]),
        "source_unit_id": np.array([-1, 4, 7, 8, 8, -1, -1, -1]),
    }


def test_source_partition_uses_semantic_roles_and_exactly_covers_ordinary_tokens():
    trace = _semantic_trace()

    partition = source_partition(trace)

    assert partition.names == SOURCE_GROUPS
    assert partition.semantic_roles_available is True
    np.testing.assert_array_equal(partition.masks[0], [0, 1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(partition.masks[1], [0, 0, 1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(partition.masks[2], [0, 0, 0, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(partition.masks[3], [0, 0, 0, 0, 0, 1, 1, 0])
    np.testing.assert_array_equal(partition.masks.any(0), ~trace["special_mask"])


def test_source_units_remain_separate_inside_a_coarse_group():
    trace = _semantic_trace()

    units = source_unit_partition(trace)

    np.testing.assert_array_equal(units.ids, [4, 7, 8])
    assert units.roles == ("constraint", "content", "other_prompt")
    np.testing.assert_array_equal(units.masks.sum(1), [1, 1, 2])


def test_legacy_source_partition_marks_evidence_as_content_without_inventing_constraints():
    trace = {
        "token_ids": np.arange(5),
        "special_mask": np.zeros(5, dtype=bool),
        "response_start": np.array(3),
        "evidence_mask": np.array([False, True, False, False, False]),
    }

    partition = source_partition(trace)

    assert partition.semantic_roles_available is False
    assert not partition.masks[0].any()
    np.testing.assert_array_equal(partition.masks[1], [0, 1, 0, 0, 0])
    np.testing.assert_array_equal(partition.masks[2], [1, 0, 1, 0, 0])
    np.testing.assert_array_equal(partition.masks[3], [0, 0, 0, 1, 1])


class FakeOperator:
    device = torch.device("cpu")
    rows = torch.tensor([3, 4])
    v = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [3.0, 1.0], [9.0, 9.0], [8.0, 8.0]]])
    output_blocks = torch.eye(2)[None]
    cache = type(
        "Cache",
        (),
        {
            "trace": {
                "token_ids": np.arange(5),
                "special_mask": np.zeros(5, dtype=bool),
            }
        },
    )()

    def rows_attention(self, stop=None):
        del stop
        attention = torch.tensor([[[0.2, 0.3, 0.1, 0.4, 0.0], [0.1, 0.2, 0.3, 0.1, 0.3]]])
        yield 0, 2, attention


def test_masked_native_remote_seeds_sum_to_the_unpartitioned_write():
    operator = FakeOperator()
    masks = (
        np.array([True, False, False, False, False]),
        np.array([False, True, True, False, False]),
        np.array([False, False, False, True, True]),
    )
    full_message, full_mass = DifferentialLayer.remote_seeds(operator, [(0, 1)], 1)[0, 1]
    components = [
        DifferentialLayer.remote_seeds(operator, [(0, 1)], 1, source_mask=mask)[0, 1]
        for mask in masks
    ]

    torch.testing.assert_close(sum(value[0] for value in components), full_message)
    np.testing.assert_allclose(sum(value[1] for value in components), full_mass)


def test_transition_seed_matches_discovery_normalization_and_closes_by_source():
    operator = FakeOperator()
    masks = (
        np.array([True, False, False, False, False]),
        np.array([False, True, True, False, False]),
        np.array([False, False, False, True, True]),
    )
    full_message, full_delta = DifferentialLayer.remote_transition_seeds(operator, [(0, 1)], 1)[
        0, 1
    ]
    components = [
        DifferentialLayer.remote_transition_seeds(operator, [(0, 1)], 1, source_mask=mask)[0, 1]
        for mask in masks
    ]
    expected_delta = np.array([-0.1, -0.1, 0.2, 0.0, 0.0])

    np.testing.assert_allclose(full_delta, expected_delta, atol=1e-7)
    torch.testing.assert_close(full_message, torch.tensor([0.5, 0.0]))
    torch.testing.assert_close(sum(value[0] for value in components), full_message)
    np.testing.assert_allclose(sum(value[1] for value in components), full_delta)


def _artifact(*, baseline=-1.0, effects=(0.0, 0.1, 0.0, -0.8)):
    group_count = len(SOURCE_GROUPS)
    margins = np.zeros((group_count, 3, 3, 1), dtype=np.float32)
    for index, effect in enumerate(effects):
        margins[index, 0, 2, 0] = effect
    layers = np.zeros((group_count, 3, 3, 3, 1), dtype=np.float32)
    layers[:, 0, 2, -1, 0] = effects
    roots = np.zeros((group_count, 6), dtype=np.float32)
    roots[:, 0] = [0.0, 0.4, -0.1, 0.5]
    transition = margins.sum(0)
    current = np.zeros_like(transition)
    current[0, 1, 0] = -0.2
    unit_margins = np.zeros((2, 3, 3, 1), dtype=np.float32)
    unit_margins[0, 0, 2, 0] = 0.3
    unit_margins[1, 0, 2, 0] = -0.2
    unit_roots = np.zeros((2, 6), dtype=np.float32)
    unit_roots[:, 0] = [0.2, -0.1]
    return {
        "source_group_names": np.array(SOURCE_GROUPS),
        "source_group_margin_response": margins,
        "source_group_layer_margin_response": layers,
        "source_group_root_coefficient": roots,
        "source_group_seed_norm": np.array([0.0, 0.3, 0.1, 0.4]),
        "source_unit_ids": np.array([4, 7]),
        "source_unit_roles": np.array(["content", "content"]),
        "source_unit_margin_response": unit_margins,
        "source_unit_root_coefficient": unit_roots,
        "source_unit_seed_norm": np.array([0.2, 0.1]),
        "transition_margin_response": transition,
        "current_remote_margin_response": current,
        "baseline_margin": np.array([baseline]),
        "target_position": np.array([9]),
        "positive_id": np.array([4]),
        "negative_id": np.array([5]),
        "semantic_source_roles_available": np.array(True),
        "closure_pass": np.array(True),
        "closure_margin_max_abs": np.array(0.0),
        "closure_layer_max_abs": np.array(0.0),
        "closure_root_max_abs": np.array(0.0),
    }


def test_mechanism_row_uses_temporal_phase_before_coarse_provenance():
    row = mechanism_target_row(
        _artifact(),
        0,
        identity={"source_id": "s", "event_position": 8},
        label=1,
        target_phase="onset",
        before_next_event=True,
    )

    assert row["onset_aligned"] is True
    assert row["transition_remote_effect"] == pytest.approx(-0.7)
    assert row["transition_multi_hop_effect"] == pytest.approx(-0.7)
    assert row["response_history_transition_margin_effect"] == pytest.approx(-0.8)
    assert row["binding_identified"] is False
    assert "primary_failure_mode" not in row


def test_opposing_group_effects_are_exposed_as_cancellation_not_no_use():
    row = mechanism_target_row(
        _artifact(effects=(0.0, 0.5, 0.0, -0.5)),
        0,
        identity={"source_id": "s", "event_position": 8},
        label=0,
        target_phase="normal",
        before_next_event=True,
    )

    assert row["transition_remote_effect"] == pytest.approx(0.0)
    assert row["source_group_effect_l1"] == pytest.approx(1.0)
    assert row["source_group_effect_cancellation"] == pytest.approx(1.0)


def test_source_unit_rows_preserve_opposing_document_effects_and_rank_them():
    rows = source_unit_target_rows(
        _artifact(), 0, identity={"source_id": "s", "target_position": 9}
    )

    assert [row["source_unit_id"] for row in rows] == [4, 7]
    assert [row["transition_margin_effect"] for row in rows] == pytest.approx([0.3, -0.2])
    assert [row["absolute_effect_rank"] for row in rows] == [1, 2]


def _full_trace(scale=1.0, root_scale=1.0, seed_kind="transition_delta_remote"):
    return {
        "event_sites": np.array([[0, 0, 2]]),
        "event_row": np.array(2),
        "event_position": np.array(7),
        "target_position": np.array([8]),
        "explicit_contrast": np.array([True]),
        "margin_response": np.ones((3, 3, 1), dtype=np.float32) * scale,
        "layer_margin_response": np.ones((3, 3, 2, 1), dtype=np.float32) * scale,
        "root_attention_sum": np.ones(5, dtype=np.float32) * root_scale,
        "baseline_margin": np.array([-1.0]),
        "positive_id": np.array([3]),
        "negative_id": np.array([4]),
        "seed_norm": np.array(0.2),
        "variants": np.array(["full", "fixed_qk", "no_mlp_paths"]),
        "hop_names": np.array(["0", "1", "2+"]),
        "seed_kind": np.array(seed_kind),
    }


def _partitions():
    coarse = SimpleNamespace(
        names=SOURCE_GROUPS,
        semantic_roles_available=True,
    )
    units = SimpleNamespace(
        ids=np.array([], dtype=int),
        roles=(),
    )
    return coarse, units


def test_source_components_must_close_each_transition_field_independently():
    current = _full_trace(seed_kind="current_remote")
    transition = _full_trace()
    components = []
    for _ in SOURCE_GROUPS:
        value = _full_trace(scale=0.25, root_scale=0.25)
        components.append(value)
    coarse, units = _partitions()

    combined = MechanismAuditor(MechanismConfig(device="cpu"))._combine(
        current, transition, components, [], coarse, units, current["event_sites"]
    )

    assert bool(combined["closure_pass"])
    assert bool(combined["closure_margin_pass"])
    assert bool(combined["closure_layer_pass"])
    assert bool(combined["closure_root_pass"])


def test_large_root_scale_cannot_hide_margin_closure_failure():
    current = _full_trace(root_scale=1e9, seed_kind="current_remote")
    transition = _full_trace(root_scale=1e9)
    components = [_full_trace(scale=0.20, root_scale=2.5e8) for _ in SOURCE_GROUPS]
    coarse, units = _partitions()

    combined = MechanismAuditor(MechanismConfig(device="cpu"))._combine(
        current, transition, components, [], coarse, units, current["event_sites"]
    )

    assert not bool(combined["closure_margin_pass"])
    assert bool(combined["closure_root_pass"])
    assert not bool(combined["closure_pass"])


def test_large_target_cannot_hide_another_targets_pointwise_closure_failure():
    auditor = MechanismAuditor(MechanismConfig(device="cpu"))

    _, _, passed = auditor._closure(
        np.array([1e9, 1.0]),
        np.array([1e9, 0.0]),
    )

    assert passed is False


def test_mechanism_rejects_a_current_trace_from_stale_layer_head_sites():
    current = _full_trace(seed_kind="current_remote")
    transition = _full_trace()
    components = [_full_trace(scale=0.25, root_scale=0.25) for _ in SOURCE_GROUPS]
    coarse, units = _partitions()

    with pytest.raises(ValueError, match="stale anchor sites"):
        MechanismAuditor(MechanismConfig(device="cpu"))._combine(
            current,
            transition,
            components,
            [],
            coarse,
            units,
            np.array([[1, 0, 2]]),
        )


def test_mechanism_auditor_reuses_frozen_anchors_and_writes_label_free_components(
    tmp_path, monkeypatch
):
    sample = AuditSample("test", "QA", "7", "source-7", Path("7.npz"), 3, 3)
    capture_path = tmp_path / "capture.npz"
    capture_fields = {
        "token_ids": np.arange(6),
        "token_text": np.array(["a", "b", "c", "d", "e", "<x>"]),
        "row_position": np.array([3, 4, 5]),
        "response_start": np.array(3),
        "special_mask": np.array([False, False, False, False, False, True]),
        "evidence_mask": np.array([False, True, False, False, False, False]),
        "source_kind": np.array(["constraint", "content", "other", "", "", ""]),
    }
    np.savez_compressed(capture_path, **capture_fields)
    qk_path = tmp_path / "capture.qk.npz"
    history_path = tmp_path / "capture.history.npz"
    states_path = tmp_path / "capture.states.npz"
    for companion in (qk_path, history_path, states_path):
        np.savez_compressed(companion, identity=np.array(1))
    run = tmp_path / "run"
    store = ArtifactStore(run)
    folder = store.sample_path(sample, "events.npz").parent
    transition_path = folder / "transitions.npz"
    event_path = folder / "events.npz"
    trace_path = folder / "traces/event_4.npz"
    store.write_npz(
        transition_path,
        remote_gain=np.array([[[0.0, 0.2, 0.0]]]),
        previous_local_mass=np.ones((1, 1, 3)),
    )
    store.write_npz(
        event_path,
        anchor=np.array([False, True, False]),
        row_position=np.array([3, 4, 5]),
    )
    full_margin = np.ones((3, 3, 2), dtype=np.float32)
    full_layers = np.ones((3, 3, 2, 2), dtype=np.float32)
    full_roots = np.array([1, 1, 1, 1, 1, 0], dtype=np.float32)
    store.write_npz(
        trace_path,
        event_sites=np.array([[0, 0, 1]]),
        event_row=np.array(1),
        event_position=np.array(4),
        target_position=np.array([4, 5]),
        explicit_contrast=np.array([False, True]),
        margin_response=full_margin,
        layer_margin_response=full_layers,
        root_attention_sum=full_roots,
        baseline_margin=np.array([0.0, -1.0]),
        positive_id=np.array([3, 4]),
        negative_id=np.array([5, 6]),
        variants=np.array(["full", "fixed_qk", "no_mlp_paths"]),
        hop_names=np.array(["0", "1", "2+"]),
        seed_kind=np.array("current_remote"),
    )
    store.write_json(
        run / "index.json",
        {
            "method_schema": "reanchor/max-null-episode@1",
            "settings": {
                "window": 1,
                "site_gain_floor": 0.1,
                "broad_gain_floor": 0.05,
                "local_floor": 0.5,
            },
            "sample_artifacts": [
                {
                    "key": sample.key,
                    "anchors": 1,
                    "transitions": str(transition_path.relative_to(run)),
                    "events": str(event_path.relative_to(run)),
                }
            ],
        },
    )
    contrast_path = tmp_path / "contrasts.json"
    contrast_path.write_text(
        json.dumps({sample.key: [{"target": 5, "positive_id": 4, "negative_id": 6}]})
    )
    _, digest = read_contrasts(str(contrast_path))
    store.write_json(
        run / "tracing.json",
        {
            "trace_schema": "reanchor/analytic-trace@2",
            "settings": {"contrast_sha256": digest},
            "sample_artifacts": [{"key": sample.key, "traces": [str(trace_path.relative_to(run))]}],
        },
    )

    class FakeDataset:
        model_path = Path("checkpoint")

        def completed_samples(self, *, require_states):
            assert require_states
            return (sample,), {"planned": 1, "completed": 1, "skipped": 0}

        def paths(self, current):
            assert current == sample
            return SimpleNamespace(
                compact=capture_path,
                qk=qk_path,
                history=history_path,
                states=states_path,
            )

    class FakeCache:
        def __init__(self, paths, weights):
            del paths, weights
            self.trace = capture_fields

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    seen_masks = []

    def fake_trace(cache, coordinates, *, source_masks, seed_kind, **options):
        del coordinates, options
        assert seed_kind == "transition_delta_remote"
        seen_masks.append(source_masks.copy())
        result = []
        for source_mask in source_masks:
            scale = float(source_mask.sum()) / 5
            result.append(
                [
                    {
                        "event_sites": np.array([[0, 0, 1]]),
                        "event_row": np.array(1),
                        "event_position": np.array(4),
                        "target_position": np.array([4, 5]),
                        "explicit_contrast": np.array([False, True]),
                        "positive_id": np.array([3, 4]),
                        "negative_id": np.array([5, 6]),
                        "baseline_margin": np.array([0.0, -1.0]),
                        "margin_response": full_margin * scale,
                        "layer_margin_response": full_layers * scale,
                        "root_attention_sum": source_mask.astype(np.float32),
                        "seed_norm": np.array(scale),
                        "variants": np.array(["full", "fixed_qk", "no_mlp_paths"]),
                        "hop_names": np.array(["0", "1", "2+"]),
                        "seed_kind": np.array("transition_delta_remote"),
                    }
                ]
            )
        return result

    import reanchor.tracing.mechanism as module

    monkeypatch.setattr(module, "CheckpointWeights", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "NativeCache", FakeCache)
    monkeypatch.setattr(module, "trace_events", fake_trace)
    config = MechanismConfig(device="cpu", contrast_file=str(contrast_path))

    summary = MechanismAuditor(config).run(FakeDataset(), run)

    assert summary["anchors_audited"] == 1
    assert summary["outcome_token_labels_used_for_mechanism_audit"] is False
    assert summary["explicit_correctness_contrasts_used"] is True
    assert len(seen_masks) == 1
    assert len(seen_masks[0]) == 1 + len(SOURCE_GROUPS)
    mechanism_path = folder / "mechanisms/event_4.npz"
    with np.load(mechanism_path, allow_pickle=False) as mechanism:
        assert bool(mechanism["closure_pass"])
        assert not bool(mechanism["outcome_token_labels_used"])
        np.testing.assert_array_equal(mechanism["source_group_names"], SOURCE_GROUPS)
        assert str(mechanism["seed_estimand"]) == "discovery_aligned_remote_attention_delta"

    resumed = MechanismAuditor(config).run(FakeDataset(), run)
    assert resumed["anchors_resumed"] == 1

    np.savez_compressed(capture_path, **{**capture_fields, "source_kind": np.array(["other"] * 6)})
    with pytest.raises(ValueError, match="different identity"):
        MechanismAuditor(config).run(FakeDataset(), run)
