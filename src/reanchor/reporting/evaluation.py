"""Join outcomes after selection and summarize at the independent-source level."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reanchor.artifacts import ArtifactStore
from reanchor.capture.protocol import AuditDataset

from .mechanisms import mechanism_target_row, source_unit_target_rows
from .records import event_records, future_records, load_labels
from .statistics import received_then_overridden, source_mean_summary, source_ratio_summary


@dataclass(frozen=True)
class ReportConfig:
    bootstrap: int = 1000
    seed: int = 0
    future_horizons: tuple[tuple[int, int], ...] = ((1, 4), (5, 16))

    def __post_init__(self):
        if self.bootstrap < 0 or any(
            lower < 1 or upper < lower for lower, upper in self.future_horizons
        ):
            raise ValueError("bootstrap and future horizons are invalid")


class ReportBuilder:
    """Create auditable CSV rows and source-balanced summaries after discovery."""

    SCHEMA = "reanchor/source-balanced-report@1"
    _EVENT_FIELDS = (
        "split",
        "task",
        "sample_id",
        "source_id",
        "query_position",
        "target_position",
        "label",
        "significant",
        "anchor",
        "episode_id",
        "channel",
        "reanchor_type",
        "family_p_value",
        "sparse_score",
        "broad_score",
    )
    _MECHANISM_BASE_FIELDS = (
        "split",
        "task",
        "sample_id",
        "source_id",
        "event_position",
        "target_position",
        "event_target_offset",
        "target_phase",
        "before_next_event",
        "onset_aligned",
        "rollout_target",
        "label",
        "positive_id",
        "negative_id",
        "baseline_margin",
        "transition_remote_effect",
        "transition_zero_hop_effect",
        "transition_one_hop_effect",
        "transition_multi_hop_effect",
        "rollout_multi_hop_error_promoting",
        "current_remote_write_effect",
        "current_remote_zero_hop_effect",
        "current_remote_one_hop_effect",
        "current_remote_multi_hop_effect",
        "linearized_margin_without_current_remote_write",
        "source_group_effect_l1",
        "source_group_effect_cancellation",
        "semantic_source_roles_available",
        "closure_pass",
        "closure_margin_max_abs",
        "closure_layer_max_abs",
        "closure_root_max_abs",
        "binding_identified",
        "exact_intervention_run",
    )
    _SOURCE_UNIT_FIELDS = (
        "split",
        "task",
        "sample_id",
        "source_id",
        "event_position",
        "target_position",
        "event_target_offset",
        "target_phase",
        "label",
        "closure_pass",
        "source_unit_id",
        "source_role",
        "transition_coefficient",
        "transition_coefficient_l1",
        "seed_norm",
        "transition_margin_effect",
        "absolute_effect_rank",
    )

    def __init__(self, config: ReportConfig = ReportConfig()):
        self.config = config

    def run(self, dataset: AuditDataset, run_root: str | Path) -> dict:
        run_root = Path(run_root).resolve()
        discovery = json.loads((run_root / "index.json").read_text(encoding="utf-8"))
        if discovery.get("method_schema") != "reanchor/max-null-episode@1":
            raise ValueError("reporting requires calibrated reanchor discovery artifacts")
        samples = {sample.key: sample for sample in dataset.samples}
        store = ArtifactStore(run_root)
        event_rows = []
        future_rows = []
        labelled_samples = 0
        for entry in discovery["sample_artifacts"]:
            sample = samples.get(entry["key"])
            if sample is None:
                raise ValueError(f"capture no longer contains {entry['key']}")
            transitions = store.read_npz(self._inside(run_root, entry["transitions"]))
            events = store.read_npz(self._inside(run_root, entry["events"]))
            labels = load_labels(dataset, sample)
            labelled_samples += int(labels is not None)
            event_rows.extend(event_records(sample, transitions, events, labels))
            future_rows.extend(
                future_records(
                    sample,
                    transitions,
                    events,
                    labels,
                    self.config.future_horizons,
                )
            )

        groups = {}
        for group in sorted({f"{row['split']}/{row['task']}" for row in event_rows}):
            rows = [row for row in event_rows if f"{row['split']}/{row['task']}" == group]
            anchors = [row for row in rows if row["anchor"]]
            known_anchors = [row for row in anchors if row["label"] in (0, 1)]
            morphology = {}
            kinds = sorted({row["reanchor_type"] for row in anchors})
            for kind in kinds:
                kind_anchors = [row for row in known_anchors if row["reanchor_type"] == kind]
                kind_future = [
                    row
                    for row in future_rows
                    if row["group"] == group and row["reanchor_type"] == kind
                ]
                morphology[kind] = {
                    "anchor_target_hallucination": source_ratio_summary(
                        [dict(row, hallucinated=row["label"] == 1) for row in kind_anchors],
                        "hallucinated",
                        bootstrap=self.config.bootstrap,
                        seed=self.config.seed,
                    ),
                    "future_hallucination": {
                        horizon: source_ratio_summary(
                            [row for row in kind_future if row["horizon"] == horizon],
                            "hallucinated",
                            bootstrap=self.config.bootstrap,
                            seed=self.config.seed,
                        )
                        for horizon in self._horizon_names()
                    },
                }
            groups[group] = {
                "event_incidence": source_ratio_summary(
                    rows,
                    "significant",
                    bootstrap=self.config.bootstrap,
                    seed=self.config.seed,
                ),
                "event_incidence_nonhallucinated": source_ratio_summary(
                    [row for row in rows if row["label"] == 0],
                    "significant",
                    bootstrap=self.config.bootstrap,
                    seed=self.config.seed,
                ),
                "event_incidence_hallucinated": source_ratio_summary(
                    [row for row in rows if row["label"] == 1],
                    "significant",
                    bootstrap=self.config.bootstrap,
                    seed=self.config.seed,
                ),
                "anchor_target_hallucination": source_ratio_summary(
                    [dict(row, hallucinated=row["label"] == 1) for row in known_anchors],
                    "hallucinated",
                    bootstrap=self.config.bootstrap,
                    seed=self.config.seed,
                ),
                "eligible_tokens": len(rows),
                "significant_transitions": sum(row["significant"] for row in rows),
                "anchors": len(anchors),
                "morphology_outcomes": morphology,
            }
        trace_summary = self._trace_summary(dataset, run_root, store)
        mechanism_summary, mechanism_rows, source_unit_rows = self._mechanism_summary(
            dataset, run_root, store
        )
        summary = {
            "report_schema": self.SCHEMA,
            "discovery_schema": discovery["method_schema"],
            "labels_joined_only_in_reporting": True,
            "labelled_samples": labelled_samples,
            "groups": groups,
            "causal_response": trace_summary,
            "mechanism_audit": mechanism_summary,
            "interpretation": {
                "observed_runner": "candidate preference, not correctness",
                "explicit_contrast": "correctness only when candidate IDs were defined externally",
                "uncertainty_unit": "independent source_id",
            },
        }
        reports = run_root / "reports"
        store.write_text(reports / "events.csv", self._csv(event_rows))
        if mechanism_rows is not None:
            store.write_text(reports / "mechanisms.csv", self._mechanism_csv(mechanism_rows))
            store.write_text(
                reports / "mechanism_source_units.csv",
                self._csv_fields(source_unit_rows, self._SOURCE_UNIT_FIELDS),
            )
        store.write_json(reports / "summary.json", summary)
        return summary

    def _horizon_names(self):
        return tuple(f"{lower}-{upper}" for lower, upper in self.config.future_horizons)

    def _trace_summary(self, dataset, run_root: Path, store: ArtifactStore) -> dict:
        path = run_root / "tracing.json"
        if not path.is_file():
            return {"status": "not_run"}
        manifest = json.loads(path.read_text(encoding="utf-8"))
        samples = {sample.key: sample for sample in dataset.samples}
        rows = []
        reversals = []
        for entry in manifest["sample_artifacts"]:
            sample = samples[entry["key"]]
            labels = load_labels(dataset, sample)
            if labels is None:
                continue
            special = np.asarray(
                dataset.load_metadata(sample, "special_mask")["special_mask"], dtype=bool
            )
            for relative in entry.get("traces", []):
                trace = store.read_npz(self._inside(run_root, relative))
                targets = np.asarray(trace["target_position"])
                if np.any(targets < 0) or np.any(targets >= len(special)):
                    raise ValueError(f"{sample.key}: trace target lies outside captured tokens")
                after = targets > int(trace["event_position"])
                label_index = targets - sample.response_start
                ordinary_after = after & ~special[targets]
                explicit = np.asarray(trace["explicit_contrast"], dtype=bool)
                if explicit.shape != targets.shape:
                    raise ValueError(f"{sample.key}: explicit contrast axis differs from targets")
                for target_index in np.flatnonzero(ordinary_after & explicit):
                    if "layer_margin_response" not in trace:
                        continue
                    trajectory = trace["layer_margin_response"][0, :, :, target_index].sum(0)
                    reversals.append(
                        {
                            "source_id": sample.source_id,
                            "reversed": received_then_overridden(trajectory),
                        }
                    )
                labelled = ordinary_after & (label_index >= 0) & (label_index < len(labels))
                for target_index in np.flatnonzero(labelled):
                    label = int(labels[label_index[target_index]])
                    if label not in (0, 1):
                        continue
                    effect = float(trace["margin_response"][0, :, target_index].sum())
                    rows.append(
                        {
                            "source_id": sample.source_id,
                            "label": label,
                            "effect": effect,
                            "contrast": "explicit_candidate"
                            if explicit[target_index]
                            else "observed_runner",
                        }
                    )
        result = {"status": "complete", "targets": len(rows)}
        for contrast in ("observed_runner", "explicit_candidate"):
            for label, name in ((0, "nonhallucinated"), (1, "hallucinated")):
                result[f"{contrast}_margin_effect_{name}"] = source_mean_summary(
                    [row for row in rows if row["contrast"] == contrast and row["label"] == label],
                    "effect",
                    bootstrap=self.config.bootstrap,
                    seed=self.config.seed,
                )
        result["explicit_candidate_received_then_overridden"] = source_ratio_summary(
            reversals,
            "reversed",
            bootstrap=self.config.bootstrap,
            seed=self.config.seed,
        )
        return result

    def _mechanism_summary(self, dataset, run_root: Path, store: ArtifactStore):
        path = run_root / "mechanism.json"
        if not path.is_file():
            return {"status": "not_run"}, None, None
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("mechanism_schema") != "reanchor/transition-mechanism-audit@2":
            raise ValueError("reporting found an unsupported mechanism audit schema")
        samples = {sample.key: sample for sample in dataset.samples}
        rows = []
        unit_rows = []
        for entry in manifest["sample_artifacts"]:
            sample = samples[entry["key"]]
            labels = load_labels(dataset, sample)
            if labels is None:
                continue
            special = np.asarray(
                dataset.load_metadata(sample, "special_mask")["special_mask"], dtype=bool
            )
            artifacts = [
                store.read_npz(self._inside(run_root, relative))
                for relative in entry.get("mechanisms", [])
            ]
            artifacts.sort(key=lambda value: int(value["event_position"]))
            event_positions = [int(value["event_position"]) for value in artifacts]
            for artifact_index, artifact in enumerate(artifacts):
                targets = np.asarray(artifact["target_position"])
                if np.any(targets < 0) or np.any(targets >= len(special)):
                    raise ValueError(f"{sample.key}: mechanism target lies outside captured tokens")
                event_position = int(artifact["event_position"])
                next_event = (
                    event_positions[artifact_index + 1]
                    if artifact_index + 1 < len(event_positions)
                    else None
                )
                after = targets > event_position
                label_index = targets - sample.response_start
                explicit = np.asarray(artifact["explicit_contrast"], dtype=bool)
                eligible = (
                    after
                    & explicit
                    & ~special[targets]
                    & (label_index >= 0)
                    & (label_index < len(labels))
                )
                identity = {
                    "split": sample.split,
                    "task": sample.task,
                    "sample_id": sample.sample_id,
                    "source_id": sample.source_id,
                    "event_position": event_position,
                }
                for target_index in np.flatnonzero(eligible):
                    response_index = int(label_index[target_index])
                    label = int(labels[response_index])
                    if label in (0, 1):
                        phase = self._target_phase(labels, response_index)
                        target_position = int(targets[target_index])
                        before_next_event = next_event is None or target_position <= next_event
                        row = mechanism_target_row(
                            artifact,
                            int(target_index),
                            identity=identity,
                            label=label,
                            target_phase=phase,
                            before_next_event=before_next_event,
                        )
                        rows.append(row)
                        unit_rows.extend(
                            source_unit_target_rows(
                                artifact,
                                int(target_index),
                                identity={
                                    **identity,
                                    "target_position": target_position,
                                    "event_target_offset": row["event_target_offset"],
                                    "target_phase": phase,
                                    "label": label,
                                    "closure_pass": row["closure_pass"],
                                },
                            )
                        )

        valid = [row for row in rows if row["closure_pass"]]
        phase_names = ("normal", "onset", "continuing", "hallucinated_boundary_unknown")
        result = {
            "status": "complete",
            "targets": len(rows),
            "nonhallucinated_targets": sum(row["label"] == 0 for row in rows),
            "hallucinated_targets": sum(row["label"] == 1 for row in rows),
            "closure_failures": sum(not row["closure_pass"] for row in rows),
            "estimand": "discovery_aligned_remote_attention_delta",
            "coarse_source_groups_are_mechanism_classes": False,
            "exact_intervention_run": False,
            "binding_identified": False,
            "binding_requirement": (
                "matched constraint counterfactual or bidirectional intervention"
            ),
            "phase_effects": {},
            "source_groups_by_phase": {},
            "same_event_onset_to_rollout": self._temporal_chains(valid),
        }
        for phase in phase_names:
            phase_rows = [row for row in valid if row["target_phase"] == phase]
            result["phase_effects"][phase] = {
                "transition_remote_effect": source_mean_summary(
                    phase_rows,
                    "transition_remote_effect",
                    bootstrap=self.config.bootstrap,
                    seed=self.config.seed,
                ),
                "transition_multi_hop_effect": source_mean_summary(
                    phase_rows,
                    "transition_multi_hop_effect",
                    bootstrap=self.config.bootstrap,
                    seed=self.config.seed,
                ),
                "current_remote_write_effect": source_mean_summary(
                    phase_rows,
                    "current_remote_write_effect",
                    bootstrap=self.config.bootstrap,
                    seed=self.config.seed,
                ),
            }
            result["source_groups_by_phase"][phase] = {}
            for name in ("constraint", "content", "other_prompt", "response_history"):
                result["source_groups_by_phase"][phase][name] = {
                    "transition_coefficient_l1_share": source_mean_summary(
                        phase_rows,
                        f"{name}_transition_coefficient_l1_share",
                        bootstrap=self.config.bootstrap,
                        seed=self.config.seed,
                    ),
                    "correct_minus_error_transition_effect": source_mean_summary(
                        phase_rows,
                        f"{name}_transition_margin_effect",
                        bootstrap=self.config.bootstrap,
                        seed=self.config.seed,
                    ),
                }
        return result, rows, unit_rows

    def _temporal_chains(self, rows) -> dict:
        grouped = {}
        for row in rows:
            key = (
                row["split"],
                row["task"],
                row["sample_id"],
                row["event_position"],
            )
            grouped.setdefault(key, []).append(row)
        chain_rows = []
        events_with_onset = events_with_rollout = 0
        for event_rows in grouped.values():
            onset = [row for row in event_rows if row["onset_aligned"]]
            rollout = [row for row in event_rows if row["rollout_target"]]
            events_with_onset += bool(onset)
            events_with_rollout += bool(rollout)
            if not onset:
                continue
            onset_effect = float(np.mean([row["transition_remote_effect"] for row in onset]))
            rollout_effect = (
                float(np.mean([row["transition_multi_hop_effect"] for row in rollout]))
                if rollout
                else np.nan
            )
            chain_rows.append(
                {
                    "source_id": onset[0]["source_id"],
                    "onset_transition_effect": onset_effect,
                    "rollout_multi_hop_effect": rollout_effect,
                    "candidate_chain": bool(rollout and onset_effect < 0 and rollout_effect < 0),
                }
            )
        return {
            "status": "exploratory_linearized_candidate_only",
            "events": len(grouped),
            "events_with_explicit_onset_target": events_with_onset,
            "events_with_pre_next_event_rollout_target": events_with_rollout,
            "onset_transition_effect": source_mean_summary(
                chain_rows,
                "onset_transition_effect",
                bootstrap=self.config.bootstrap,
                seed=self.config.seed,
            ),
            "rollout_multi_hop_effect": source_mean_summary(
                chain_rows,
                "rollout_multi_hop_effect",
                bootstrap=self.config.bootstrap,
                seed=self.config.seed,
            ),
            "candidate_chain_rate": source_ratio_summary(
                chain_rows,
                "candidate_chain",
                bootstrap=self.config.bootstrap,
                seed=self.config.seed,
            ),
            "claim_supported": False,
            "missing_for_causal_claim": (
                "matched pseudo-onsets, exact route interventions, free-run decoding, "
                "and held-out cross-model replication"
            ),
        }

    @staticmethod
    def _target_phase(labels, response_index: int) -> str:
        label = int(labels[response_index])
        if label == 0:
            return "normal"
        if response_index == 0 or int(labels[response_index - 1]) not in (0, 1):
            return "hallucinated_boundary_unknown"
        return "onset" if int(labels[response_index - 1]) == 0 else "continuing"

    def _csv(self, rows) -> str:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=self._EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        return stream.getvalue()

    def _mechanism_csv(self, rows) -> str:
        group_fields = tuple(
            f"{name}_{suffix}"
            for name in ("constraint", "content", "other_prompt", "response_history")
            for suffix in (
                "transition_coefficient",
                "transition_coefficient_l1_share",
                "seed_norm",
                "transition_margin_effect",
                "layer_sign_reversal",
            )
        )
        return self._csv_fields(rows, (*self._MECHANISM_BASE_FIELDS, *group_fields))

    @staticmethod
    def _csv_fields(rows, fields) -> str:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        return stream.getvalue()

    @staticmethod
    def _inside(root: Path, relative: str) -> Path:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"run artifact is outside run root: {relative}") from error
        return path
