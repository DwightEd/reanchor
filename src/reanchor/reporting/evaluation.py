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
        summary = {
            "report_schema": self.SCHEMA,
            "discovery_schema": discovery["method_schema"],
            "labels_joined_only_in_reporting": True,
            "labelled_samples": labelled_samples,
            "groups": groups,
            "causal_response": trace_summary,
            "interpretation": {
                "observed_runner": "candidate preference, not correctness",
                "explicit_contrast": "correctness only when candidate IDs were defined externally",
                "uncertainty_unit": "independent source_id",
            },
        }
        reports = run_root / "reports"
        store.write_text(reports / "events.csv", self._csv(event_rows))
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

    def _csv(self, rows) -> str:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=self._EVENT_FIELDS)
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
