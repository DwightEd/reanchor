"""Join outcomes after selection and summarize at the independent-source level."""

from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reanchor.artifacts import ArtifactStore
from reanchor.capture.protocol import AuditDataset


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


def _interval(values: np.ndarray, bootstrap: int, seed: int) -> tuple[float, float]:
    if len(values) == 1 or bootstrap == 0:
        value = float(values.mean())
        return value, value
    random = np.random.default_rng(seed)
    draws = random.choice(values, size=(bootstrap, len(values)), replace=True).mean(1)
    lower, upper = np.quantile(draws, (0.025, 0.975))
    return float(lower), float(upper)


def source_ratio_summary(rows, numerator: str, *, bootstrap: int, seed: int) -> dict:
    """Average within-source ratios so prolific sources do not dominate."""

    by_source = defaultdict(list)
    for row in rows:
        by_source[str(row["source_id"])].append(bool(row[numerator]))
    if not by_source:
        return {"estimate": None, "ci95": [None, None], "sources": 0, "observations": 0}
    values = np.array([np.mean(items) for items in by_source.values()], dtype=float)
    lower, upper = _interval(values, bootstrap, seed)
    return {
        "estimate": float(values.mean()),
        "ci95": [lower, upper],
        "sources": len(values),
        "observations": sum(map(len, by_source.values())),
    }


def source_mean_summary(rows, value: str, *, bootstrap: int, seed: int) -> dict:
    by_source = defaultdict(list)
    for row in rows:
        number = float(row[value])
        if np.isfinite(number):
            by_source[str(row["source_id"])].append(number)
    if not by_source:
        return {"estimate": None, "ci95": [None, None], "sources": 0, "observations": 0}
    values = np.array([np.mean(items) for items in by_source.values()], dtype=float)
    lower, upper = _interval(values, bootstrap, seed)
    return {
        "estimate": float(values.mean()),
        "ci95": [lower, upper],
        "sources": len(values),
        "observations": sum(map(len, by_source.values())),
    }


def received_then_overridden(layer_response: np.ndarray) -> bool:
    """Detect a positive intermediate explicit-candidate effect ending nonpositive."""

    values = np.asarray(layer_response, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return False
    return bool(np.max(finite[:-1]) > 0 and finite[-1] <= 0)


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
            labels = self._labels(dataset, sample)
            labelled_samples += int(labels is not None)
            event_rows.extend(self._event_rows(sample, transitions, events, labels))
            future_rows.extend(self._future_rows(sample, transitions, events, labels))

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

    def _event_rows(self, sample, transitions, events, labels):
        positions = np.asarray(events["row_position"])
        eligible = np.asarray(events["eligible"], dtype=bool)
        response_start = int(transitions["response_start"])
        token_count = len(transitions.get("special_mask", []))
        special = np.asarray(
            transitions.get("special_mask", np.zeros(max(token_count, positions.max() + 2), bool))
        )
        rows = []
        for index in np.flatnonzero(eligible):
            target = int(positions[index]) + 1
            label_index = target - response_start
            label = ""
            if (
                labels is not None
                and 0 <= label_index < len(labels)
                and (target >= len(special) or not special[target])
                and int(labels[label_index]) in (0, 1)
            ):
                label = int(labels[label_index])
            rows.append(
                {
                    "split": sample.split,
                    "task": sample.task,
                    "sample_id": sample.sample_id,
                    "source_id": sample.source_id,
                    "query_position": int(positions[index]),
                    "target_position": target,
                    "label": label,
                    "significant": bool(events["significant"][index]),
                    "anchor": bool(events["anchor"][index]),
                    "episode_id": int(events["episode_id"][index]),
                    "channel": str(events["channel"][index]),
                    "reanchor_type": str(events["reanchor_type"][index]),
                    "family_p_value": float(events["family_p_value"][index]),
                    "sparse_score": float(events["sparse_score"][index]),
                    "broad_score": float(events["broad_score"][index]),
                }
            )
        return rows

    def _future_rows(self, sample, transitions, events, labels):
        if labels is None:
            return []
        response_start = int(transitions["response_start"])
        positions = np.asarray(events["row_position"])
        special = np.asarray(
            transitions.get("special_mask", np.zeros(response_start + len(labels), bool))
        )
        rows = []
        for anchor_index in np.flatnonzero(events["anchor"]):
            first_label = int(positions[anchor_index]) + 1 - response_start
            kind = str(events["reanchor_type"][anchor_index])
            for (lower, upper), horizon in zip(self.config.future_horizons, self._horizon_names()):
                for offset in range(lower, upper + 1):
                    label_index = first_label + offset
                    token_position = response_start + label_index
                    if not 0 <= label_index < len(labels):
                        continue
                    if token_position < len(special) and special[token_position]:
                        continue
                    label = int(labels[label_index])
                    if label in (0, 1):
                        rows.append(
                            {
                                "source_id": sample.source_id,
                                "group": f"{sample.split}/{sample.task}",
                                "reanchor_type": kind,
                                "horizon": horizon,
                                "hallucinated": label == 1,
                            }
                        )
        return rows

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
            labels = self._labels(dataset, sample)
            if labels is None:
                continue
            for relative in entry.get("traces", []):
                trace = store.read_npz(self._inside(run_root, relative))
                targets = np.asarray(trace["target_position"])
                after = targets > int(trace["event_position"])
                label_index = targets - sample.response_start
                valid = after & (label_index >= 0) & (label_index < len(labels))
                for target_index in np.flatnonzero(valid):
                    label = int(labels[label_index[target_index]])
                    if label not in (0, 1):
                        continue
                    effect = float(trace["margin_response"][0, :, target_index].sum())
                    rows.append(
                        {
                            "source_id": sample.source_id,
                            "label": label,
                            "effect": effect,
                        }
                    )
                    if "layer_margin_response" in trace and bool(
                        trace["explicit_contrast"][target_index]
                    ):
                        trajectory = trace["layer_margin_response"][0, :, :, target_index].sum(0)
                        reversals.append(
                            {
                                "source_id": sample.source_id,
                                "reversed": received_then_overridden(trajectory),
                            }
                        )
        result = {"status": "complete", "targets": len(rows)}
        for label, name in ((0, "nonhallucinated"), (1, "hallucinated")):
            result[f"observed_margin_effect_{name}"] = source_mean_summary(
                [row for row in rows if row["label"] == label],
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

    @staticmethod
    def _labels(dataset, sample):
        path = dataset.paths(sample).labels
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as archive:
            labels = np.asarray(archive["labels"], dtype=np.int8)
        if len(labels) != sample.response_tokens or not np.isin(labels, (-1, 0, 1)).all():
            raise ValueError(f"{sample.key}: labels must cover response tokens with -1/0/1")
        return labels

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
