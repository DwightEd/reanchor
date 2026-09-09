"""Mechanism audit at statistically frozen reanchor anchors."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from reanchor.artifacts import ArtifactStore
from reanchor.capture.protocol import AuditDataset

from .anchors import anchor_coordinates
from .cache import NativeCache, read_trace
from .checkpoint import CheckpointWeights
from .contrasts import read_contrasts
from .propagation import trace_events
from .source_groups import source_partition, source_unit_partition


@dataclass(frozen=True)
class MechanismConfig:
    device: str = "cuda:0"
    query_chunk: int = 8
    event_batch: int = 2
    contrast_file: str | None = None
    closure_atol: float = 1e-5
    closure_rtol: float = 1e-3

    def __post_init__(self) -> None:
        if self.query_chunk < 1 or self.event_batch < 1:
            raise ValueError("query_chunk and event_batch must be positive")
        if self.closure_atol < 0 or self.closure_rtol < 0:
            raise ValueError("closure tolerances must be nonnegative")


class MechanismAuditor:
    """Trace the selected attention innovation without using outcome labels.

    The primary seed is the same adjacent-row, ordinary-normalized remote
    attention delta used by discovery. Four coarse provenance groups provide a
    complete additive audit. Material source units are retained separately so
    heterogeneous documents cannot silently cancel inside one broad bucket.
    """

    SCHEMA = "reanchor/transition-mechanism-audit@2"

    def __init__(self, config: MechanismConfig = MechanismConfig(), *, progress=None):
        self.config = config
        self.progress = progress

    def run(self, dataset: AuditDataset, run_root: str | Path) -> dict:
        run_root = Path(run_root).resolve()
        index_path = run_root / "index.json"
        tracing_path = run_root / "tracing.json"
        discovery = json.loads(index_path.read_text(encoding="utf-8"))
        if discovery.get("method_schema") != "reanchor/max-null-episode@1":
            raise ValueError("mechanism audit requires calibrated reanchor discovery artifacts")
        if not tracing_path.is_file():
            raise ValueError("mechanism audit requires completed causal tracing")
        tracing = json.loads(tracing_path.read_text(encoding="utf-8"))
        if tracing.get("trace_schema") != "reanchor/analytic-trace@2":
            raise ValueError("mechanism audit requires current-remote trace schema v2")
        contrasts, contrast_digest = read_contrasts(self.config.contrast_file)
        if not contrasts:
            raise ValueError("mechanism audit requires explicit correct/error --contrasts")
        if tracing.get("settings", {}).get("contrast_sha256") != contrast_digest:
            raise ValueError("mechanism audit contrasts differ from the completed trace")

        discovery_keys = {entry["key"] for entry in discovery["sample_artifacts"]}
        unknown = sorted(set(contrasts) - discovery_keys)
        if unknown:
            raise ValueError(f"contrast file contains unknown sample keys: {unknown[:5]}")

        completed, coverage = dataset.completed_samples(require_states=True)
        samples = {sample.key: sample for sample in completed}
        trace_entries = {entry["key"]: entry for entry in tracing["sample_artifacts"]}
        store = ArtifactStore(run_root)
        config_identity = asdict(self.config)
        config_identity.pop("contrast_file")
        settings_value = {
            **config_identity,
            "contrast_sha256": contrast_digest,
            "seed_estimand": "discovery_aligned_remote_attention_delta",
        }
        settings = json.dumps(settings_value, sort_keys=True)
        manifest_identity = {
            "discovery_manifest_sha256": self._file_digest(index_path),
            "trace_manifest_sha256": self._file_digest(tracing_path),
        }
        discovery_settings = discovery["settings"]
        active_gain_floor = min(
            float(discovery_settings["site_gain_floor"]),
            float(discovery_settings["broad_gain_floor"]),
        )
        local_floor = float(discovery_settings["local_floor"])
        window = int(discovery_settings["window"])
        weights = None
        sample_summaries = []
        audited = computed = resumed = 0

        for entry in discovery["sample_artifacts"]:
            key = entry["key"]
            sample = samples.get(key)
            trace_entry = trace_entries.get(key)
            if sample is None or trace_entry is None:
                if int(entry.get("anchors", 0)):
                    raise ValueError(f"{key}: frozen anchors have no complete trace capture")
                continue
            if key not in contrasts:
                sample_summaries.append(
                    {"key": key, "anchors_audited": 0, "reason": "no_explicit_contrast"}
                )
                continue
            transition_path = self._inside(run_root, entry["transitions"])
            event_path = self._inside(run_root, entry["events"])
            transitions = store.read_npz(transition_path)
            events = store.read_npz(event_path)
            coordinates = anchor_coordinates(
                transitions["remote_gain"],
                transitions["previous_local_mass"],
                events["anchor"],
                active_gain_floor=active_gain_floor,
                local_floor=local_floor,
            )
            event_rows = np.unique(coordinates[:, 2]) if len(coordinates) else np.array([], int)
            full_traces = self._full_traces(store, run_root, trace_entry)
            if set(map(int, event_rows)) != set(full_traces):
                raise ValueError(f"{key}: frozen anchors and completed traces disagree")

            capture_paths = dataset.paths(sample)
            metadata = read_trace(capture_paths.compact)
            sample_identity = {
                **manifest_identity,
                "transition_sha256": self._file_digest(transition_path),
                "event_sha256": self._file_digest(event_path),
                "source_annotations_sha256": self._metadata_digest(metadata),
                "qk_sha256": self._file_digest(capture_paths.qk),
                "history_sha256": self._file_digest(capture_paths.history),
                "states_sha256": self._file_digest(capture_paths.states),
            }
            folder = store.sample_path(sample, "mechanisms").parent
            committed = {}
            pending_rows = []
            row_identities = {}
            for row in event_rows:
                _, full_digest = full_traces[int(row)]
                row_identity = sha256(
                    json.dumps(
                        {
                            **sample_identity,
                            "settings": settings_value,
                            "sample_key": key,
                            "event_row": int(row),
                            "full_trace_sha256": full_digest,
                        },
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()
                row_identities[int(row)] = row_identity
                position = int(events["row_position"][row])
                path = folder / f"mechanisms/event_{position}.npz"
                if path.is_file():
                    artifact = store.read_npz(path)
                    if (
                        str(artifact.get("mechanism_schema", "")) != self.SCHEMA
                        or str(artifact.get("mechanism_settings", "")) != settings
                        or str(artifact.get("mechanism_identity", "")) != row_identity
                        or str(artifact.get("sample_key", "")) != key
                        or int(artifact.get("event_row", -1)) != row
                        or bool(artifact.get("outcome_token_labels_used", True))
                    ):
                        raise ValueError(
                            f"{key}: existing mechanism artifact has different identity"
                        )
                    committed[int(row)] = path
                else:
                    pending_rows.append(int(row))

            if pending_rows and weights is None:
                weights = CheckpointWeights(dataset.model_path, self.config.device)
            pending_rows_array = np.asarray(pending_rows, dtype=int)
            pending = coordinates[np.isin(coordinates[:, 2], pending_rows_array)]
            if len(pending):
                with NativeCache(capture_paths, weights) as cache:
                    coarse = source_partition(cache.trace)
                    units = source_unit_partition(cache.trace, coarse)
                    ordinary = ~np.asarray(cache.trace["special_mask"], dtype=bool)
                    masks = np.concatenate((ordinary[None], coarse.masks, units.masks), axis=0)
                    partition_results = self._trace_partitions(
                        cache, pending, window, contrasts[key], masks
                    )
                    transition = partition_results[0]
                    grouped = partition_results[1 : 1 + len(coarse.names)]
                    unit_results = partition_results[1 + len(coarse.names) :]
                    for index, row in enumerate(pending_rows_array):
                        current, _ = full_traces[int(row)]
                        artifact = self._combine(
                            current,
                            transition[index],
                            [values[index] for values in grouped],
                            [values[index] for values in unit_results],
                            coarse,
                            units,
                            pending[pending[:, 2] == row],
                        )
                        position = int(artifact["event_position"])
                        path = folder / f"mechanisms/event_{position}.npz"
                        store.write_npz(
                            path,
                            **artifact,
                            mechanism_schema=np.array(self.SCHEMA),
                            mechanism_settings=np.array(settings),
                            mechanism_identity=np.array(row_identities[int(row)]),
                            sample_key=np.array(key),
                        )
                        committed[int(row)] = path
                        computed += 1

            paths = [str(committed[int(row)].relative_to(run_root)) for row in event_rows]
            audited += len(event_rows)
            resumed += len(event_rows) - len(pending_rows)
            sample_summaries.append(
                {"key": key, "anchors_audited": len(event_rows), "mechanisms": paths}
            )

        summary = {
            "mechanism_schema": self.SCHEMA,
            "discovery_schema": discovery["method_schema"],
            "trace_schema": tracing["trace_schema"],
            "coverage": coverage,
            "anchors_audited": audited,
            "anchors_computed": computed,
            "anchors_resumed": resumed,
            "outcome_token_labels_used_for_mechanism_audit": False,
            "explicit_correctness_contrasts_used": True,
            "primary_estimand": (
                "discovery-aligned adjacent-row remote attention delta propagated "
                "to the correct-minus-error margin"
            ),
            "reference_estimand": "current native remote write at the same frozen sites",
            "coarse_source_groups_are_mechanism_classes": False,
            "source_unit_effects_retained": True,
            "exact_intervention_run": False,
            "binding_identified": False,
            "binding_requirement": (
                "matched constraint counterfactual or bidirectional exact intervention"
            ),
            "settings": settings_value,
            "sample_artifacts": sample_summaries,
        }
        store.write_json(run_root / "mechanism.json", summary)
        return summary

    def _trace_partitions(self, cache, pending, window, contrasts, masks):
        if self.progress:
            self.progress(f"mechanism audit: shared propagation for {len(masks)} partitions")
        return trace_events(
            cache,
            pending,
            window=window,
            query_chunk=self.config.query_chunk,
            contrasts=contrasts,
            event_batch=self.config.event_batch,
            progress=self.progress,
            source_masks=masks,
            seed_kind="transition_delta_remote",
        )

    def _combine(
        self,
        current,
        transition,
        components,
        unit_components,
        coarse,
        units,
        expected_sites,
    ):
        expected_sites = np.asarray(expected_sites, dtype=int).reshape(-1, 3)
        if not np.array_equal(current.get("event_sites"), expected_sites):
            raise ValueError("completed current trace uses stale anchor sites")
        if str(current.get("seed_kind", "")) != "current_remote":
            raise ValueError("completed trace does not identify the current-remote seed")
        if str(transition.get("seed_kind", "")) != "transition_delta_remote":
            raise ValueError("mechanism trace does not identify the transition seed")
        for field in ("event_sites", "variants", "hop_names"):
            if not np.array_equal(transition.get(field), current.get(field)):
                raise ValueError(f"transition changed completed trace field {field}")
        identity_fields = (
            "event_row",
            "event_position",
            "target_position",
            "explicit_contrast",
            "positive_id",
            "negative_id",
            "baseline_margin",
        )
        for value in (transition, *components, *unit_components):
            for field in identity_fields:
                if not np.array_equal(value[field], current[field]):
                    raise ValueError(f"source component changed trace field {field}")
            for field in ("event_sites", "variants", "hop_names", "seed_kind"):
                if not np.array_equal(value.get(field), transition.get(field)):
                    raise ValueError(f"source component changed transition field {field}")
        margins = np.stack([value["margin_response"] for value in components])
        layers = np.stack([value["layer_margin_response"] for value in components])
        roots = np.stack([value["root_attention_sum"] for value in components])
        closure = {
            "margin": self._closure(margins.sum(0), transition["margin_response"]),
            "layer": self._closure(layers.sum(0), transition["layer_margin_response"]),
            "root": self._closure(roots.sum(0), transition["root_attention_sum"]),
        }
        unit_margins = self._stack_or_empty(
            unit_components, "margin_response", transition["margin_response"]
        )
        unit_layers = self._stack_or_empty(
            unit_components, "layer_margin_response", transition["layer_margin_response"]
        )
        unit_roots = self._stack_or_empty(
            unit_components, "root_attention_sum", transition["root_attention_sum"]
        )
        return {
            "event_row": current["event_row"],
            "event_position": current["event_position"],
            "target_position": current["target_position"],
            "seed_estimand": np.array("discovery_aligned_remote_attention_delta"),
            "source_group_names": np.asarray(coarse.names),
            "source_group_margin_response": margins,
            "source_group_layer_margin_response": layers,
            "source_group_root_coefficient": roots,
            "source_group_seed_norm": np.asarray(
                [value["seed_norm"] for value in components], dtype=np.float32
            ),
            "source_unit_ids": np.asarray(units.ids, dtype=np.int64),
            "source_unit_roles": np.asarray(units.roles),
            "source_unit_margin_response": unit_margins,
            "source_unit_layer_margin_response": unit_layers,
            "source_unit_root_coefficient": unit_roots,
            "source_unit_seed_norm": np.asarray(
                [value["seed_norm"] for value in unit_components], dtype=np.float32
            ),
            "transition_margin_response": transition["margin_response"],
            "transition_layer_margin_response": transition["layer_margin_response"],
            "transition_root_coefficient": transition["root_attention_sum"],
            "transition_seed_norm": transition["seed_norm"],
            "current_remote_margin_response": current["margin_response"],
            "current_remote_layer_margin_response": current["layer_margin_response"],
            "current_remote_root_attention": current["root_attention_sum"],
            "baseline_margin": current["baseline_margin"],
            "positive_id": current["positive_id"],
            "negative_id": current["negative_id"],
            "explicit_contrast": current["explicit_contrast"],
            "semantic_source_roles_available": np.array(coarse.semantic_roles_available),
            "coarse_groups_are_mechanism_classes": np.array(False),
            "closure_margin_max_abs": np.array(closure["margin"][0]),
            "closure_margin_threshold": np.array(closure["margin"][1]),
            "closure_margin_pass": np.array(closure["margin"][2]),
            "closure_layer_max_abs": np.array(closure["layer"][0]),
            "closure_layer_threshold": np.array(closure["layer"][1]),
            "closure_layer_pass": np.array(closure["layer"][2]),
            "closure_root_max_abs": np.array(closure["root"][0]),
            "closure_root_threshold": np.array(closure["root"][1]),
            "closure_root_pass": np.array(closure["root"][2]),
            "closure_pass": np.array(all(value[2] for value in closure.values())),
            "outcome_token_labels_used": np.array(False),
        }

    def _closure(self, observed, expected) -> tuple[float, float, bool]:
        residual = np.asarray(observed) - np.asarray(expected)
        pointwise_threshold = self.config.closure_atol + self.config.closure_rtol * np.abs(
            np.asarray(expected)
        )
        max_abs = self._max_abs(residual)
        max_threshold = self._max_abs(pointwise_threshold)
        return max_abs, max_threshold, bool(np.all(np.abs(residual) <= pointwise_threshold))

    @staticmethod
    def _stack_or_empty(components, field, template):
        if components:
            return np.stack([value[field] for value in components])
        return np.empty((0, *np.asarray(template).shape), dtype=np.asarray(template).dtype)

    @staticmethod
    def _full_traces(store, run_root: Path, entry: dict):
        result = {}
        for relative in entry.get("traces", []):
            path = MechanismAuditor._inside(run_root, relative)
            trace = store.read_npz(path)
            row = int(trace["event_row"])
            if row in result:
                raise ValueError(f"duplicate completed trace for event row {row}")
            result[row] = trace, MechanismAuditor._file_digest(path)
        return result

    @staticmethod
    def _file_digest(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _metadata_digest(trace: dict[str, np.ndarray]) -> str:
        digest = sha256()
        fields = (
            "token_ids",
            "token_text",
            "row_position",
            "response_start",
            "special_mask",
            "evidence_mask",
            "source_kind",
            "source_unit_id",
        )
        for name in fields:
            digest.update(name.encode("utf-8"))
            if name not in trace:
                digest.update(b"<missing>")
                continue
            value = np.ascontiguousarray(np.asarray(trace[name]))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(json.dumps(value.shape).encode("ascii"))
            digest.update(value.tobytes())
        return digest.hexdigest()

    @staticmethod
    def _max_abs(value) -> float:
        array = np.asarray(value, dtype=float)
        return float(np.max(np.abs(array))) if array.size else 0.0

    @staticmethod
    def _inside(root: Path, relative: str) -> Path:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"run artifact is outside run root: {relative}") from error
        return path
