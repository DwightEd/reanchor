"""Run analytic causal tracing only for statistically frozen anchors."""

from __future__ import annotations

import json
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from reanchor.artifacts import ArtifactStore
from reanchor.capture.protocol import AuditDataset

from .anchors import anchor_coordinates
from .cache import NativeCache
from .checkpoint import CheckpointWeights
from .cuts import CutRecorder, prepare_local_readout
from .propagation import trace_events


@dataclass(frozen=True)
class TraceConfig:
    device: str = "cuda:0"
    query_chunk: int = 8
    event_batch: int = 2
    save_edges: bool = True
    contrast_file: str | None = None

    def __post_init__(self):
        if self.query_chunk < 1 or self.event_batch < 1:
            raise ValueError("query_chunk and event_batch must be positive")


@contextmanager
def open_cut_resources(
    cache,
    folder: Path,
    event_rows: np.ndarray,
    config: TraceConfig,
    contrasts=None,
):
    """Open one shared suffix readout and one streaming edge writer per anchor."""

    if not config.save_edges:
        yield None, None
        return
    readout_path = folder / "local_readout.npz"
    prepare_local_readout(
        cache,
        readout_path,
        query_chunk=config.query_chunk,
        contrasts=contrasts,
    )
    with np.load(readout_path, allow_pickle=False) as readout, ExitStack() as stack:
        recorders = {}
        for row in event_rows:
            position = int(cache.trace["row_position"][row])
            recorder = CutRecorder(folder / f"edges/event_{position}.npz", cache, int(row))
            stack.callback(recorder.close)
            recorders[int(row)] = recorder
        yield readout, recorders


class CausalTracer:
    """Trace native remote messages from discovered anchors to future margins."""

    SCHEMA = "reanchor/analytic-trace@1"

    def __init__(self, config: TraceConfig = TraceConfig(), *, progress=None):
        self.config = config
        self.progress = progress

    def run(self, dataset: AuditDataset, run_root: str | Path) -> dict:
        run_root = Path(run_root).resolve()
        discovery = json.loads((run_root / "index.json").read_text(encoding="utf-8"))
        if discovery.get("method_schema") != "reanchor/max-null-episode@1":
            raise ValueError("tracing requires calibrated reanchor discovery artifacts")
        discovery_settings = discovery["settings"]
        active_gain_floor = min(
            float(discovery_settings["site_gain_floor"]),
            float(discovery_settings["broad_gain_floor"]),
        )
        local_floor = float(discovery_settings["local_floor"])
        window = int(discovery_settings["window"])
        completed, coverage = dataset.completed_samples(require_states=True)
        sample_by_key = {sample.key: sample for sample in completed}
        store = ArtifactStore(run_root)
        weights = None
        sample_summaries = []
        traced_count = 0
        resumed_count = 0
        computed_count = 0
        contrasts, contrast_digest = self._contrasts()
        identity = {**asdict(self.config), "contrast_sha256": contrast_digest}
        settings = json.dumps(identity, sort_keys=True)
        for entry in discovery["sample_artifacts"]:
            sample = sample_by_key.get(entry["key"])
            if sample is None:
                if int(entry.get("anchors", 0)):
                    raise ValueError(
                        f"{entry['key']}: anchor capture has no complete state archive"
                    )
                continue
            transitions = store.read_npz(self._inside(run_root, entry["transitions"]))
            events = store.read_npz(self._inside(run_root, entry["events"]))
            coordinates = anchor_coordinates(
                transitions["remote_gain"],
                transitions["previous_local_mass"],
                events["anchor"],
                active_gain_floor=active_gain_floor,
                local_floor=local_floor,
            )
            if not len(coordinates):
                sample_summaries.append({"key": sample.key, "anchors_traced": 0})
                continue
            folder = store.sample_path(sample, "traces").parent
            event_rows = np.unique(coordinates[:, 2])
            committed = {}
            pending_rows = []
            for row in event_rows:
                position = int(events["row_position"][row])
                trace_path = folder / f"traces/event_{position}.npz"
                edge_path = folder / f"edges/event_{position}.npz"
                if trace_path.is_file() and (not self.config.save_edges or edge_path.is_file()):
                    trace = store.read_npz(trace_path)
                    if (
                        str(trace.get("trace_schema", "")) != self.SCHEMA
                        or str(trace.get("trace_settings", "")) != settings
                        or str(trace.get("sample_key", "")) != sample.key
                        or int(trace.get("event_row", -1)) != row
                        or bool(trace.get("labels_used", True))
                    ):
                        raise ValueError(
                            f"{sample.key}: existing trace artifact has different identity"
                        )
                    committed[int(row)] = trace_path
                else:
                    pending_rows.append(int(row))
            pending_rows = np.asarray(pending_rows, dtype=int)
            if len(pending_rows) and weights is None:
                weights = CheckpointWeights(dataset.model_path, self.config.device)
            pending = coordinates[np.isin(coordinates[:, 2], pending_rows)]
            results = []
            if len(pending):
                sample_contrasts = contrasts.get(sample.key)
                with NativeCache(dataset.paths(sample), weights) as cache:
                    with open_cut_resources(
                        cache,
                        folder,
                        pending_rows,
                        self.config,
                        sample_contrasts,
                    ) as (
                        readout,
                        recorders,
                    ):
                        results = trace_events(
                            cache,
                            pending,
                            window=window,
                            query_chunk=self.config.query_chunk,
                            cut_readout=readout,
                            cut_recorders=recorders,
                            contrasts=sample_contrasts,
                            event_batch=self.config.event_batch,
                            progress=self.progress,
                        )
            returned_rows = np.array([int(result["event_row"]) for result in results])
            if not np.array_equal(returned_rows, pending_rows):
                raise ValueError(f"{sample.key}: tracing returned different anchor rows")
            for result in results:
                position = int(result["event_position"])
                path = folder / f"traces/event_{position}.npz"
                store.write_npz(
                    path,
                    **result,
                    trace_schema=np.array(self.SCHEMA),
                    trace_settings=np.array(settings),
                    sample_key=np.array(sample.key),
                )
                committed[int(result["event_row"])] = path
            paths = [str(committed[int(row)].relative_to(run_root)) for row in event_rows]
            traced_count += len(event_rows)
            resumed_count += len(event_rows) - len(results)
            computed_count += len(results)
            sample_summaries.append(
                {"key": sample.key, "anchors_traced": len(event_rows), "traces": paths}
            )
        summary = {
            "trace_schema": self.SCHEMA,
            "discovery_schema": discovery["method_schema"],
            "coverage": coverage,
            "anchors_traced": traced_count,
            "anchors_computed": computed_count,
            "anchors_resumed": resumed_count,
            "labels_used_for_tracing": False,
            "settings": identity,
            "sample_artifacts": sample_summaries,
        }
        store.write_json(run_root / "tracing.json", summary)
        return summary

    def _contrasts(self) -> tuple[dict[str, list[dict]], str | None]:
        if self.config.contrast_file is None:
            return {}, None
        path = Path(self.config.contrast_file)
        payload = path.read_bytes()
        values = json.loads(payload)
        if not isinstance(values, dict):
            raise ValueError("contrast file must map sample keys to candidate lists")
        required = {"target", "positive_id", "negative_id"}
        for sample_key, entries in values.items():
            if not isinstance(sample_key, str) or not isinstance(entries, list):
                raise ValueError("contrast file must map sample keys to candidate lists")
            if any(
                not isinstance(entry, dict) or not required <= entry.keys() for entry in entries
            ):
                raise ValueError("each contrast needs target, positive_id and negative_id")
        return values, sha256(payload).hexdigest()

    @staticmethod
    def _inside(root: Path, relative: str) -> Path:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"run artifact is outside run root: {relative}") from error
        return path
