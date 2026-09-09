"""Run analytic causal tracing only for statistically frozen anchors."""

from __future__ import annotations

import json
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
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

    def __post_init__(self):
        if self.query_chunk < 1 or self.event_batch < 1:
            raise ValueError("query_chunk and event_batch must be positive")


@contextmanager
def open_cut_resources(cache, folder: Path, event_rows: np.ndarray, config: TraceConfig):
    """Open one shared suffix readout and one streaming edge writer per anchor."""

    if not config.save_edges:
        yield None, None
        return
    readout_path = folder / "local_readout.npz"
    prepare_local_readout(cache, readout_path, query_chunk=config.query_chunk)
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
            if weights is None:
                weights = CheckpointWeights(dataset.model_path, self.config.device)
            folder = store.sample_path(sample, "traces").parent
            event_rows = np.unique(coordinates[:, 2])
            with NativeCache(dataset.paths(sample), weights) as cache:
                with open_cut_resources(cache, folder, event_rows, self.config) as (
                    readout,
                    recorders,
                ):
                    results = trace_events(
                        cache,
                        coordinates,
                        window=window,
                        query_chunk=self.config.query_chunk,
                        cut_readout=readout,
                        cut_recorders=recorders,
                        event_batch=self.config.event_batch,
                        progress=self.progress,
                    )
            returned_rows = np.array([int(result["event_row"]) for result in results])
            if not np.array_equal(returned_rows, event_rows):
                raise ValueError(f"{sample.key}: tracing returned different anchor rows")
            paths = []
            settings = json.dumps(asdict(self.config), sort_keys=True)
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
                paths.append(str(path.relative_to(run_root)))
            traced_count += len(results)
            sample_summaries.append(
                {"key": sample.key, "anchors_traced": len(results), "traces": paths}
            )
        summary = {
            "trace_schema": self.SCHEMA,
            "discovery_schema": discovery["method_schema"],
            "coverage": coverage,
            "anchors_traced": traced_count,
            "labels_used_for_tracing": False,
            "settings": asdict(self.config),
            "sample_artifacts": sample_summaries,
        }
        store.write_json(run_root / "tracing.json", summary)
        return summary

    @staticmethod
    def _inside(root: Path, relative: str) -> Path:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"run artifact is outside run root: {relative}") from error
        return path
