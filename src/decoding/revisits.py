"""Analyze all saved decisions, then export their events and reading paths."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from decoding.io import empty_directory, read_jsonl, write_json
from decoding.reading_graph import ReadingGraph, RevisitSignal
from decoding.route_readout import RouteReadout


def write_rows(path, columns, rows):
    with path.open("x", encoding="utf8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns.split())
        for row in rows:
            writer.writerow(
                [
                    "" if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v
                    for v in row
                ]
            )


class RevisitAnalysis:
    def __init__(
        self,
        samples: Path,
        output: Path,
        window: int,
        quantile: float,
        context: int,
        hops: int = 3,
        states: Path | None = None,
    ):
        self.samples, self.output, self.states = samples, output, states
        self.window, self.quantile, self.context, self.hops = window, quantile, context, hops

    def run(self) -> None:
        if self.window < 2 or not 0 < self.quantile < 1 or self.context < 0 or self.hops < 1:
            raise ValueError("window >= 2, 0 < quantile < 1, context >= 0, hops >= 1 required")
        samples = list(read_jsonl(self.samples / "samples.jsonl"))
        if not samples or len({s["trace"] for s in samples}) != len(samples):
            raise ValueError("samples.jsonl must contain unique completed traces")
        empty_directory(self.output)
        shutil.copyfile(self.samples / "samples.jsonl", self.output / "samples.jsonl")
        write_json(
            self.output / "analysis.json",
            dict(
                method="content_revisit_paths",
                samples=str(self.samples.resolve()),
                states=str(self.states.resolve()) if self.states else None,
                window=self.window,
                quantile=self.quantile,
                context=self.context,
                hops=self.hops,
            ),
        )
        progress = tqdm(samples, desc="revisits", unit="answer")
        checks = []
        for sample in progress:
            progress.set_postfix_str(sample["trace"])
            checks.append(self._analyze(sample["trace"]))
        write_rows(
            self.output / "state_checks.csv", "trace max_logit_error max_attention_error", checks
        )

    def _load(self, name):
        if Path(name).name != name or not name.endswith(".npz"):
            raise ValueError("trace must be an NPZ filename inside samples")
        with np.load(self.samples / name, allow_pickle=False) as saved:
            trace = {key: saved[key] for key in saved.files}
        if self.states:
            from decoding.fixed_prefix import trace_signature

            with np.load(self.states / name, allow_pickle=False) as state:
                if not np.array_equal(state["token_ids"], trace["token_ids"]) or str(
                    state["signature"]
                ) != trace_signature(trace):
                    raise ValueError(f"{name}: states belong to different tokens/logits")
                fields = (
                    "hidden",
                    "source_mask",
                    "logit_entropy",
                    "max_logit_error",
                    "max_attention_error",
                )
                trace.update({key: state[key] for key in fields})
        self._validate(trace)
        return trace

    @staticmethod
    def _validate(trace):
        weights, ids = trace["attention"], trace["token_ids"]
        if weights.ndim != 4:
            raise ValueError("attention must have layer/head/step/key axes")
        layers, heads, steps, length = weights.shape
        prompt = int(trace["prompt_length"])
        if min(layers, heads, steps, prompt) < 1 or length != prompt + steps or len(ids) != length:
            raise ValueError("inconsistent generation positions")
        if "source_mask" not in trace:
            raise ValueError("missing source_mask: run main.py states and pass --states DIR")
        expected = dict(
            token_text=(length,),
            special_mask=(length,),
            source_mask=(prompt,),
            log_normalizer=(steps,),
        )
        if any(trace[k].shape != shape for k, shape in expected.items()):
            raise ValueError("token metadata does not match attention")
        if trace["top_logits"].ndim != 2 or trace["top_logits"].shape[0] != steps:
            raise ValueError("logits do not match attention")
        if trace["top_logits"].shape[1] < 2 or not trace["source_mask"].any():
            raise ValueError("two candidates and ordinary source tokens are required")
        if "hidden" in trace and trace["hidden"].shape[:2] != (layers + 1, length - 1):
            raise ValueError("hidden states do not match model layers and input positions")
        if "hidden" in trace and (
            trace["hidden"].ndim != 3 or not np.isfinite(trace["hidden"]).all()
        ):
            raise ValueError("hidden states must be a finite layer/position/dimension array")
        if "logit_entropy" in trace and trace["logit_entropy"].shape != (steps,):
            raise ValueError("logit entropy does not match generation positions")

    def _analyze(self, name):
        trace = self._load(name)
        signal = RevisitSignal(
            trace["attention"],
            trace["source_mask"],
            trace["special_mask"],
            self.window,
            self.quantile,
        ).run()
        graph = ReadingGraph(trace["attention"], trace["source_mask"], self.hops).run()
        readout = RouteReadout(graph, trace.get("hidden"), int(trace["prompt_length"])).run()
        output = self.output / Path(name).stem
        empty_directory(output)
        self._write_tokens(output, trace, signal)
        self._write_layers(output, signal)
        self._write_paths(output, trace, graph, readout)
        self._write_edges(output, trace, signal)
        np.savez_compressed(
            output / "graph.npz",
            **graph,
            **readout,
            token_ids=trace["token_ids"],
            token_text=trace["token_text"],
            prompt_length=trace["prompt_length"],
        )
        return [
            name,
            float(trace.get("max_logit_error", np.nan)),
            float(trace.get("max_attention_error", np.nan)),
        ]

    @staticmethod
    def _write_tokens(output, trace, signal):
        prompt = int(trace["prompt_length"])
        steps = len(signal["event"])
        text, logits = trace["token_text"], trace["top_logits"]
        probabilities = np.exp(logits[:, 0] - trace["log_normalizer"])
        entropies = trace.get("logit_entropy", np.full(steps, np.nan))
        rows = (
            [
                t,
                t - 1,
                prompt + t - 1,
                text[prompt + t - 1],
                text[prompt + t],
                signal["shift"][t].mean(),
                signal["revisit"][t],
                signal["threshold"][t],
                int(signal["active"][t]),
                int(signal["event"][t]),
                probabilities[t],
                logits[t, 0] - logits[t, 1],
                entropies[t],
            ]
            for t in range(steps)
        )
        write_rows(
            output / "tokens.csv",
            "step query_step query_position query token shift revisit threshold active event "
            "top1_probability top2_margin logit_entropy",
            rows,
        )

    @staticmethod
    def _write_layers(output, signal):
        steps, layers = signal["shift"].shape
        fields = ("shift", "layer_revisit", "dispersion", "disagreement", "effective_rank")
        rows = (
            [t, layer, *(signal[k][t, layer] for k in fields)]
            for t in range(steps)
            for layer in range(layers)
        )
        write_rows(
            output / "layers.csv",
            "step layer shift revisit dispersion disagreement effective_rank",
            rows,
        )

    @staticmethod
    def _write_paths(output, trace, graph, readout):
        fields = ("route_divergence", "relation_residual", "relation_shuffled", "relation_coverage")
        paths, positions = graph["paths"], graph["source_positions"]
        rows = []
        for hop, layer, t in np.ndindex(paths.shape[:3]):
            if hop > layer:
                continue
            weights = paths[hop, layer, t]
            keys = np.argsort(-weights, kind="stable")[:3]
            sources = [
                [int(positions[k]), str(trace["token_text"][positions[k]]), float(weights[k])]
                for k in keys
                if weights[k] > 0
            ]
            rows.append(
                [
                    t,
                    layer,
                    hop + 1,
                    weights.sum(),
                    *(readout[k][hop, layer, t] for k in fields),
                    json.dumps(sources, ensure_ascii=False),
                ]
            )
        write_rows(
            output / "paths.csv", "step layer hops path_mass " + " ".join(fields) + " sources", rows
        )

    def _write_edges(self, output, trace, signal):
        events = np.flatnonzero(signal["event"])
        weights, text = trace["attention"], trace["token_text"]
        prompt, steps = int(trace["prompt_length"]), weights.shape[2]
        selected = sorted(
            {
                t
                for e in events
                for t in range(max(0, e - self.context), min(steps, e + self.context + 1))
            }
        )
        columns = "step query_position layer head key_position key_token distance attention gain"
        rows = (row for t in selected for row in self._edges_at(t, trace, weights, text, prompt))
        write_rows(output / "edges.csv", columns, rows)

    @staticmethod
    def _edges_at(step, trace, weights, text, prompt):
        query = prompt + step - 1
        allowed = ~trace["special_mask"][:query].copy()
        allowed[: min(query, prompt)] &= trace["source_mask"][: min(query, prompt)]
        for layer, head in np.ndindex(weights.shape[:2]):
            row = weights[layer, head, step, :query].astype(np.float32) * allowed
            old = weights[layer, head, step - 1, :query] if step else np.zeros_like(row)
            keys = set(np.argsort(-row, kind="stable")[:2].tolist())
            if query > prompt:
                gain = (row - old)[prompt:]
                if gain.max() > 0:
                    keys.add(prompt + int(gain.argmax()))
            for key in sorted(keys, key=lambda k: (-row[k], k)):
                if row[key] > 0:
                    yield [
                        step,
                        query,
                        layer,
                        head,
                        key,
                        text[key],
                        query - key,
                        float(row[key]),
                        float(row[key] - old[key]) if step else np.nan,
                    ]
