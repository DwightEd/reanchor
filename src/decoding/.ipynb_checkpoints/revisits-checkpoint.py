"""Scan native reading graphs without entity lists or hallucination labels."""

import csv
import shutil
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from decoding.attention import entropy
from decoding.io import empty_directory, read_jsonl, write_json


class RevisitAnalysis:
    def __init__(self, samples: Path, output: Path, window: int, quantile: float, context: int):
        self.samples, self.output = samples, output
        self.window, self.quantile, self.context = window, quantile, context

    def run(self) -> None:
        if self.window < 2 or not 0 < self.quantile < 1 or self.context < 0:
            raise ValueError("window >= 2, 0 < quantile < 1 and context >= 0 are required")
        samples = list(read_jsonl(self.samples / "samples.jsonl"))
        if not samples or len({s["trace"] for s in samples}) != len(samples):
            raise ValueError("samples.jsonl must contain unique completed traces")
        for sample in samples:
            name = sample["trace"]
            if Path(name).name != name or not name.endswith(".npz"):
                raise ValueError("trace must be an NPZ filename inside samples")
        empty_directory(self.output)
        shutil.copyfile(self.samples / "samples.jsonl", self.output / "samples.jsonl")
        write_json(
            self.output / "analysis.json",
            {
                "samples": str(self.samples.resolve()),
                "window": self.window,
                "quantile": self.quantile,
                "context": self.context,
            },
        )
        progress = tqdm(samples, desc="revisits", unit="answer")
        for sample in progress:
            progress.set_postfix_str(sample["trace"])
            self._analyze(sample)

    def _analyze(self, sample):
        name = sample["trace"]
        with np.load(self.samples / name, allow_pickle=False) as trace:
            weights, ids, text = trace["attention"], trace["token_ids"], trace["token_text"]
            special = trace["special_mask"]
            prompt = int(trace["prompt_length"])
            logits, normalizer = trace["top_logits"], trace["log_normalizer"]
            logit_entropy = trace["logit_entropy"] if "logit_entropy" in trace else None
        if weights.ndim != 4:
            raise ValueError(f"{name}: attention must have layer/head/step/key axes")
        layers, heads, steps, length = weights.shape
        if min(layers, heads, steps, prompt) < 1 or length != prompt + steps or len(ids) != length:
            raise ValueError(f"{name}: inconsistent generation positions")
        if (
            text.shape != (length,)
            or special.shape != (length,)
            or logits.ndim != 2
            or logits.shape[0] != steps
            or logits.shape[1] < 2
            or normalizer.shape != (steps,)
            or (logit_entropy is not None and logit_entropy.shape != (steps,))
        ):
            raise ValueError(f"{name}: token metadata or logits do not match attention")
        output = self.output / Path(name).stem
        empty_directory(output)
        shifts = np.full((steps, layers, heads), np.nan)
        scores = np.full(steps, np.nan)
        events, previous, above_previous = [], None, False
        with (
            (output / "tokens.csv").open("x", encoding="utf-8", newline="") as tokens,
            (output / "layers.csv").open("x", encoding="utf-8", newline="") as layer_stream,
        ):
            token_writer, layer_writer = csv.writer(tokens), csv.writer(layer_stream)
            token_writer.writerow(
                "step query_position query token shift revisit threshold event "
                "top1_probability top2_margin logit_entropy".split()
            )
            layer_writer.writerow(
                "step layer shift revisit dispersion disagreement effective_rank".split()
            )
            for t in tqdm(range(steps), desc=name, unit="token", leave=False):
                # Row t is the query at P+t-1, before sampling response token t.
                query = prompt + t - 1
                current = weights[:, :, t, : query + 1].astype(np.float64)
                mass = current.sum(-1, keepdims=True)
                if (
                    not np.isfinite(current).all()
                    or (current < 0).any()
                    or not np.allclose(mass, 1, atol=0.005, rtol=0)
                    or np.count_nonzero(weights[:, :, t, query + 1 :])
                ):
                    raise ValueError(f"{name} step {t}: invalid attention or visible future keys")
                p = current / mass
                excess = np.full((layers, heads), np.nan)
                if previous is not None:
                    old = np.pad(previous, ((0, 0), (0, 0), (0, 1)))
                    shifts[t] = np.abs(np.cumsum(p - old, axis=-1)).sum(-1)
                    past = shifts[max(1, t - self.window) : t]
                    if len(past):
                        excess = np.maximum(0, shifts[t] - np.median(past, axis=0))
                        scores[t] = excess.mean()
                previous = p
                history = scores[max(0, t - self.window) : t]
                history = history[np.isfinite(history)]
                threshold = np.quantile(history, self.quantile) if len(history) else np.nan
                above = t > self.window and scores[t] > threshold + 1e-10
                event = bool(above and not above_previous)
                above_previous = above
                if event:
                    events.append(t)
                dispersion = entropy(p).mean(-1)
                disagreement = np.maximum(0, entropy(p.mean(1)) - dispersion)
                # Head--key bipartite graph: affinity of square-root probability rows.
                # Shared diffuse heads have rank 1; disjoint one-hot heads have rank H.
                root = np.sqrt(p)
                gram = (root @ root.swapaxes(-1, -2)) / heads
                spectrum = np.maximum(0, np.linalg.eigvalsh(gram))
                spectrum /= spectrum.sum(-1, keepdims=True)
                rank = np.exp2(entropy(spectrum))
                token_writer.writerow(
                    [
                        t,
                        query,
                        text[query],
                        text[prompt + t],
                        "" if t == 0 else float(shifts[t].mean()),
                        "" if not np.isfinite(scores[t]) else float(scores[t]),
                        "" if not np.isfinite(threshold) else float(threshold),
                        int(event),
                        float(np.exp(logits[t, 0] - normalizer[t])),
                        float(logits[t, 0] - logits[t, 1]),
                        "" if logit_entropy is None else float(logit_entropy[t]),
                    ]
                )
                for layer in range(layers):
                    layer_writer.writerow(
                        [
                            t,
                            layer,
                            "" if t == 0 else float(shifts[t, layer].mean()),
                            "" if t < 2 else float(excess[layer].mean()),
                            dispersion[layer],
                            disagreement[layer],
                            rank[layer],
                        ]
                    )
        self._write_reads(output, weights, prompt, text, special, events, shifts)

    def _write_reads(self, output, weights, prompt, text, special, events, shifts):
        layers, heads, steps, _ = weights.shape
        selected = sorted(
            {
                t
                for event in events
                for t in range(max(0, event - self.context), min(steps, event + self.context + 1))
            }
        )
        prompt_keys = np.flatnonzero(~special[:prompt])
        relay_cache = {}
        with (output / "reads.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                "step query_position layer head head_entropy head_shift "
                "key_position key_region key_token distance "
                "attention previous_attention gain context relay_position relay_token "
                "relay_attention path_weight".split()
            )
            for t in tqdm(selected, desc="event reads", unit="token", leave=False):
                query = prompt + t - 1
                for layer in range(layers):
                    for head in range(heads):
                        current = weights[layer, head, t, : query + 1].astype(np.float64)
                        head_entropy = float(entropy(current / current.sum()))
                        previous = weights[layer, head, t - 1] if t else None
                        keys = set(np.argsort(-current, kind="stable")[:2].tolist())
                        # Include a newly strengthened history read even when self/sink dominate.
                        if query > prompt:
                            history_gain = current[prompt:query] - previous[prompt:query]
                            k = prompt + int(history_gain.argmax())
                            if current[k] > 0 and history_gain[k - prompt] > 0:
                                keys.add(k)
                        for key in sorted(keys, key=lambda k: (-current[k], k)):
                            if current[key] <= 0:
                                continue
                            region = "prompt" if key < prompt else "history"
                            if key == query:
                                region = "self"
                            if special[key]:
                                region = "special"
                            old = float(previous[key]) if t and key < query else None
                            relay = [""] * 4
                            if region == "history" and layer > 0 and len(prompt_keys):
                                identity = layer, key
                                if identity not in relay_cache:
                                    # History token u was INPUT at prediction row u+1.
                                    # Follow a strictly earlier layer, averaging its heads.
                                    row = key - prompt + 1
                                    source = weights[layer - 1, :, row, :prompt].mean(
                                        axis=0, dtype=np.float64
                                    )
                                    anchor = int(prompt_keys[source[prompt_keys].argmax()])
                                    relay_cache[identity] = anchor, float(source[anchor])
                                anchor, weight = relay_cache[identity]
                                if weight > 0:
                                    relay = [anchor, text[anchor], weight, current[key] * weight]
                            writer.writerow(
                                [
                                    t,
                                    query,
                                    layer,
                                    head,
                                    head_entropy,
                                    "" if t == 0 else shifts[t, layer, head],
                                    key,
                                    region,
                                    text[key],
                                    query - key,
                                    current[key],
                                    "" if old is None else old,
                                    "" if old is None else current[key] - old,
                                    "".join(text[max(0, key - 4) : min(query + 1, key + 5)]),
                                    *relay,
                                ]
                            )
