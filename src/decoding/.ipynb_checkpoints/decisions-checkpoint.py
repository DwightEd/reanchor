"""Inspect saved native predictions and specific reads in fixed generation windows."""

import csv
import hashlib
import shutil
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from decoding.io import empty_directory, read_jsonl


class DecisionInspection:
    def __init__(self, samples: Path, cases: Path, output: Path):
        self.samples, self.cases, self.output = samples, cases, output

    def run(self) -> None:
        with self.cases.open(encoding="utf-8", newline="") as stream:
            cases = list(csv.DictReader(stream))
        if not cases or len({c["case"] for c in cases}) != len(cases):
            raise ValueError("cases must have unique case names")
        samples = list(read_jsonl(self.samples / "samples.jsonl"))
        for sample in samples:
            name = sample["trace"]
            if Path(name).name != name or not name.endswith(".npz"):
                raise ValueError("trace must be an NPZ filename inside samples")
        selected = {}
        for case in cases:
            matches = [
                s
                for s in samples
                if str(s["source_id"]) == case["source_id"] and s["seed"] == int(case["seed"])
            ]
            if len(matches) != 1:
                raise ValueError(f"{case['case']}: expected one source_id/seed match")
            name = matches[0]["trace"]
            selected.setdefault(name, []).append(case)

        empty_directory(self.output)
        self._check_prefixes(samples)
        with ExitStack() as stack:
            writers = {}
            columns = {
                "tokens": "case trace step query_position token_id token probability "
                "chosen_rank chosen_margin top2_margin logit_entropy",
                "candidates": "case trace step rank candidate_id candidate logit probability "
                "is_chosen",
                "reads": "case trace step query_position layer head key_region key_position "
                "key_step key_token distance previous_attention attention gain",
            }
            for name, header in columns.items():
                stream = stack.enter_context(
                    (self.output / f"{name}.csv").open("x", encoding="utf-8", newline="")
                )
                writers[name] = csv.writer(stream)
                writers[name].writerow(header.split())
            for name, windows in tqdm(selected.items(), desc="decisions", unit="answer"):
                with np.load(self.samples / name, allow_pickle=False) as trace:
                    prompt = int(trace["prompt_length"])
                    ids, text = trace["token_ids"], trace["token_text"]
                    top_ids, logits = trace["top_ids"], trace["top_logits"]
                    top_text = trace["top_text"]
                    chosen, normalizers = trace["chosen_logit"], trace["log_normalizer"]
                    entropies = trace["logit_entropy"] if "logit_entropy" in trace else None
                    weights = trace["attention"]
                steps = len(chosen)
                if prompt < 1 or len(ids) != prompt + steps or logits.shape[1] < 2:
                    raise ValueError(f"{name}: inconsistent generation positions or candidates")
                if weights.ndim != 4 or weights.shape[2:] != (steps, len(ids)):
                    raise ValueError(f"{name}: attention does not match generation positions")
                for case in windows:
                    start, stop = int(case["start"]), int(case["stop"])
                    if not 0 <= start < stop <= steps:
                        raise ValueError(f"{case['case']}: window is outside the response")
                    if "".join(text[prompt + start : prompt + stop]) != case["text"]:
                        raise ValueError(
                            f"{case['case']}: window text differs from captured tokens"
                        )
                    history = sorted(set(map(int, case["history_steps"].split())))
                    prompt_keys = sorted(set(map(int, case["prompt_positions"].split())))
                    if any(k < 0 or k >= prompt for k in prompt_keys):
                        raise ValueError(f"{case['case']}: prompt key is outside prompt")
                    if any(k < 0 or k >= steps for k in history):
                        raise ValueError(f"{case['case']}: history step is outside response")
                    keys = prompt_keys + [prompt + k for k in history]
                    for t in tqdm(range(start, stop), desc=case["case"], leave=False, unit="token"):
                        actual = ids[prompt + t]
                        ranks = np.flatnonzero(top_ids[t] == actual)
                        competitor = logits[t, 1 if actual == top_ids[t, 0] else 0]
                        prefix = [case["case"], name, t]
                        writers["tokens"].writerow(
                            prefix
                            + [
                                prompt + t - 1,
                                actual,
                                text[prompt + t],
                                float(np.exp(chosen[t] - normalizers[t])),
                                int(ranks[0]) + 1 if len(ranks) else "",
                                float(chosen[t] - competitor),
                                float(logits[t, 0] - logits[t, 1]),
                                "" if entropies is None else float(entropies[t]),
                            ]
                        )
                        for rank, candidate in enumerate(top_ids[t]):
                            writers["candidates"].writerow(
                                prefix
                                + [
                                    rank + 1,
                                    candidate,
                                    top_text[t, rank],
                                    float(logits[t, rank]),
                                    float(np.exp(logits[t, rank] - normalizers[t])),
                                    int(candidate == actual),
                                ]
                            )
                        self._write_reads(writers["reads"], prefix, prompt, text, weights, keys, t)
        for name in ("samples.jsonl", "prompts.jsonl", "settings.json"):
            shutil.copyfile(self.samples / name, self.output / name)
        shutil.copyfile(self.cases, self.output / "cases.csv")

    def _check_prefixes(self, samples):
        seen = {}
        with (self.output / "same_prefix.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                (
                    "trace_a trace_b step prefix_tokens next_id_a next_id_b candidate_ids_equal "
                    "max_top_logit_error log_normalizer_error"
                ).split()
            )
            for sample in tqdm(samples, desc="same prefixes", unit="answer"):
                name = sample["trace"]
                # No attention decompression is needed for this numerical audit.
                with np.load(self.samples / name, allow_pickle=False) as trace:
                    ids = trace["token_ids"].astype("<i8")
                    prompt = int(trace["prompt_length"])
                    candidates, logits = trace["top_ids"], trace["top_logits"]
                    normalizers = trace["log_normalizer"]
                if prompt < 1 or len(ids) != prompt + len(logits):
                    raise ValueError(f"{name}: inconsistent prefix lengths")
                digest = hashlib.sha256(ids[:prompt].tobytes())
                for t in range(len(logits)):
                    # Exclude the token being sampled: the branching decision shares a prefix.
                    key = (prompt, digest.digest())
                    previous = seen.get(key)
                    if previous is None:
                        seen[key] = (
                            name,
                            ids[prompt + t],
                            candidates[t],
                            logits[t],
                            normalizers[t],
                        )
                    else:
                        old_name, old_next, old_ids, old_logits, old_normalizer = previous
                        writer.writerow(
                            [
                                old_name,
                                name,
                                t,
                                prompt + t,
                                old_next,
                                ids[prompt + t],
                                int(np.array_equal(old_ids, candidates[t])),
                                float(np.max(np.abs(old_logits.astype(float) - logits[t]))),
                                abs(float(old_normalizer) - float(normalizers[t])),
                            ]
                        )
                    digest.update(ids[prompt + t : prompt + t + 1].tobytes())

    @staticmethod
    def _write_reads(writer, prefix, prompt, text, weights, keys, t):
        query = prompt + t - 1
        layers, heads = weights.shape[:2]
        for key in keys:
            if key > query:
                continue
            # A key at the current query was not visible in the preceding forward pass.
            comparable = t > 0 and key < query
            for layer in range(layers):
                for head in range(heads):
                    now = float(weights[layer, head, t, key])
                    old = float(weights[layer, head, t - 1, key]) if comparable else None
                    writer.writerow(
                        prefix
                        + [
                            query,
                            layer,
                            head,
                            "prompt" if key < prompt else "history",
                            key,
                            "" if key < prompt else key - prompt,
                            text[key],
                            query - key,
                            "" if old is None else old,
                            now,
                            "" if old is None else now - old,
                        ]
                    )
