"""WAAD and FAI from Li et al., arXiv:2510.13554v2; descriptive baselines."""

import csv
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from reanchor.io import read_jsonl


class AttentionAnalysis:
    def __init__(self, samples: Path, output: Path, window: int, future_range: tuple[int, int]):
        self.samples, self.output = samples, output
        self.window, self.future_range = window, future_range

    def run(self) -> int:
        low, high = self.future_range
        if self.window < 1 or not 1 <= low <= high:
            raise ValueError("attention windows must be positive and ordered")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with self.output.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "source_id",
                    "seed",
                    "token_index",
                    "query_position",
                    "layer",
                    "head",
                    "waad",
                    "fai",
                    "fai_queries",
                    "peak_source",
                ]
            )
            for sample in tqdm(list(read_jsonl(self.samples / "samples.jsonl")), desc="attention"):
                with np.load(self.samples / sample["trace"], allow_pickle=False) as trace:
                    attention = trace["attention"]
                    prompt = int(trace["prompt_length"])
                    layers, heads, steps, tokens = attention.shape
                    if prompt < 1 or tokens != prompt + steps:
                        raise ValueError("attention rows do not match generation positions")
                    queries = np.arange(prompt - 1, prompt + steps - 1)
                    for t, query in enumerate(queries):
                        row = attention[:, :, t].astype(np.float32)
                        distance = np.clip(query - np.arange(tokens), 0, self.window)
                        waad = (row * distance).sum(-1)
                        peak = row.argmax(-1)
                        # FAI belongs to the generated token P+t, not to its predictor P+t-1.
                        readers = (queries >= prompt + t + low) & (queries <= prompt + t + high)
                        n = int(readers.sum())
                        fai = (
                            attention[:, :, readers, prompt + t].astype(np.float32).mean(-1)
                            if n
                            else None
                        )
                        for layer in range(layers):
                            for head in range(heads):
                                writer.writerow(
                                    [
                                        sample["source_id"],
                                        sample["seed"],
                                        t,
                                        query,
                                        layer,
                                        head,
                                        float(waad[layer, head]),
                                        float(fai[layer, head]) if n else "",
                                        n,
                                        int(peak[layer, head]),
                                    ]
                                )
                                count += 1
        return count
