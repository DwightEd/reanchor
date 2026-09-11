"""Read concrete past-token attention changes at each decoding decision."""

import csv
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from decoding.io import empty_directory, read_jsonl


class AttentionAnalysis:
    def __init__(
        self,
        samples: Path,
        output: Path,
        trace: str | None,
        first_error: int | None,
        before: int,
        after: int,
        min_distance: int,
    ):
        self.samples, self.output, self.trace = samples, output, trace
        self.first_error, self.before, self.after = first_error, before, after
        self.min_distance = min_distance

    def run(self) -> None:
        if self.before < 0 or self.after < 0 or self.min_distance < 1:
            raise ValueError("windows must be nonnegative and min-distance positive")
        if self.first_error is not None and (self.trace is None or self.first_error < 0):
            raise ValueError("first-error requires one trace and a nonnegative token index")
        samples = list(read_jsonl(self.samples / "samples.jsonl"))
        if not samples:
            raise ValueError("samples.jsonl contains no completed samples")
        if self.trace is not None:
            samples = [sample for sample in samples if sample["trace"] == self.trace]
            if len(samples) != 1:
                raise ValueError("trace must identify exactly one sample")
        empty_directory(self.output)
        with (
            (self.output / "tokens.csv").open("x", newline="", encoding="utf-8") as tokens,
            (self.output / "attention.csv").open("x", newline="", encoding="utf-8") as attention,
        ):
            token_writer, attention_writer = csv.writer(tokens), csv.writer(attention)
            token_writer.writerow(
                (
                    "trace step error_offset token_id token_piece token probability "
                    "top1 top1_probability top2 top2_probability top2_margin chosen_margin"
                ).split()
            )
            attention_writer.writerow(
                (
                    "trace step error_offset query_position query layer head shift "
                    "previous_peak_position previous_peak read_position read_token read_context "
                    "distance previous_attention attention gain"
                ).split()
            )
            for sample in tqdm(samples, desc="inspect", unit="answer"):
                with np.load(self.samples / sample["trace"], allow_pickle=False) as trace:
                    weights = trace["attention"]
                    prompt = int(trace["prompt_length"])
                    ids, text = trace["token_ids"], trace["token_text"]
                    pieces, special = trace["token_pieces"], trace["special_mask"]
                    top_ids, top_logits = trace["top_ids"], trace["top_logits"]
                    top_text = trace["top_text"]
                    normalizer, chosen = trace["log_normalizer"], trace["chosen_logit"]
                layers, heads, steps, length = weights.shape
                if prompt < 1 or length != prompt + steps or len(ids) != length:
                    raise ValueError("trace does not match generation positions")
                first = self.first_error
                if first is not None and first >= steps:
                    raise ValueError("first-error is outside this response")
                start = 0 if first is None else max(0, first - self.before)
                stop = steps if first is None else min(steps, first + self.after + 1)
                for t in tqdm(range(start, stop), desc=sample["trace"], unit="token", leave=False):
                    offset = "" if first is None else t - first
                    prefix = [sample["trace"], t, offset]
                    competitor = top_logits[t, 1 if ids[prompt + t] == top_ids[t, 0] else 0]
                    token_writer.writerow(
                        prefix
                        + [
                            int(ids[prompt + t]),
                            pieces[prompt + t],
                            text[prompt + t],
                            float(np.exp(chosen[t] - normalizer[t])),
                            top_text[t, 0],
                            float(np.exp(top_logits[t, 0] - normalizer[t])),
                            top_text[t, 1],
                            float(np.exp(top_logits[t, 1] - normalizer[t])),
                            float(top_logits[t, 0] - top_logits[t, 1]),
                            float(chosen[t] - competitor),
                        ]
                    )
                    # Query q predicts token P+t. Compare exactly the same past keys:
                    # exclude newly available q and special tokens from BOTH distributions.
                    query = prompt + t - 1
                    past = np.flatnonzero((np.arange(length) < query) & ~special)
                    current = weights[:, :, t, past].astype(np.float64)
                    previous = weights[:, :, t - 1, past].astype(np.float64) if t else None
                    mass = current.sum(-1)
                    shift = np.full((layers, heads), np.nan)
                    if t and len(past):
                        old_mass = previous.sum(-1)
                        valid = (mass > 0) & (old_mass > 0)
                        p = np.divide(
                            current,
                            mass[..., None],
                            out=np.zeros_like(current),
                            where=mass[..., None] > 0,
                        )
                        old_p = np.divide(
                            previous,
                            old_mass[..., None],
                            out=np.zeros_like(previous),
                            where=old_mass[..., None] > 0,
                        )
                        shift[valid] = 0.5 * np.abs(p - old_p).sum(-1)[valid]
                    remote = np.flatnonzero(query - past >= self.min_distance)
                    for layer in range(layers):
                        for head in range(heads):
                            old_peak = None
                            if t and len(past) and previous[layer, head].sum() > 0:
                                old_peak = int(past[previous[layer, head].argmax()])
                            read = None
                            before, now, gain = "", "", ""
                            if len(remote):
                                values = current[layer, head, remote]
                                if t:
                                    values = values - previous[layer, head, remote]
                                # Zero-mass keys and decreases are not positive rereading events.
                                # Keep signed gain when a previously attended key loses mass.
                                active = current[layer, head, remote] > 0
                                if t:
                                    active |= previous[layer, head, remote] > 0
                                if active.any():
                                    k = int(remote[np.where(active, values, -np.inf).argmax()])
                                    read = int(past[k])
                                    now = float(current[layer, head, k])
                                    if t:
                                        before = float(previous[layer, head, k])
                                        gain = now - before
                            attention_writer.writerow(
                                prefix
                                + [
                                    query,
                                    text[query],
                                    layer,
                                    head,
                                    float(shift[layer, head])
                                    if np.isfinite(shift[layer, head])
                                    else "",
                                    old_peak if old_peak is not None else "",
                                    text[old_peak] if old_peak is not None else "",
                                    read if read is not None else "",
                                    text[read] if read is not None else "",
                                    "".join(text[max(0, read - 4) : min(query + 1, read + 5)])
                                    if read is not None
                                    else "",
                                    query - read if read is not None else "",
                                    before,
                                    now,
                                    gain,
                                ]
                            )
