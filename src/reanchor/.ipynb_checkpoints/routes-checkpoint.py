"""Read head dispersion, disagreement and source-edge changes in saved generations."""

import csv
import json
import re
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from reanchor.io import empty_directory, read_jsonl


def entropy(probabilities: np.ndarray) -> np.ndarray:
    """Shannon entropy in bits along the final axis; zero mass contributes zero."""
    logs = np.zeros_like(probabilities)
    np.log2(probabilities, out=logs, where=probabilities > 0)
    return -(probabilities * logs).sum(-1)


def valid_mean(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    counts = valid.sum(-1)
    return np.divide(
        np.where(valid, values, 0).sum(-1),
        counts,
        out=np.full(counts.shape, np.nan),
        where=counts > 0,
    )


def source_positions(tokenizer, prompt: str, prompt_ids: np.ndarray):
    """Align passage bodies to captured IDs, excluding question, labels and instructions."""
    # A template may embed today's date. Recover the captured prefix instead of
    # applying today's template to an older sample.
    rendered = tokenizer.decode(
        prompt_ids.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if not np.array_equal(encoded["input_ids"], prompt_ids):
        raise ValueError("tokenizer cannot round-trip the captured prompt token IDs")
    if rendered.count(prompt) != 1:
        raise ValueError("raw prompt must occur exactly once in the decoded captured prefix")
    origin = rendered.index(prompt)
    markers = list(re.finditer(r"(?m)^passage ([123]):", prompt))
    footer = prompt.find(
        "\n\nIn case the passages do not contain", markers[-1].end() if markers else 0
    )
    if [m.group(1) for m in markers] != ["1", "2", "3"] or footer < 0:
        raise ValueError("expected the three RAGTruth QA passages and the answer instruction")
    passage = np.zeros(len(prompt_ids), dtype=int)
    contexts = {}
    for i, marker in enumerate(markers):
        start = marker.end()
        end = markers[i + 1].start() if i < 2 else footer
        body = prompt[start:end]
        start += len(body) - len(body.lstrip())
        end -= len(body) - len(body.rstrip())
        start, end = start + origin, end + origin
        for key, (left, right) in enumerate(encoded["offset_mapping"]):
            if right > start and left < end and prompt_ids[key] not in tokenizer.all_special_ids:
                passage[key] = i + 1
                contexts[key] = rendered[max(start, left - 50) : min(end, right + 50)]
    positions = np.flatnonzero(passage)
    if set(passage[positions]) != {1, 2, 3}:
        raise ValueError("each passage must contain at least one ordinary token")
    return positions, passage, contexts


class RouteAnalysis:
    def __init__(
        self,
        samples: Path,
        cases: Path,
        output: Path,
        tokenizer: Path | None,
        before: int,
        after: int,
        baseline: int,
    ):
        self.samples, self.cases, self.output = samples, cases, output
        self.tokenizer_path = tokenizer
        self.before, self.after, self.baseline = before, after, baseline

    def run(self) -> None:
        if self.before < 0 or self.after < 0 or self.baseline < 1:
            raise ValueError("windows must be nonnegative and baseline positive")
        with self.cases.open(encoding="utf-8", newline="") as stream:
            cases = list(csv.DictReader(stream))
        if not cases or any(not row.get("focus") for row in cases):
            raise ValueError("cases must specify source_id, seed and a nonempty focus quote")
        samples = list(read_jsonl(self.samples / "samples.jsonl"))
        prompts = {
            str(row["source_id"]): row["prompt"]
            for row in read_jsonl(self.samples / "prompts.jsonl")
        }
        selected = []
        for case in cases:
            matches = [
                s
                for s in samples
                if str(s["source_id"]) == case["source_id"] and s["seed"] == int(case["seed"])
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one captured sample for {case['source_id']} seed {case['seed']}"
                )
            sample = matches[0]
            if Path(sample["trace"]).name != sample["trace"] or not sample["trace"].endswith(
                ".npz"
            ):
                raise ValueError("trace must be an NPZ filename within samples")
            if sample["response"].count(case["focus"]) != 1:
                raise ValueError(
                    f"{sample['trace']}: focus quote must match exactly once; check this generation"
                )
            selected.append((sample, case["focus"]))
        if len({s["trace"] for s, _ in selected}) != len(selected):
            raise ValueError("specify only one focus window per captured sample")
        path = self.tokenizer_path
        if path is None:
            path = json.loads((self.samples / "settings.json").read_text(encoding="utf-8"))["model"]
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
        if not tokenizer.is_fast:
            raise ValueError("source alignment requires a fast tokenizer with character offsets")
        empty_directory(self.output)
        for sample, focus in tqdm(selected, desc="routes", unit="answer"):
            self._analyze(sample, prompts[str(sample["source_id"])], focus, tokenizer)

    def _analyze(self, sample, prompt, focus, tokenizer):
        with np.load(self.samples / sample["trace"], allow_pickle=False) as trace:
            weights, ids = trace["attention"], trace["token_ids"]
            prompt_length = int(trace["prompt_length"])
            pieces, text = trace["token_pieces"], trace["token_text"]
            top_logits = trace["top_logits"]
            logit_entropy = trace["logit_entropy"] if "logit_entropy" in trace else None
        layers, heads, steps, length = weights.shape
        if prompt_length < 1 or length != prompt_length + steps or len(ids) != length:
            raise ValueError("trace does not match generation positions")
        keys, passage, contexts = source_positions(tokenizer, prompt, ids[:prompt_length])
        response_ids = ids[prompt_length:].tolist()
        decode = dict(skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if tokenizer.decode(response_ids, **decode) != sample["response"]:
            raise ValueError("captured response IDs do not decode to samples.jsonl")
        # Incomplete UTF-8 byte tokens decode to replacement characters. Align only
        # the prefix shared with the final text, and include every byte of a character.
        offsets = []
        for i in range(steps + 1):
            prefix = tokenizer.decode(response_ids[:i], **decode)
            common = next(
                (j for j, (a, b) in enumerate(zip(prefix, sample["response"])) if a != b),
                min(len(prefix), len(sample["response"])),
            )
            offsets.append(common)
        spans = []
        for i, start in enumerate(offsets[:-1]):
            end = offsets[i + 1]
            if end == start and response_ids[i] not in tokenizer.all_special_ids:
                end = next((n for n in offsets[i + 1 :] if n > start), start)
            spans.append((start, end))
        left = sample["response"].index(focus)
        right = left + len(focus)
        focus_steps = [
            i
            for i, (start, end) in enumerate(spans)
            if end > left and start < right and end > start
        ]
        if not focus_steps:
            raise ValueError("focus quote does not overlap generated tokens")
        first, last = focus_steps[0], focus_steps[-1]
        window = range(max(0, first - self.before), min(steps, last + self.after + 1))
        output = self.output / Path(sample["trace"]).stem
        empty_directory(output)
        curves = np.full((4, layers, steps), np.nan)
        with ExitStack() as stack:
            writers = {}
            columns = {
                "tokens": (
                    "step position query_position token_id token_piece token focus "
                    "top2_margin logit_entropy"
                ),
                "source_tokens": "position passage token_id token_piece token context",
                "trajectory": (
                    "step layer source_mass valid_heads dispersion disagreement "
                    "shift shift_baseline"
                ),
                "reading": (
                    "step layer head source_mass dispersion shift "
                    "peak_position peak_passage peak_context peak_attention "
                    "gain_position gain_passage gain_context previous_attention attention gain"
                ),
            }
            for name, header in columns.items():
                stream = stack.enter_context(
                    (output / f"{name}.csv").open("x", encoding="utf-8", newline="")
                )
                writers[name] = csv.writer(stream)
                writers[name].writerow(header.split())
            for key in keys:
                writers["source_tokens"].writerow(
                    [key, passage[key], ids[key], pieces[key], text[key], contexts[key]]
                )
            previous, old_mass, old_p = None, None, None
            for t in tqdm(range(steps), desc=sample["trace"], unit="token", leave=False):
                position = prompt_length + t
                writers["tokens"].writerow(
                    [
                        t,
                        position,
                        position - 1,
                        ids[position],
                        pieces[position],
                        text[position],
                        int(t in focus_steps),
                        float(top_logits[t, 0] - top_logits[t, 1]),
                        "" if logit_entropy is None else float(logit_entropy[t]),
                    ]
                )
                current = weights[:, :, t, keys].astype(np.float64)
                if not np.isfinite(current).all() or (current < 0).any():
                    raise ValueError(f"{sample['trace']} step {t}: invalid source attention")
                mass = current.sum(-1)
                valid = mass > 0
                p = np.divide(
                    current, mass[..., None], out=np.zeros_like(current), where=valid[..., None]
                )
                head_entropy = entropy(p)
                dispersion = valid_mean(head_entropy, valid)
                counts = valid.sum(-1)
                pooled = np.divide(
                    p.sum(1),
                    counts[:, None],
                    out=np.zeros((layers, len(keys))),
                    where=counts[:, None] > 0,
                )
                disagreement = entropy(pooled) - dispersion
                shifts = np.full((layers, heads), np.nan)
                if previous is not None:
                    pair = valid & (old_mass > 0)
                    shifts[pair] = (0.5 * np.abs(p - old_p).sum(-1))[pair]
                shift = valid_mean(shifts, np.isfinite(shifts))
                curves[:, :, t] = np.stack([dispersion, disagreement, shift, mass.mean(-1)])
                for layer in range(layers):
                    history = curves[2, layer, max(0, t - self.baseline) : t]
                    history = history[np.isfinite(history)]
                    values = [
                        t,
                        layer,
                        mass[layer].mean(),
                        counts[layer],
                        dispersion[layer],
                        disagreement[layer],
                        shift[layer],
                        np.median(history) if len(history) else np.nan,
                    ]
                    writers["trajectory"].writerow(
                        ["" if not np.isfinite(v) else v for v in values]
                    )
                    if t not in window:
                        continue
                    for head in range(heads):
                        peak = int(current[layer, head].argmax()) if valid[layer, head] else None
                        gain = (
                            None
                            if previous is None
                            else current[layer, head] - previous[layer, head]
                        )
                        gained = None
                        if gain is not None:
                            active = (current[layer, head] > 0) | (previous[layer, head] > 0)
                            if active.any():
                                gained = int(np.where(active, gain, -np.inf).argmax())
                        peak_key = keys[peak] if peak is not None else None
                        gain_key = keys[gained] if gained is not None else None
                        writers["reading"].writerow(
                            [
                                t,
                                layer,
                                head,
                                float(mass[layer, head]),
                                float(head_entropy[layer, head]) if valid[layer, head] else "",
                                float(shifts[layer, head])
                                if np.isfinite(shifts[layer, head])
                                else "",
                                *(
                                    [
                                        peak_key,
                                        passage[peak_key],
                                        contexts[peak_key],
                                        float(current[layer, head, peak]),
                                    ]
                                    if peak_key is not None
                                    else [""] * 4
                                ),
                                *(
                                    [
                                        gain_key,
                                        passage[gain_key],
                                        contexts[gain_key],
                                        float(previous[layer, head, gained]),
                                        float(current[layer, head, gained]),
                                        float(gain[gained]),
                                    ]
                                    if gain_key is not None
                                    else [""] * 6
                                ),
                            ]
                        )
                previous, old_mass, old_p = current, mass, p
        plot_trajectory(curves, first, last, output / "trajectory.png", sample)
        tqdm.write(
            f"{sample['source_id']} seed={sample['seed']} trace={sample['trace']} "
            f"focus={first}:{last + 1} window={window.start}:{window.stop}"
        )


def plot_trajectory(curves, first, last, path, sample):
    """Plot every layer; the shaded quote is an inspection window, not an error label."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    figure = Figure(figsize=(13, 9), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(4, 1, sharex=True)
    layers, steps = curves.shape[1:]
    for index, (values, axis, title) in enumerate(
        zip(
            curves,
            axes,
            [
                "Within-head entropy (bits)",
                "Between-head disagreement (bits)",
                "Source redistribution (TV)",
                "Source attention mass",
            ],
        )
    ):
        finite = values[np.isfinite(values)]
        maximum = max(float(finite.max()), 1e-6) if len(finite) else 1
        if index >= 2:
            maximum = 1  # Fixed scales for TV and raw mass across examples.
        heatmap = axis.imshow(
            values,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            vmin=0,
            vmax=maximum,
            extent=(-0.5, steps - 0.5, -0.5, layers - 0.5),
        )
        axis.axvspan(first - 0.5, last + 0.5, facecolor="none", edgecolor="red", linewidth=1.5)
        axis.set_ylabel("Layer (0-based)")
        axis.set_yticks(np.unique(np.linspace(0, layers - 1, min(layers, 9), dtype=int)))
        axis.set_title(title, loc="left")
        figure.colorbar(heatmap, ax=axis, pad=0.01)
    axes[-1].set_xlabel("Prediction step t: attention sees only response tokens before t")
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    figure.suptitle(
        f"source={sample['source_id']} seed={sample['seed']} "
        "| red box: quoted text, not an error label"
    )
    figure.savefig(path, dpi=150)
