"""Independent-source factorial contrasts and complete single-layer spectrum."""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from analyze_ownership_factorial import table, verify
from route_graph import adoption

from decoding.adoption_probe import sha256, write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def analyze(args):
    run, output = args.run, args.output
    verified = verify(run)
    settings = json.loads((run / "settings.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    worlds = json.loads((run / "source_worlds.json").read_text())
    layers = json.loads((run / "layer_patches.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    contrasts, rows = [], []
    for trace, windows in settings["cases"].items():
        records = [r for r in worlds if r["trace"] == trace]
        logits = {
            (r["a"], r["b"], r["h"]): torch.from_numpy(
                np.load(run / f"{r['condition']}_logits.npy")
            )
            for r in records
        }
        for record in records:
            current = logits[record["a"], record["b"], record["h"]]
            for row, name in enumerate(windows):
                point = record["endpoints"][name]
                if int(current[row].argmax()) != point["argmax_id"]:
                    raise ValueError("native source argmax mismatch")
                rows.append(
                    dict(
                        trace=trace,
                        window=name,
                        a=record["a"],
                        b=record["b"],
                        h=record["h"],
                        margin=point["margin_14_minus_12"] if name in ("grill", "onion") else None,
                        argmax=point["argmax_text"],
                        max_ties=point["max_tie_count"],
                    )
                )
        # Token values are saved for every actual input, with no reconstructed prompt.
        first = next(r for r in records if (r["a"], r["b"], r["h"]) == (0, 0, 0))
        tokens = np.load(run / f"{first['condition']}_tokens.npy")
        # Prompt size verified from the original trace identified in the input hash map.
        original = next(
            Path(p)
            for p in settings["input_sha256"]
            if p.endswith(f"samples_20260911_145421_235/{trace}.npz")
        )
        if sha256(original) != settings["input_sha256"][str(original)]:
            raise ValueError(f"original trace identity changed: {original}")
        with np.load(original, allow_pickle=False) as saved:
            prompt = int(saved["prompt_length"])
        for axis, factor in enumerate(("a", "b", "h")):
            for others in itertools.product((0, 1), repeat=2):
                before, after = list(others), list(others)
                before.insert(axis, 0)
                after.insert(axis, 1)
                effect = adoption.distribution_effect(
                    logits[tuple(before)],
                    logits[tuple(after)],
                    [int(tokens[prompt + t]) for t in windows.values()],
                )
                for row, name in enumerate(windows):
                    contrasts.append(
                        dict(
                            trace=trace,
                            window=name,
                            factor=factor,
                            before="".join(map(str, before)),
                            after="".join(map(str, after)),
                            js_nats=effect["js_nats"][row],
                            argmax_changed=effect["argmax_changed"][row],
                            saved_token_logp_change=effect["saved_token_logp_change"][row],
                        )
                    )
    layer_rows = []
    for record in layers:
        row = ["grill", "onion"].index(record["window"])
        logits = np.load(run / f"{record['condition']}_logits.npy")[row]
        if (
            int(logits.argmax()) != record["argmax_id"]
            or int((logits == logits.max()).sum()) != record["max_tie_count"]
        ):
            raise ValueError("layer native result mismatch")
        layer_rows.append(
            dict(
                layer=record["layer"],
                window=record["window"],
                margin=record["margin_14_minus_12"],
                margin_change=record["margin_change"],
                argmax=record["argmax_text"],
                argmax_changed=record["effect"]["argmax_changed"][0],
                js_nats=record["effect"]["js_nats"][0],
            )
        )
    table(output / "source_worlds.csv", rows)
    table(output / "source_contrasts.csv", contrasts)
    table(output / "all_single_layers.csv", layer_rows)
    contrast_summary = []
    for trace, windows in settings["cases"].items():
        for name in windows:
            for factor in ("a", "b", "h"):
                selected = [
                    r
                    for r in contrasts
                    if r["trace"] == trace and r["window"] == name and r["factor"] == factor
                ]
                contrast_summary.append(
                    dict(
                        trace=trace,
                        window=name,
                        factor=factor,
                        mean_js_nats=float(np.mean([r["js_nats"] for r in selected])),
                        max_js_nats=max(r["js_nats"] for r in selected),
                        native_argmax_changes=sum(r["argmax_changed"] for r in selected),
                        comparisons=len(selected),
                    )
                )
    layer_summary = {}
    for window in ("grill", "onion"):
        selected = [r for r in layer_rows if r["window"] == window]
        layer_summary[window] = dict(
            conditions=len(selected),
            native_argmax_flips=sum(r["argmax_changed"] for r in selected),
            min_margin_change=min(r["margin_change"] for r in selected),
            max_margin_change=max(r["margin_change"] for r in selected),
            largest_absolute_effect=max(selected, key=lambda r: abs(r["margin_change"])),
        )
    numbers = dict(
        verified_manifest_files=verified,
        summary=summary,
        factor_distribution_effects=contrast_summary,
        single_layer_summary=layer_summary,
        scientific_review="REVIEW_UNAVAILABLE",
        scope="4 response seeds from one source, one observed incorrect numeric window",
    )
    write_json(output / "numbers.json", numbers)
    (output / "executed_analyzer.py").write_bytes(Path(__file__).read_bytes())
    (output / "executed_shared_analyzer.py").write_bytes(
        Path(__file__).with_name("analyze_ownership_factorial.py").read_bytes()
    )
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    for window in ("grill", "onion"):
        selected = sorted(
            [r for r in layer_rows if r["window"] == window], key=lambda r: r["layer"]
        )
        ax.plot(
            [r["layer"] for r in selected],
            [r["margin_change"] for r in selected],
            "o-",
            label=window,
        )
    ax.axhline(0, color="gray", lw=0.5)
    ax.set(
        xlabel="Single patched layer (zero based)",
        ylabel="Change in logit(14) - logit(12)",
        title="Final-query source V exchange, one layer at a time",
        xticks=list(range(0, 32, 2)),
    )
    ax.legend()
    fig.savefig(output / "single_layers.svg")
    plt.close(fig)
    write_json(
        output / "provenance.json",
        dict(
            analyzer_sha256=sha256(Path(__file__)),
            shared_analyzer_sha256=sha256(
                Path(__file__).with_name("analyze_ownership_factorial.py")
            ),
            adoption_sha256=sha256(Path(adoption.__file__)),
            input_manifest_sha256=sha256(run / "manifest.json"),
        ),
    )
    write_json(output / "manifest.json", {p.name: sha256(p) for p in sorted(output.iterdir())})
    print(json.dumps(dict(layers=layer_summary, source=summary["source_main_effects"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    analyze(parser.parse_args())
