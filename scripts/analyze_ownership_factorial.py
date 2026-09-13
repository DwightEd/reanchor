"""Verify saved ownership runs and report interventions without fitting labels."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from route_graph import adoption

from decoding.adoption_probe import sha256, write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def verify(run):
    manifest = json.loads((run / "manifest.json").read_text())
    for name, expected in manifest.items():
        if sha256(run / name) != expected:
            raise ValueError(f"manifest mismatch: {run / name}")
    return len(manifest)


def table(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args):
    run, output = args.run, args.output
    verified = {str(run): verify(run)}
    if args.multilayer:
        verified[str(args.multilayer)] = verify(args.multilayer)
    output.mkdir(parents=True, exist_ok=False)
    settings = json.loads((run / "settings.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    worlds = json.loads((run / "worlds.json").read_text())
    patches = json.loads((run / "patches.json").read_text())
    logits = {
        key: torch.from_numpy(np.load(run / f"{key}_logits.npy", allow_pickle=False))
        for key in ("s0h0", "s0h1", "s1h0", "s1h1")
    }
    tokens = np.load(run / "s0h0_tokens.npy", allow_pickle=False)
    twelve, fourteen = map(int, tokens[settings["source_swap_positions"]])
    steps = [t for start, stop in settings["windows"].values() for t in range(start, stop)]
    world_rows, contrasts = [], []
    for record in worlds:
        stop = settings["windows"][record["window"]][1]
        row = steps.index(stop - 1)
        native = logits[record["condition"]][row]
        margin = float(native[fourteen] - native[twelve])
        if margin != record["margin_14_minus_12"]:
            raise ValueError("saved world table disagrees with raw logits")
        world_rows.append(
            dict(
                condition=record["condition"],
                window=record["window"],
                margin_14_minus_12=margin,
                argmax_id=int(native.argmax()),
                max_tie_count=int((native == native.max()).sum()),
                entropy_nats=record["entropy_nats"],
            )
        )
    for window, (_, stop) in settings["windows"].items():
        row = steps.index(stop - 1)
        for kind, before, after in (
            ("source_h0", "s0h0", "s1h0"),
            ("source_h1", "s0h1", "s1h1"),
            ("history_s0", "s0h0", "s0h1"),
            ("history_s1", "s1h0", "s1h1"),
        ):
            effect = adoption.distribution_effect(
                logits[before][row : row + 1],
                logits[after][row : row + 1],
                [int(tokens[settings["queries"][row] + 1])],
            )
            contrasts.append(
                dict(
                    window=window, contrast=kind, **{key: value[0] for key, value in effect.items()}
                )
            )
    single_rows = [
        dict(
            window=r["window"],
            layer=r["layer"],
            scope=r["scope"],
            factor=r["factor"],
            margin=r["margin_14_minus_12"],
            margin_change=r["margin_change"],
            js_nats=r["distribution_effect"]["js_nats"][0],
            argmax_changed=r["distribution_effect"]["argmax_changed"][0],
        )
        for r in patches
        if r["factor"] != "sham"
    ]
    table(output / "worlds.csv", world_rows)
    table(output / "world_contrasts.csv", contrasts)
    table(output / "single_layer.csv", single_rows)
    numbers = dict(
        verified_manifest_counts=verified,
        factorial_effects=summary["factorial_effects"],
        worlds=world_rows,
        world_contrasts=contrasts,
        single_layer_conditions=len(single_rows),
        single_layer_native_argmax_flips=sum(r["argmax_changed"] for r in single_rows),
        original_control=json.loads((run / "correct_onion_control.json").read_text()),
        interpretation_scope=(
            "one source, one observed incorrect numeric window; "
            "counterfactual influence, not detection accuracy"
        ),
        scientific_review="REVIEW_UNAVAILABLE",
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for ax, window in zip(axes, settings["windows"], strict=True):
        values = np.array(
            [
                [
                    next(
                        r["margin_14_minus_12"]
                        for r in world_rows
                        if r["condition"] == f"s{s}h{h}" and r["window"] == window
                    )
                    for h in range(2)
                ]
                for s in range(2)
            ]
        )
        plot = ax.imshow(values, cmap="RdBu", vmin=-11, vmax=11)
        for s in range(2):
            for h in range(2):
                ax.text(h, s, f"{values[s, h]:.3f}", ha="center", va="center", color="black")
        ax.set(
            xticks=[0, 1],
            xticklabels=["History original", "History 14 -> 12"],
            yticks=[0, 1],
            yticklabels=["Source original", "Source 12 <-> 14"],
            title=window,
        )
    fig.colorbar(plot, ax=axes, label="logit(14) - logit(12)")
    fig.savefig(output / "source_history.svg")
    plt.close(fig)
    if args.multilayer:
        multi = args.multilayer
        records = json.loads((multi / "conditions.json").read_text())
        multi_rows = []
        for r in records:
            if r["window"] is None:
                continue
            window = r["window"]
            endpoint = r["endpoints"][window]
            stop = settings["windows"][window][1]
            multi_rows.append(
                dict(
                    condition=r["condition"],
                    window=window,
                    layer_count=len(r["layers"]),
                    query_count=len(r["prediction_steps"]),
                    factor=r["factor"],
                    margin=endpoint["margin_14_minus_12"],
                    margin_change=endpoint["margin_change"],
                    argmax_id=endpoint["argmax_id"],
                    argmax_text=endpoint["argmax_text"],
                    js_nats=r["trajectory"]["js_nats"][stop - 1],
                    argmax_changed=r["trajectory"]["argmax_changed"][stop - 1],
                )
            )
        table(output / "multilayer.csv", multi_rows)
        nodes = []
        for window, (start, stop) in settings["windows"].items():
            whole = next(r for r in multi_rows if r["condition"] == f"all_{window}_window_x")
            for t in range(start, stop):
                only = next(r for r in multi_rows if r["condition"] == f"only_{window}_{t}")
                leave = next(r for r in multi_rows if r["condition"] == f"leave_{window}_{t}")
                nodes.append(
                    dict(
                        window=window,
                        prediction_step=t,
                        only_margin_change=only["margin_change"],
                        full_minus_leave_one_out=whole["margin_change"] - leave["margin_change"],
                        only_argmax=only["argmax_text"],
                        leave_argmax=leave["argmax_text"],
                    )
                )
        table(output / "individual_queries.csv", nodes)
        numbers["multilayer_summary"] = json.loads((multi / "summary.json").read_text())
        numbers["clean_control_with_native_ties"] = json.loads(
            (multi / "clean_control.json").read_text()
        )
        numbers["multilayer_matrix"] = [
            r for r in multi_rows if not r["condition"].startswith(("only_", "leave_"))
        ]
        numbers["individual_queries"] = nodes
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
        for col, window in enumerate(settings["windows"]):
            for factor in ("x", "e", "xe", "mlp"):
                values = [
                    next(
                        r["margin_change"]
                        for r in multi_rows
                        if r["condition"] == f"all_{window}_{scope}_{factor}"
                    )
                    for scope in ("query", "window", "history")
                ]
                axes[0, col].plot([0, 1, 2], values, "o-", label=factor.upper())
            axes[0, col].set(
                title=f"{window}: all 32 layers",
                xticks=[0, 1, 2],
                xticklabels=["Final query", "Current window", "All history"],
                ylabel="Change in logit(14) - logit(12)",
            )
            axes[0, col].axhline(0, color="gray", lw=0.5)
            axes[0, col].legend()
            subset = [r for r in nodes if r["window"] == window]
            for field, label in (
                ("only_margin_change", "Only this query"),
                ("full_minus_leave_one_out", "Full minus leave-one-out"),
            ):
                axes[1, col].plot(
                    [r["prediction_step"] for r in subset],
                    [r[field] for r in subset],
                    "o-",
                    label=label,
                )
            axes[1, col].set(
                xlabel="Prediction step (patched state is one token earlier)",
                ylabel="Signed margin effect",
                title=f"{window}: individual queries, all-layer X",
            )
            axes[1, col].legend()
        fig.savefig(output / "multilayer_nodes.svg")
        plt.close(fig)
    write_json(output / "numbers.json", numbers)
    write_json(
        output / "provenance.json",
        dict(
            analyzer_sha256=sha256(Path(__file__)),
            adoption_sha256=sha256(Path(adoption.__file__)),
            input_manifests={
                str(p): sha256(p / "manifest.json") for p in (run, args.multilayer) if p
            },
        ),
    )
    write_json(output / "manifest.json", {p.name: sha256(p) for p in sorted(output.iterdir())})
    print(json.dumps(dict(verified=verified, effects=numbers["factorial_effects"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--multilayer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    analyze(parser.parse_args())
