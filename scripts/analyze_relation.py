"""Raw-logit verification and scoped O4/O5 summaries; no fitted thresholds."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(directory):
    if not (directory / "COMPLETE").exists():
        raise ValueError("capture is not complete")
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest.items():
        if sha256(directory / name) != expected:
            raise ValueError(f"artifact changed: {directory / name}")
    return len(manifest)


def native(logits, twelve, fourteen):
    if logits.shape != (128256,) or not np.isfinite(logits).all():
        raise ValueError("invalid full vocabulary endpoint")
    choice = int(logits.argmax())
    ties = int((logits == logits.max()).sum())
    return dict(
        margin=float(logits[fourteen] - logits[twelve]),
        choice={twelve: "12", fourteen: "14"}.get(choice, f"token:{choice}"),
        id=choice,
        ties=ties,
    )


def analyze(args):
    counts = {"O4": verify(args.relation), "O5": verify(args.routing)}
    inputs = json.loads((args.relation / "inputs.json").read_text())
    first = inputs["real_after_0"]
    twelve, fourteen = [first["token_ids"][p] for p in first["numeric_positions"]]
    worlds = json.loads((args.relation / "worlds.json").read_text())
    for row in worlds:
        key = f"{row['context']}_{row['world']}"
        logits = np.load(args.relation / f"{key}_logits.npy", mmap_mode="r")
        measured = native(logits[-1], twelve, fourteen)
        if (
            measured["margin"] != row["margin_14_minus_12"]
            or measured["id"] != row["argmax_id"]
            or measured["ties"] != row["max_tie_count"]
        ):
            raise ValueError("world raw logits and summary differ")
        with np.load(args.relation / f"{key}_readouts.npz") as scores:
            for name in ("node", "attention", "graph"):
                values = scores[name][:, -1].mean(0)
                if not np.allclose(values, row["readouts"][name]["values_12_14"], atol=1e-9):
                    raise ValueError("saved readout and summary differ")
        row["grill_control"] = native(logits[99], twelve, fourteen)
    patches = json.loads((args.relation / "patches.json").read_text())
    routes = json.loads((args.routing / "patches.json").read_text())
    for rows, directory, ending in (
        (patches, args.relation, "_endpoint"),
        (routes, args.routing, "_logits"),
    ):
        for row in rows:
            logits = np.load(directory / f"{row['condition']}{ending}.npy")
            value = native(logits, twelve, fourteen)
            if (
                value["margin"] != row["margin_14_minus_12"]
                or value["id"] != row["argmax_id"]
                or value["ties"] != row["max_tie_count"]
            ):
                raise ValueError("patch raw logits and summary differ")
            if row["earlier_query_max_logit_error"] != 0:
                raise ValueError("reported earlier-query error is nonzero")
    x_scan = [r for r in patches if "_scan_" in r["condition"]]
    e_scan = [r for r in routes if r["condition"].startswith("query_")]
    layer_scan = [r for r in routes if r["condition"].startswith("layer_")]
    if (len(worlds), len(patches), len(x_scan), len(routes), len(e_scan), len(layer_scan)) != (
        4,
        160,
        132,
        167,
        132,
        32,
    ):
        raise ValueError("incomplete planned condition coverage")

    def sufficient(rows):
        return [r["steps"][0] for r in rows if r["argmax_text"] == "12" and r["max_tie_count"] == 1]

    top8 = json.loads((args.relation / "candidate_queries.json").read_text())["top8"]
    e_positive = sufficient(e_scan)
    result = dict(
        scope="one source, two controlled query contexts; not natural detection accuracy",
        verified_files=counts,
        worlds=worlds,
        readout_correct_counts={
            name: sum(r["readouts"][name]["choice"] == r["expected_value"] for r in worlds)
            for name in ("node", "attention", "graph")
        },
        fixed_patches=patches[:28],
        x_single_query_unique_flips=sufficient(x_scan),
        e_single_query_unique_flips=e_positive,
        e_single_layer_unique_flips=[
            r["layers"][0]
            for r in layer_scan
            if r["argmax_text"] == "12" and r["max_tie_count"] == 1
        ],
        routing_scope_controls=[
            r for r in routes if r["condition"] in ("all_queries", "without_final", "sham")
        ],
        candidate_top8=top8,
        candidate_recall=(
            len(set(top8) & set(e_positive)) / len(e_positive) if e_positive else None
        ),
        candidate_target="single-query E yields unique 12; not universal lookback truth",
        raw_manifest_sha256={
            "O4": sha256(args.relation / "manifest.json"),
            "O5": sha256(args.routing / "manifest.json"),
        },
        scientific_review="REVIEW_UNAVAILABLE",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "numbers.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(
        [r["steps"][0] for r in x_scan],
        [r["margin_14_minus_12"] for r in x_scan],
        label="source V exchange (X)",
    )
    axes[0].plot(
        [r["steps"][0] for r in e_scan],
        [r["margin_14_minus_12"] for r in e_scan],
        label="source routing exchange (E)",
    )
    axes[0].scatter(top8, [1.25] * len(top8), marker="x", color="black", label="source-rise top8")
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].set(
        xlabel="Intervened prediction step t (query = P+t-1)",
        ylabel="Final logit(14) - logit(12)",
        title="One-query interventions, all 32 layers",
    )
    axes[0].legend(fontsize=8)
    axes[1].plot(
        [r["layers"][0] for r in layer_scan], [r["margin_14_minus_12"] for r in layer_scan]
    )
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].axhline(1.25, color="gray", linestyle="--", label="baseline")
    axes[1].set(
        xlabel="Intervened layer (zero-based)",
        ylabel="Final logit(14) - logit(12)",
        title="Final-query routing exchange, one layer",
    )
    axes[1].legend()
    fig.savefig(args.output / "relation_routing.png", dpi=160)
    fig.savefig(args.output / "relation_routing.pdf")
    plt.close(fig)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "verified_files",
                    "readout_correct_counts",
                    "x_single_query_unique_flips",
                    "e_single_query_unique_flips",
                    "e_single_layer_unique_flips",
                    "candidate_top8",
                    "candidate_recall",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relation", type=Path, default=Path("outputs/relation_only_20260913"))
    parser.add_argument("--routing", type=Path, default=Path("outputs/relation_routing_20260913"))
    parser.add_argument("--output", type=Path, required=True)
    analyze(parser.parse_args())
