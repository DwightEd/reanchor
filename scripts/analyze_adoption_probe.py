"""Recompute numerical pilot summaries and a standalone influence plot."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def analyze(run, output):
    manifest = json.loads((run / "manifest.json").read_text())
    for name, expected in manifest.items():
        if hashlib.sha256((run / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"manifest mismatch: {name}")
    output.mkdir(parents=True, exist_ok=False)
    influence = json.loads((run / "influence.json").read_text())
    rows = []
    for path in sorted(run.glob("messages_*.json")):
        record = json.loads(path.read_text())
        rows.append(
            dict(
                step=record["step"],
                saved_token=record["next_saved_token"],
                native_pair=record["candidate_text"][:2],
                competition=record["opposition"]["competition"],
                absolute_mass=record["opposition"]["absolute_mass"],
                reconstruction_relative_l2=record["reconstruction"]["relative_l2_error"],
                selected_logit_mismatch=record["surrogate_max_logit_error"],
                local_error=record["finite_attenuation"]["0.1"]["relative_l1_error"],
                small_error=record["finite_attenuation"].get("0.01", {}).get("relative_l1_error"),
                full_error=record["finite_attenuation"]["1.0"]["relative_l1_error"],
                all_candidate_local_error=record["finite_attenuation"]["0.1"][
                    "all_candidate_relative_l1_error"
                ],
            )
        )
    conditions = {}
    for name, values in influence["conditions"].items():
        js = np.asarray(values["js_nats"])
        conditions[name] = dict(
            post134_mean_js=float(js[134:].mean()),
            post134_argmax_flips=int(np.sum(values["argmax_changed"][134:])),
            largest_post134_step=int(134 + np.argmax(js[134:])),
            largest_post134_js=float(js[134:].max()),
            pre_activation_max_logit_error=values["max_logit_error_before_activation"],
        )
    summary = dict(
        run=str(run.resolve()),
        manifest_sha256=hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        query_rows=rows,
        conditions=conditions,
        interpretation="single inspected case; mechanism/numerical diagnostics only",
    )
    (output / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    steps = np.arange(len(influence["response_tokens"]))
    axes[0].plot(steps, influence["entropy_nats"], color="#444444")
    axes[0].set_ylabel("Native entropy (nats)")
    axes[0].axvspan(119, 133, color="#d55e00", alpha=0.15, label="Inspected history segment")
    for name, color in (("history_cut", "#d55e00"), ("earlier_cut", "#0072b2")):
        axes[1].plot(
            steps[134:], influence["conditions"][name]["js_nats"][134:], label=name, color=color
        )
    axes[1].set_ylabel("JS after access cut (nats)")
    axes[1].set_xlabel("Prediction step t (saved continuation fixed)")
    for ax in axes:
        ax.axvline(134, color="#777777", linestyle="--", label="First affected prediction")
        ax.axvline(157, color="#009e73", linestyle=":", label="'Note' token")
        ax.set_xlim(80, len(steps) - 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("One inspected response: influence persists into a discourse transition")
    fig.savefig(output / "influence.png", dpi=180)
    fig.savefig(output / "influence.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.run, args.output)
