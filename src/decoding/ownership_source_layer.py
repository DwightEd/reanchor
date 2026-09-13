"""Disambiguate two source edits and exhaust single-layer X interventions."""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from route_graph import adoption, ownership
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import __version__ as transformers_version

from decoding import adoption_probe, ownership_factorial, ownership_multilayer
from decoding.adoption_probe import sha256, write_json
from decoding.ownership_factorial import describe, load_trace
from decoding.ownership_multilayer import native

CASES = {
    "00012": {"grill": 99, "onion": 131},
    "00013": {"grill": 100, "correct_onion": 126},
    "00014": {"grill": 98, "simmer_control": 61},
    "00015": {"grill": 109, "alternative_grill": 142},
}


def run(args):
    root = Path(__file__).resolve().parents[2]
    samples = root / "outputs/samples_20260911_145421_235"
    states = root / "outputs/states_samples_20260911_145421_235"
    previous = root / "outputs/ownership_multilayer_20260912"
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = json.loads((samples / "settings.json").read_text())
    code = [
        Path(module.__file__)
        for module in (
            ownership,
            adoption,
            ownership_factorial,
            ownership_multilayer,
            adoption_probe,
        )
    ]
    code += [Path(__file__), root / "scripts/run_ownership_source_layer.sh"]
    inputs = [samples / f"{name}.npz" for name in CASES]
    inputs += [states / f"{name}.npz" for name in CASES]
    inputs += [
        samples / "settings.json",
        samples / "samples.jsonl",
        previous / "manifest.json",
        root.parent / "graph/docs/OWNERSHIP_SOURCE_LAYER_PLAN_20260912.md",
    ]
    original = root / "outputs/ownership_factorial_20260912"
    inputs += [original / "manifest.json", original / "patches.json"]
    inputs += [original / f"s{s}h{h}_logits.npy" for s, h in itertools.product((0, 1), repeat=2)]
    # The prior manifest authenticates the reused factors; verify it before loading.
    for name, digest in json.loads((previous / "manifest.json").read_text()).items():
        if sha256(previous / name) != digest:
            raise ValueError(f"previous manifest mismatch: {name}")
    settings = dict(
        model=config["model"],
        cases=CASES,
        dtype="bfloat16",
        attention="eager",
        seed=20260912,
        torch=torch.__version__,
        transformers=transformers_version,
        gpu=torch.cuda.get_device_name(0),
        source_edits={"a": [76, "12", "14"], "b": [341, "14", "12"]},
        code_sha256={str(p.resolve()): sha256(p) for p in code},
        input_sha256={str(p.resolve()): sha256(p) for p in inputs},
        model_files=[
            dict(name=p.name, size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
            for p in sorted(Path(config["model"]).iterdir())
            if p.is_file()
        ],
    )
    for p in code:
        (output / f"executed_{p.name}").write_bytes(p.read_bytes())
    write_json(output / "settings.json", settings)
    torch.manual_seed(20260912)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
    value_ids = [tokenizer.encode(value, add_special_tokens=False) for value in ("14", "12")]
    if any(len(v) != 1 for v in value_ids):
        raise ValueError("expected single token constraint values")
    fourteen, twelve = [v[0] for v in value_ids]
    candidates = (fourteen, twelve)
    model = (
        AutoModelForCausalLM.from_pretrained(
            config["model"],
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .to("cuda:0")
        .eval()
        .requires_grad_(False)
    )
    records, effects, futures = [], {}, {}
    source_reproduction = []
    for trace, windows in CASES.items():
        ids, prompt, mask = load_trace(samples, states, f"{trace}.npz")
        history_step = windows["grill"]
        if ids[76] != twelve or ids[341] != fourteen or ids[prompt + history_step] != fourteen:
            raise ValueError(f"preinspected value identity differs: {trace}")
        positions = list(windows.values())
        queries = [prompt + t - 1 for t in positions]
        worlds = {}
        for a, b, h in itertools.product((0, 1), repeat=3):
            changed = ids.copy()
            if a:
                changed[76] = fourteen
            if b:
                changed[341] = twelve
            if h:
                changed[prompt + history_step] = twelve
            logits, _ = ownership.intervene_source_factors(
                model, changed[:-1], queries, np.flatnonzero(mask), []
            )
            key = f"{trace}_a{a}b{b}h{h}"
            worlds[a, b, h] = logits
            np.save(output / f"{key}_logits.npy", logits.numpy(), allow_pickle=False)
            np.save(output / f"{key}_tokens.npy", changed, allow_pickle=False)
            base = worlds[0, 0, 0]
            record = dict(
                trace=trace,
                a=a,
                b=b,
                h=h,
                condition=key,
                endpoints={
                    name: dict(
                        prediction_step=step,
                        saved_token=tokenizer.decode([int(ids[prompt + step])]),
                        **describe(logits[row], candidates, tokenizer),
                        **native(logits[row], tokenizer),
                    )
                    for row, (name, step) in enumerate(windows.items())
                },
                effect=adoption.distribution_effect(
                    base, logits, [int(ids[prompt + t]) for t in positions]
                ),
            )
            records.append(record)
            if trace == "00012" and a == b:
                old = torch.from_numpy(
                    np.load(root / f"outputs/ownership_factorial_20260912/s{a}h{h}_logits.npy")
                )[[6, 19]]
                error = float((logits - old).abs().max())
                source_reproduction.append(error)
                if error:
                    raise ValueError("first factorial worlds did not reproduce")
        before = [row for row, step in enumerate(positions) if step <= history_step]
        error = max(
            float((worlds[a, b, 0][before] - worlds[a, b, 1][before]).abs().max())
            for a, b in itertools.product((0, 1), repeat=2)
        )
        futures[trace] = error
        if error:
            raise ValueError("future history affected a preceding query")
        effects[trace] = {}
        for row, name in enumerate(windows):
            # Only grill/onion are numeric ownership comparisons; others retain native outcomes.
            if name not in ("grill", "onion"):
                continue
            margins = np.array(
                [
                    [
                        [
                            float(worlds[a, b, h][row, fourteen] - worlds[a, b, h][row, twelve])
                            for h in (0, 1)
                        ]
                        for b in (0, 1)
                    ]
                    for a in (0, 1)
                ]
            )
            effects[trace][name] = dict(
                margins_a_b_h=margins.tolist(),
                a_main=float((margins[1] - margins[0]).mean()),
                b_main=float((margins[:, 1] - margins[:, 0]).mean()),
                h_main=float((margins[:, :, 1] - margins[:, :, 0]).mean()),
            )
        write_json(output / "source_worlds.json", records)
        print(f"SOURCE {trace}: {effects[trace]}", flush=True)
    ids, prompt, mask = load_trace(samples, states, "00012.npz")
    sources = np.flatnonzero(mask).tolist()
    queries = [prompt + t - 1 for t in range(132)]
    baseline_full = torch.from_numpy(np.load(previous / "s0h0_logits.npy"))
    baseline = baseline_full[[99, 131]]
    patches = []
    first = json.loads((root / "outputs/ownership_factorial_20260912/patches.json").read_text())
    with np.load(previous / "s1h0_factors.npz", allow_pickle=False) as donor:
        for layer in range(32):
            values = torch.from_numpy(donor[f"values_{layer}"])
            weights = torch.from_numpy(donor[f"attention_{layer}"])
            for row, (window, step) in enumerate(CASES["00012"].items()):
                spec = dict(
                    layer=layer,
                    queries=[prompt + step - 1],
                    factor="x",
                    attention=weights[:, [step]],
                    values=values,
                )
                full_logits, diagnostics = ownership.intervene_source_factors(
                    model, ids[:-1], queries, sources, [spec]
                )
                earlier_error = float((full_logits[:step] - baseline_full[:step]).abs().max())
                if earlier_error:
                    raise ValueError("single-layer patch changed earlier queries")
                logits = full_logits[[99, 131]]
                key = f"layer_{layer}_{window}"
                np.save(output / f"{key}_logits.npy", logits.numpy(), allow_pickle=False)
                record = dict(
                    condition=key,
                    layer=layer,
                    window=window,
                    **describe(logits[row], candidates, tokenizer),
                    **native(logits[row], tokenizer),
                    earlier_query_count=step,
                    earlier_query_max_logit_error=earlier_error,
                    diagnostics=diagnostics,
                    effect=adoption.distribution_effect(
                        baseline[row : row + 1], logits[row : row + 1], [int(ids[prompt + step])]
                    ),
                )
                record["margin_change"] = record["margin_14_minus_12"] - float(
                    baseline[row, fourteen] - baseline[row, twelve]
                )
                if layer in (7, 15, 23, 31):
                    old = next(
                        r
                        for r in first
                        if r.get("layer") == layer
                        and r.get("window") == window
                        and r.get("scope") == "query"
                        and r.get("factor") == "x"
                    )
                    if record["margin_14_minus_12"] != old["margin_14_minus_12"]:
                        raise ValueError("previous single-layer margin did not reproduce")
                patches.append(record)
            write_json(output / "layer_patches.json", patches)
    summary = dict(
        source_worlds=len(records),
        layer_patches=len(patches),
        source_main_effects=effects,
        future_history_errors=futures,
        source_reproduction_errors=source_reproduction,
        elapsed_seconds=time.monotonic() - started,
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(0),
        scientific_review="REVIEW_UNAVAILABLE",
    )
    if len(records) != 32 or len(patches) != 64:
        raise ValueError("incomplete declared matrix")
    write_json(output / "summary.json", summary)
    write_json(output / "manifest.json", {p.name: sha256(p) for p in sorted(output.iterdir())})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
