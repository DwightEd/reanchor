"""Exploratory distributed-layer and individual-query ownership interventions."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from route_graph import adoption, ownership
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import __version__ as transformers_version

from decoding import adoption_probe, ownership_factorial
from decoding.adoption_probe import sha256, write_json
from decoding.ownership_factorial import WINDOWS, changed_tokens, describe, load_trace


def native(logits, tokenizer):
    argmax = int(logits.argmax())
    return dict(
        argmax_id=argmax,
        argmax_text=tokenizer.decode([argmax]),
        max_tie_count=int((logits == logits.max()).sum()),
    )


def run(args):
    root = Path(__file__).resolve().parents[2]
    samples = root / "outputs/samples_20260911_145421_235"
    states = root / "outputs/states_samples_20260911_145421_235"
    previous = root / "outputs/ownership_factorial_20260912"
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    settings = json.loads((previous / "settings.json").read_text())
    ids, prompt, mask = load_trace(samples, states, "00012.npz")
    sources = np.flatnonzero(mask).tolist()
    steps = list(range(132))
    queries = [prompt + t - 1 for t in steps]
    groups = {"four": [7, 15, 23, 31], "all": list(range(32))}
    code = [
        Path(__file__),
        Path(ownership.__file__),
        Path(adoption.__file__),
        Path(ownership_factorial.__file__),
        Path(adoption_probe.__file__),
        root / "scripts/run_ownership_multilayer.sh",
    ]
    inputs = [
        samples / "00012.npz",
        samples / "00013.npz",
        samples / "samples.jsonl",
        samples / "settings.json",
        states / "00012.npz",
        states / "00013.npz",
        previous / "manifest.json",
        root.parent / "graph/docs/OWNERSHIP_MULTILAYER_PLAN_20260912.md",
    ]
    # Use fresh identity records; preserve original provenance in its own directory.
    settings.update(
        experiment="multilayer_post_factorial_exploratory",
        queries=queries,
        layers=groups,
        code_sha256={str(p.resolve()): sha256(p) for p in code},
        input_sha256={str(p.resolve()): sha256(p) for p in inputs},
        source_positions=sources,
        planned_conditions=98,
        raw_logit_policy="all 132 queries for worlds; two endpoints for patches; clean control",
        torch=torch.__version__,
        transformers=transformers_version,
        gpu=torch.cuda.get_device_name(0),
        model_files=[
            dict(name=p.name, size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
            for p in sorted(Path(settings["model"]).iterdir())
            if p.is_file()
        ],
    )
    for path in code:
        (output / f"executed_{path.name}").write_bytes(path.read_bytes())
    write_json(output / "settings.json", settings)
    torch.manual_seed(20260912)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(settings["model"], local_files_only=True)
    candidate_ids = [tokenizer.encode(t, add_special_tokens=False) for t in ("14", "12")]
    if any(len(tokens) != 1 for tokens in candidate_ids):
        raise ValueError("expected one token per predeclared candidate")
    fourteen, twelve = [tokens[0] for tokens in candidate_ids]
    candidates = (fourteen, twelve)
    model = (
        AutoModelForCausalLM.from_pretrained(
            settings["model"],
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .to("cuda:0")
        .eval()
        .requires_grad_(False)
    )
    worlds, factors = {}, {}
    for source in (0, 1):
        key = f"s{source}h0"
        changed = changed_tokens(ids, prompt, source, 0, twelve, fourteen)
        logits, values = ownership.capture_source_factors(
            model, changed[:-1], queries, sources, groups["all"]
        )
        worlds[key], factors[key] = logits, values
        np.save(output / f"{key}_logits.npy", logits.numpy(), allow_pickle=False)
        np.save(output / f"{key}_tokens.npy", changed, allow_pickle=False)
        np.savez(
            output / f"{key}_factors.npz",
            **{
                f"{name}_{layer}": tensor.numpy()
                for layer, data in values.items()
                for name, tensor in data.items()
            },
        )
    base = worlds["s0h0"]
    # Reproduction check against the preceding run, at every previously read query.
    old_steps = [t for start, stop in WINDOWS.values() for t in range(start, stop)]
    reproduction_errors = {
        key: float(
            (logits[old_steps] - torch.from_numpy(np.load(previous / f"{key}_logits.npy")))
            .abs()
            .max()
        )
        for key, logits in worlds.items()
    }
    if any(reproduction_errors.values()):
        raise ValueError(f"baseline protocol changed: {reproduction_errors}")
    endpoints = [stop - 1 for _, stop in WINDOWS.values()]
    records = []

    def condition(name, window, chosen, layers, factor, donor_key="s1h0"):
        specs = []
        for layer in layers:
            donor = factors[donor_key][layer]
            attention = donor["attention"][:, chosen].clone()
            specs.append(
                dict(
                    layer=layer,
                    queries=[queries[t] for t in chosen],
                    factor=factor,
                    source_indices=[sources.index(position) for position in (76, 341)],
                    attention=attention,
                    values=donor["values"],
                    mlp_update=donor["mlp_update"][chosen],
                )
            )
        logits, diagnostics = ownership.intervene_source_factors(
            model, ids[:-1], queries, sources, specs
        )
        before = min(chosen)
        error = float((logits[:before] - base[:before]).abs().max()) if before else None
        if error:
            raise ValueError(f"earlier query changed: {name}: {error}")
        is_sham = name.startswith("sham")
        sham_error = float((logits - base).abs().max()) if is_sham else None
        if sham_error:
            raise ValueError(f"nonzero sham: {name}: {sham_error}")
        record = dict(
            condition=name,
            window=window,
            layers=layers,
            factor=factor,
            prediction_steps=chosen,
            earlier_query_count=before,
            earlier_query_max_logit_error=error,
            sham_max_logit_error=sham_error,
            diagnostics=diagnostics,
            endpoints={
                w: dict(
                    **describe(logits[stop - 1], candidates, tokenizer),
                    **native(logits[stop - 1], tokenizer),
                    margin_change=float(
                        logits[stop - 1, fourteen]
                        - logits[stop - 1, twelve]
                        - base[stop - 1, fourteen]
                        + base[stop - 1, twelve]
                    ),
                )
                for w, (_, stop) in WINDOWS.items()
            },
            trajectory=adoption.distribution_effect(base, logits, ids[prompt : prompt + 132]),
            native_argmax_ids=logits.argmax(-1).tolist(),
            native_max_tie_counts=(logits == logits.max(-1, keepdim=True).values).sum(-1).tolist(),
        )
        np.save(output / f"logits_{name}.npy", logits[endpoints].numpy(), allow_pickle=False)
        records.append(record)
        write_json(output / "conditions.json", records)
        print(
            f"{name}: {[(w, d['margin_14_minus_12']) for w, d in record['endpoints'].items()]}",
            flush=True,
        )

    for group, layers in groups.items():
        for window, (start, stop) in WINDOWS.items():
            for scope, chosen in (
                ("query", [stop - 1]),
                ("window", list(range(start, stop))),
                ("history", list(range(stop))),
            ):
                for factor in ("x", "e", "xe", "mlp"):
                    condition(f"{group}_{window}_{scope}_{factor}", window, chosen, layers, factor)
        for factor in ("xe", "mlp"):
            condition(f"sham_{group}_{factor}", None, steps, layers, factor, "s0h0")
    for window, (start, stop) in WINDOWS.items():
        span = list(range(start, stop))
        for t in span:
            condition(f"only_{window}_{t}", window, [t], groups["all"], "x")
            condition(
                f"leave_{window}_{t}", window, [q for q in span if q != t], groups["all"], "x"
            )
        for scope, chosen in (
            ("query", [stop - 1]),
            ("window", span),
            ("history", list(range(stop))),
        ):
            condition(
                f"endpoint_{window}_{scope}", window, chosen, groups["all"], "endpoint_swap", "s0h0"
            )
    if len(records) != 98:
        raise ValueError("incomplete planned matrix")
    clean_ids, clean_prompt, clean_mask = load_trace(samples, states, "00013.npz")
    clean = []
    for source in (0, 1):
        changed = changed_tokens(clean_ids, clean_prompt, source, 0, twelve, fourteen)
        logits, _ = ownership.intervene_source_factors(
            model, changed[:-1], [clean_prompt + 125], np.flatnonzero(clean_mask), []
        )
        np.save(output / f"clean_s{source}_logits.npy", logits.numpy(), allow_pickle=False)
        clean.append(logits)
    write_json(
        output / "clean_control.json",
        dict(
            prediction_step=126,
            native=[native(logits[0], tokenizer) for logits in clean],
            effect=adoption.distribution_effect(
                clean[0], clean[1], [int(clean_ids[clean_prompt + 126])]
            ),
        ),
    )
    summary = dict(
        conditions=len(records),
        reproduction_errors=reproduction_errors,
        baseline={
            key: {
                window: {
                    **describe(logits[stop - 1], candidates, tokenizer),
                    **native(logits[stop - 1], tokenizer),
                }
                for window, (_, stop) in WINDOWS.items()
            }
            for key, logits in worlds.items()
        },
        elapsed_seconds=time.monotonic() - started,
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(0),
        scientific_review="REVIEW_UNAVAILABLE",
    )
    write_json(output / "summary.json", summary)
    write_json(output / "manifest.json", {p.name: sha256(p) for p in sorted(output.iterdir())})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())
