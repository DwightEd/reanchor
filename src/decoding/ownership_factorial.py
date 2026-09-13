"""Real-window source/history factorial and source node/edge interventions."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from route_graph import ownership
from route_graph.adoption import distribution_effect
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import __version__ as transformers_version

from decoding.adoption_probe import sha256, write_json

WINDOWS = {"grill": (93, 100), "onion": (119, 132)}
SOURCE_SWAP = (76, 341)
HISTORY_STEP = 99
LAYERS = (7, 15, 23, 31)


def changed_tokens(ids, prompt, source_world, history_world, twelve, fourteen):
    """Precisely declared edit; future history edit must not alter earlier outputs."""
    result = ids.copy()
    if result[SOURCE_SWAP[0]] != twelve or result[SOURCE_SWAP[1]] != fourteen:
        raise ValueError("source constraint identities differ from the inspected case")
    if source_world:
        result[list(SOURCE_SWAP)] = result[list(SOURCE_SWAP)][::-1]
    if history_world:
        if result[prompt + HISTORY_STEP] != fourteen:
            raise ValueError("earlier response constraint identity differs")
        result[prompt + HISTORY_STEP] = twelve
    return result


def describe(logits, candidates, tokenizer):
    fourteen, twelve = candidates
    top = logits.topk(5)
    margin = float(logits[fourteen] - logits[twelve])
    logp = logits.log_softmax(-1)
    return dict(
        margin_14_minus_12=margin,
        pair_preference_14=float(torch.softmax(logits[[fourteen, twelve]], dim=0)[0]),
        top_ids=top.indices.tolist(),
        top_text=[tokenizer.decode([token]) for token in top.indices.tolist()],
        top_logits=top.values.tolist(),
        entropy_nats=float(-(logp.exp() * logp).sum()),
    )


def load_trace(samples, states, name):
    with np.load(samples / name, allow_pickle=False) as saved:
        ids, prompt = saved["token_ids"].copy(), int(saved["prompt_length"])
    with np.load(states / name, allow_pickle=False) as saved:
        if not np.array_equal(saved["token_ids"], ids):
            raise ValueError("state token identity differs")
        mask = saved["source_mask"].copy()
    return ids, prompt, mask


def run(args):
    root = Path(__file__).resolve().parents[2]
    samples, states = [
        root / "outputs" / name
        for name in ("samples_20260911_145421_235", "states_samples_20260911_145421_235")
    ]
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    config = json.loads((samples / "settings.json").read_text())
    ids, prompt, mask = load_trace(samples, states, "00012.npz")
    sources = np.flatnonzero(mask).tolist()
    steps = [step for start, stop in WINDOWS.values() for step in range(start, stop)]
    queries = [prompt + step - 1 for step in steps]
    targets = {name: steps.index(stop - 1) for name, (_, stop) in WINDOWS.items()}
    layers = tuple(args.layers)
    if (
        not layers
        or len(set(layers)) != len(layers)
        or any(layer < 0 or layer > 31 for layer in layers)
    ):
        raise ValueError("invalid layer list")
    code = [Path(__file__), Path(ownership.__file__), Path(__file__).with_name("adoption_probe.py")]
    settings = dict(
        model=config["model"],
        layers=layers,
        windows=WINDOWS,
        source_swap_positions=SOURCE_SWAP,
        history_edit_step=HISTORY_STEP,
        queries=queries,
        source_positions=sources,
        source_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
        main_candidates=["14", "12"],
        source_mass_policy="preserve receiver source mass per query/head",
        scope="real observed windows plus explicit counterfactual inputs; no fitted detector",
        dtype="bfloat16",
        attention="eager",
        torch=torch.__version__,
        transformers=transformers_version,
        gpu=torch.cuda.get_device_name(0),
        seed=20260912,
        code_sha256={str(p.resolve()): sha256(p) for p in code},
        input_sha256={
            str(p.resolve()): sha256(p)
            for p in [
                samples / "00012.npz",
                samples / "00013.npz",
                samples / "samples.jsonl",
                samples / "settings.json",
            ]
        },
        model_files=[
            dict(name=p.name, size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
            for p in sorted(Path(config["model"]).iterdir())
            if p.is_file()
        ],
    )
    for path in code:
        (output / f"executed_{path.name}").write_bytes(path.read_bytes())
    write_json(output / "settings.json", settings)
    torch.manual_seed(20260912)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
    candidate_tokens = [tokenizer.encode(value, add_special_tokens=False) for value in ["14", "12"]]
    if any(len(tokens) != 1 for tokens in candidate_tokens):
        raise ValueError("predeclared values are not distinct single tokens")
    fourteen, twelve = [tokens[0] for tokens in candidate_tokens]
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
    world_logits, world_factors, records = {}, {}, []
    for source_world in range(2):
        for history_world in range(2):
            key = f"s{source_world}h{history_world}"
            changed = changed_tokens(ids, prompt, source_world, history_world, twelve, fourteen)
            logits, factors = ownership.capture_source_factors(
                model, changed[:-1], queries, sources, layers if history_world == 0 else ()
            )
            world_logits[key], world_factors[key] = logits, factors
            np.save(output / f"{key}_tokens.npy", changed, allow_pickle=False)
            np.save(output / f"{key}_logits.npy", logits.numpy(), allow_pickle=False)
            if factors:
                np.savez(
                    output / f"{key}_factors.npz",
                    **{
                        f"{name}_{layer}": tensor.numpy()
                        for layer, values in factors.items()
                        for name, tensor in values.items()
                    },
                )
            for name, row in targets.items():
                records.append(
                    dict(condition=key, window=name, **describe(logits[row], candidates, tokenizer))
                )
            print(
                f"WORLD {key}: {[(r['window'], r['margin_14_minus_12']) for r in records[-2:]]}",
                flush=True,
            )
    write_json(output / "worlds.json", records)
    base = world_logits["s0h0"]
    effects = {}
    for name, row in targets.items():
        margins = [
            [
                float(
                    world_logits[f"s{s}h{h}"][row, fourteen]
                    - world_logits[f"s{s}h{h}"][row, twelve]
                )
                for h in range(2)
            ]
            for s in range(2)
        ]
        effects[name] = ownership.factorial_effects(margins)
    early = steps.index(99)
    future_history_errors = {
        f"s{s}": float(
            (world_logits[f"s{s}h0"][: early + 1] - world_logits[f"s{s}h1"][: early + 1])
            .abs()
            .max()
        )
        for s in range(2)
    }
    if any(future_history_errors.values()):
        raise ValueError("future history edit changed earlier prediction")
    patches_output = []
    for layer in layers:
        for name, (start, stop) in WINDOWS.items():
            for scope in ("query", "window"):
                chosen_steps = [stop - 1] if scope == "query" else list(range(start, stop))
                indices = [steps.index(t) for t in chosen_steps]
                for factor in ("x", "e", "xe", "mlp"):
                    donor = world_factors["s1h0"][layer]
                    spec = dict(
                        layer=layer,
                        queries=[queries[i] for i in indices],
                        factor=factor,
                        attention=donor["attention"][:, indices],
                        values=donor["values"],
                        mlp_update=donor["mlp_update"][indices],
                    )
                    logits, diagnostics = ownership.intervene_source_factors(
                        model, ids[:-1], queries, sources, [spec]
                    )
                    row = targets[name]
                    item = dict(
                        layer=layer,
                        window=name,
                        scope=scope,
                        factor=factor,
                        **describe(logits[row], candidates, tokenizer),
                        diagnostics=diagnostics,
                    )
                    item["margin_change"] = item["margin_14_minus_12"] - float(
                        base[row, fourteen] - base[row, twelve]
                    )
                    before = [i for i, q in enumerate(queries) if q < min(spec["queries"])]
                    item["earlier_query_max_logit_error"] = (
                        0.0 if not before else float((logits[before] - base[before]).abs().max())
                    )
                    item["distribution_effect"] = distribution_effect(
                        base[row : row + 1], logits[row : row + 1], [int(ids[prompt + stop - 1])]
                    )
                    if item["earlier_query_max_logit_error"]:
                        raise ValueError("patch affected an earlier query")
                    patches_output.append(item)
        same = world_factors["s0h0"][layer]
        logits, diagnostics = ownership.intervene_source_factors(
            model,
            ids[:-1],
            queries,
            sources,
            [dict(layer=layer, queries=queries, factor="xe", **same)],
        )
        error = float((logits - base).abs().max())
        if error:
            raise ValueError(f"same-world sham has nonzero logit error {error}")
        patches_output.append(
            dict(layer=layer, factor="sham", max_logit_error=error, diagnostics=diagnostics)
        )
        write_json(output / "patches.json", patches_output)
        print(f"PATCH layer={layer} complete", flush=True)
    clean_ids, clean_prompt, clean_mask = load_trace(samples, states, "00013.npz")
    clean = []
    for source_world in range(2):
        changed = changed_tokens(clean_ids, clean_prompt, source_world, 0, twelve, fourteen)
        logits, _ = ownership.intervene_source_factors(
            model, changed[:-1], [clean_prompt + 125], np.flatnonzero(clean_mask), []
        )
        clean.append(logits)
    write_json(
        output / "correct_onion_control.json",
        dict(
            prediction_step=126,
            source_mask_sha256=hashlib.sha256(clean_mask.tobytes()).hexdigest(),
            baseline=describe(clean[0][0], candidates, tokenizer),
            swapped=describe(clean[1][0], candidates, tokenizer),
            effect=distribution_effect(clean[0], clean[1], [int(clean_ids[clean_prompt + 126])]),
        ),
    )
    summary = dict(
        factorial_effects=effects,
        future_history_errors=future_history_errors,
        patch_conditions=len(patches_output),
        elapsed_seconds=time.time() - started,
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(0),
        scientific_review="REVIEW_UNAVAILABLE",
    )
    write_json(output / "summary.json", summary)
    write_json(
        output / "manifest.json",
        {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=list(LAYERS))
    run(parser.parse_args())
