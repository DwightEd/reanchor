"""Real-prefix stage relation test, frozen readouts, and exhaustive query X scan."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from route_graph import adoption, ownership, relation_readout
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import __version__ as transformers_version

from decoding import adoption_probe, ownership_factorial, ownership_multilayer, relation_inputs
from decoding.adoption_probe import sha256, write_json
from decoding.ownership_factorial import describe, load_trace
from decoding.ownership_multilayer import native

ROOT = Path(__file__).resolve().parents[2]


def describe_row(logits, baseline, candidates, saved_token, tokenizer):
    return dict(
        **describe(logits, candidates, tokenizer),
        **native(logits, tokenizer),
        effect=adoption.distribution_effect(baseline[None], logits[None], [saved_token]),
        margin_change=float(
            (logits[candidates[0]] - logits[candidates[1]])
            - (baseline[candidates[0]] - baseline[candidates[1]])
        ),
    )


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    samples = ROOT / "outputs/samples_20260911_145421_235"
    states = ROOT / "outputs/states_samples_20260911_145421_235"
    config = json.loads((samples / "settings.json").read_text())
    code = [Path(__file__), ROOT / "scripts/run_relation_only.sh"] + [
        Path(module.__file__)
        for module in (
            ownership,
            relation_readout,
            relation_inputs,
            adoption,
            adoption_probe,
            ownership_factorial,
            ownership_multilayer,
        )
    ]
    inputs = [
        samples / "00012.npz",
        states / "00012.npz",
        samples / "settings.json",
        ROOT.parent / "graph/docs/RELATION_ONLY_PLAN_20260913.md",
    ]
    settings = dict(
        experiment="relation_only_stage_ownership",
        source_id="14375",
        natural_trace="00012",
        seed=20260913,
        model=config["model"],
        dtype="bfloat16",
        attention="eager",
        torch=torch.__version__,
        transformers=transformers_version,
        gpu=torch.cuda.get_device_name(0),
        conditions="four worlds; 28 patches; all prior queries X",
        scope="constructed stage rules with natural and constructed prefixes; not natural labels",
        scientific_review="REVIEW_UNAVAILABLE",
        trained_parameters=0,
        code_sha256={str(p.resolve()): sha256(p) for p in code},
        input_sha256={str(p.resolve()): sha256(p) for p in inputs},
        model_files=[
            dict(name=p.name, size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
            for p in sorted(Path(config["model"]).iterdir())
            if p.is_file()
        ],
    )
    write_json(output / "settings.json", settings)
    for p in code:
        (output / f"executed_{p.name}").write_bytes(p.read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
    original_ids, original_prompt, original_mask = load_trace(samples, states, "00012.npz")
    worlds = relation_inputs.build_worlds(tokenizer, original_ids, original_prompt)
    write_json(output / "inputs.json", {f"{c}_{w}": r for (c, w), r in worlds.items()})
    encoded = [tokenizer.encode(x, add_special_tokens=False) for x in ("14", "12")]
    if any(len(x) != 1 for x in encoded):
        raise ValueError("candidates must be single tokens")
    candidates = tuple(x[0] for x in encoded)
    torch.manual_seed(settings["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
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
    logits, _ = ownership.intervene_source_factors(
        model,
        original_ids[:-1],
        [original_prompt + 98, original_prompt + 130],
        np.flatnonzero(original_mask),
        [],
    )
    np.save(output / "natural_reference_logits.npy", logits.numpy(), allow_pickle=False)
    write_json(
        output / "natural_reference.json",
        {
            name: {**describe(logits[i], candidates, tokenizer), **native(logits[i], tokenizer)}
            for i, name in enumerate(("grill", "onion"))
        },
    )
    base, factors, world_records, patches = {}, {}, [], []
    for (context, world), item in worlds.items():
        key = f"{context}_{world}"
        prompt, step = item["prompt_length"], item["response_step"]
        queries = list(range(prompt - 1, prompt + step))
        measured, captured, readouts = relation_readout.capture_relation(
            model,
            item["token_ids"][:-1],
            queries,
            item["source_positions"],
            item["numeric_positions"],
        )
        base[context, world], factors[context, world] = measured, captured
        np.save(output / f"{key}_logits.npy", measured.numpy(), allow_pickle=False)
        np.savez(
            output / f"{key}_factors.npz",
            **{
                f"{name}_{layer}": value.numpy()
                for layer, group in captured.items()
                for name, value in group.items()
            },
        )
        np.savez(output / f"{key}_readouts.npz", **readouts)
        ranks = {}
        for name, score in readouts.items():
            values = score[:, -1].mean(0)
            ranks[name] = dict(
                values_12_14=values.tolist(),
                choice="tie" if values[0] == values[1] else ("12", "14")[int(values.argmax())],
            )
        row = dict(
            context=context,
            world=world,
            expected_value=item["expected_value"],
            constructed_source=True,
            natural_prefix=context == "real_after",
            **describe(measured[-1], candidates, tokenizer),
            **native(measured[-1], tokenizer),
            readouts=ranks,
        )
        world_records.append(row)
        write_json(output / "worlds.json", world_records)
        print(
            f"WORLD {key} margin={row['margin_14_minus_12']} native={row['argmax_text']}",
            flush=True,
        )

    write_json(
        output / "world_effects.json",
        {
            context: adoption.distribution_effect(
                base[context, 0][-1:], base[context, 1][-1:], [candidates[1]]
            )
            for context in ("real_after", "contrast_before")
        },
    )

    def patch(context, world, factor, steps, tag, same_world=False):
        item = worlds[context, world]
        prompt, step = item["prompt_length"], item["response_step"]
        donor = factors[context, world if same_world else 1 - world]
        specs = []
        for layer, group in donor.items():
            spec = dict(layer=layer, factor=factor, queries=[prompt + t - 1 for t in steps])
            if factor == "endpoint_swap":
                spec["source_indices"] = [
                    item["source_positions"].index(p) for p in item["numeric_positions"]
                ]
            elif factor == "mlp":
                spec["mlp_update"] = group["mlp_update"][steps]
            else:
                spec.update(attention=group["attention"][:, steps], values=group["values"])
            specs.append(spec)
        measured, diagnostics = ownership.intervene_source_factors(
            model,
            item["token_ids"][:-1],
            list(range(prompt - 1, prompt + step)),
            item["source_positions"],
            specs,
        )
        reference = base[context, world]
        first = min(steps)
        error = float((measured[:first] - reference[:first]).abs().max()) if first else 0.0
        if error or (same_world and not torch.equal(measured, reference)):
            raise ValueError("nonzero earlier-query or same-world sham error")
        key = f"{context}_{world}_{tag}"
        np.save(output / f"{key}_endpoint.npy", measured[-1].numpy(), allow_pickle=False)
        row = dict(
            condition=key,
            context=context,
            world=world,
            factor=factor,
            steps=steps,
            sham=same_world,
            earlier_query_count=first,
            earlier_query_max_logit_error=error,
            **describe_row(
                measured[-1], reference[-1], candidates, item["token_ids"][prompt + step], tokenizer
            ),
            diagnostics=diagnostics,
        )
        if same_world:
            row["all_query_max_logit_error"] = float((measured - reference).abs().max())
        patches.append(row)
        write_json(output / "patches.json", patches)
        print(f"PATCH {key} margin={row['margin_14_minus_12']}", flush=True)

    for context, world in worlds:
        step = worlds[context, world]["response_step"]
        patch(context, world, "x", [step], "sham", same_world=True)
        for factor in ("x", "e", "xe", "mlp", "endpoint_swap"):
            patch(context, world, factor, [step], factor)
        patch(context, world, "x", list(range(step + 1)), "all_queries_x")
    item = worlds["real_after", 0]
    for step in range(item["response_step"] + 1):
        patch("real_after", 0, "x", [step], f"scan_{step}")
    source_mass = np.stack(
        [g["attention"].sum(-1).mean(0).numpy() for g in factors["real_after", 0].values()]
    ).mean(0)
    source_rise = np.maximum(np.diff(source_mass, prepend=source_mass[0]), 0)
    top8 = np.argsort(-source_rise, kind="stable")[:8].tolist()
    write_json(
        output / "candidate_queries.json",
        dict(
            rule="positive increase in mean all-layer/head source attention; first score zero",
            scores=source_rise.tolist(),
            top8=top8,
            endpoint_known=True,
        ),
    )
    summary = dict(
        worlds=len(world_records),
        patches=len(patches),
        scan_queries=item["response_step"] + 1,
        elapsed_seconds=time.monotonic() - started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
        scientific_review="REVIEW_UNAVAILABLE",
    )
    write_json(output / "summary.json", summary)
    write_json(
        output / "manifest.json",
        {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    )
    (output / "COMPLETE").write_text(
        "capture completed; scientific claims require separate analysis\n"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
