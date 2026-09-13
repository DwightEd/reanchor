"""E-only query and layer scans using the frozen O4 relation worlds."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from route_graph import ownership
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import __version__ as transformers_version

from decoding import relation_only
from decoding.adoption_probe import sha256, write_json
from decoding.relation_only import describe_row

ROOT = Path(__file__).resolve().parents[2]


def run(args):
    started = time.monotonic()
    previous = ROOT / "outputs/relation_only_20260913"
    manifest = json.loads((previous / "manifest.json").read_text())
    names = [
        "inputs.json",
        "settings.json",
        "real_after_0_logits.npy",
        "patches.json",
        "real_after_0_factors.npz",
        "real_after_1_factors.npz",
        "real_after_0_e_endpoint.npy",
    ]
    for name in names:
        if sha256(previous / name) != manifest[name]:
            raise ValueError(f"O4 input changed: {name}")
    settings = json.loads((previous / "settings.json").read_text())
    for name, expected in settings["code_sha256"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"O4 dependency code changed: {name}")
    model_files = [
        dict(name=p.name, size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
        for p in sorted(Path(settings["model"]).iterdir())
        if p.is_file()
    ]
    if model_files != settings["model_files"]:
        raise ValueError("O4 model file identity changed")
    item = json.loads((previous / "inputs.json").read_text())["real_after_0"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = [
        Path(__file__),
        Path(ownership.__file__),
        Path(relation_only.__file__),
        ROOT / "scripts/run_relation_routing.sh",
        ROOT.parent / "graph/docs/RELATION_ROUTING_PLAN_20260913.md",
    ]
    execution = dict(
        experiment="relation_routing_post_O4",
        model=settings["model"],
        seed=settings["seed"],
        dtype="bfloat16",
        attention="eager",
        previous_manifest_sha256=sha256(previous / "manifest.json"),
        inputs={name: manifest[name] for name in names},
        code_sha256={str(p.resolve()): sha256(p) for p in files},
        scientific_review="REVIEW_UNAVAILABLE",
        planned_conditions=167,
        torch=torch.__version__,
        transformers=transformers_version,
        gpu=torch.cuda.get_device_name(0),
        model_files=model_files,
        inherited_code_sha256=settings["code_sha256"],
    )
    write_json(output / "settings.json", execution)
    for path in files:
        (output / f"executed_{path.name}").write_bytes(path.read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(settings["model"], local_files_only=True)
    encoded = [tokenizer.encode(x, add_special_tokens=False) for x in ("14", "12")]
    if any(len(x) != 1 for x in encoded):
        raise ValueError("candidate token identity changed")
    candidates = tuple(x[0] for x in encoded)
    torch.manual_seed(settings["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
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
    prompt, endpoint = item["prompt_length"], item["response_step"]
    queries = list(range(prompt - 1, prompt + endpoint))
    baseline = torch.from_numpy(np.load(previous / "real_after_0_logits.npy"))
    current, _ = ownership.intervene_source_factors(
        model,
        item["token_ids"][:-1],
        queries,
        item["source_positions"],
        [],
    )
    if not torch.equal(current, baseline):
        raise ValueError("O4 complete baseline does not reproduce exactly")
    factors = {}
    for world in (0, 1):
        with np.load(previous / f"real_after_{world}_factors.npz") as saved:
            factors[world] = {
                layer: {
                    name: torch.from_numpy(saved[f"{name}_{layer}"].copy())
                    for name in ("attention", "values")
                }
                for layer in range(32)
            }
    all_layers = list(range(32))
    conditions = [(f"query_{t}", [t], all_layers, 1) for t in range(endpoint + 1)]
    conditions += [(f"layer_{layer}", [endpoint], [layer], 1) for layer in all_layers]
    conditions += [
        ("all_queries", list(range(endpoint + 1)), all_layers, 1),
        ("without_final", list(range(endpoint)), all_layers, 1),
        ("sham", [endpoint], all_layers, 0),
    ]
    old_patch = next(
        r
        for r in json.loads((previous / "patches.json").read_text())
        if r["condition"] == "real_after_0_e"
    )
    old_endpoint = torch.from_numpy(np.load(previous / "real_after_0_e_endpoint.npy"))
    if sha256(previous / "real_after_0_e_endpoint.npy") != manifest["real_after_0_e_endpoint.npy"]:
        raise ValueError("O4 final E endpoint changed")
    rows = []
    for name, steps, layers, donor in conditions:
        specs = [
            dict(
                layer=layer,
                factor="e",
                queries=[prompt + t - 1 for t in steps],
                values=factors[donor][layer]["values"],
                attention=factors[donor][layer]["attention"][:, steps],
            )
            for layer in layers
        ]
        logits, diagnostics = ownership.intervene_source_factors(
            model,
            item["token_ids"][:-1],
            queries,
            item["source_positions"],
            specs,
        )
        first = min(steps)
        error = float((logits[:first] - baseline[:first]).abs().max()) if first else 0.0
        if error or (name == "sham" and not torch.equal(logits, baseline)):
            raise ValueError("nonzero preceding-query or same-world error")
        if name == f"query_{endpoint}" and not torch.equal(logits[-1], old_endpoint):
            raise ValueError("O4 final-query E did not reproduce")
        np.save(output / f"{name}_logits.npy", logits[-1].numpy(), allow_pickle=False)
        row = dict(
            condition=name,
            steps=steps,
            layers=layers,
            donor_world=donor,
            earlier_query_count=first,
            earlier_query_max_logit_error=error,
            **describe_row(
                logits[-1],
                baseline[-1],
                candidates,
                item["token_ids"][prompt + endpoint],
                tokenizer,
            ),
            diagnostics=diagnostics,
        )
        if name == "sham":
            row["all_query_max_logit_error"] = float((logits - baseline).abs().max())
        rows.append(row)
        write_json(output / "patches.json", rows)
        print(
            f"E {name} margin={row['margin_14_minus_12']} native={row['argmax_text']}", flush=True
        )
    write_json(
        output / "summary.json",
        dict(
            conditions=len(rows),
            baseline_max_error=0.0,
            reproduced_O4_margin=old_patch["margin_14_minus_12"],
            elapsed_seconds=time.monotonic() - started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
        ),
    )
    write_json(
        output / "manifest.json",
        {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    )
    (output / "COMPLETE").write_text("capture complete; scoped interpretation remains separate\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
