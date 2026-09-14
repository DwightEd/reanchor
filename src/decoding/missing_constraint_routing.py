"""O6 follow-up: separate source value, routing, and historical mediation.

Uses the predeclared, two-relation-token aligned pair only. These interventions
test native model computation, not a detector or a natural truth classifier.
"""

import argparse
import json
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path


def query_groups(prompt_length, input_length):
    if not 0 < prompt_length < input_length:
        raise ValueError("a nonempty response prefix is required")
    queries = list(range(prompt_length - 1, input_length))
    return {"last": queries[-1:], "earlier": queries[:-1], "all": queries}


def validate_pair(a, b):
    if a["prompt_length"] != b["prompt_length"] or a["source_mask"] != b["source_mask"]:
        raise ValueError("prompt/source partitions differ")
    x, y = a["input_ids"], b["input_ids"]
    changed = [i for i, (u, v) in enumerate(zip(x, y)) if u != v]
    if len(x) != len(y) or Counter(x) != Counter(y) or len(changed) != 2:
        raise ValueError("pair is not a two-token permutation")
    if any(i >= a["prompt_length"] or not a["source_mask"][i] for i in changed):
        raise ValueError("only source relation tokens may differ")
    return changed


def validate_relation_tokens(a, b, decode):
    changed = validate_pair(a, b)
    left = [decode([a["input_ids"][i]]).strip() for i in changed]
    right = [decode([b["input_ids"][i]]).strip() for i in changed]
    if set(left) != {"Before", "After"} or right != left[::-1]:
        raise ValueError("changed tokens are not the declared Before/After relation swap")
    return changed


def run(args):
    import datetime
    import numpy as np
    import torch
    import transformers
    from route_graph import ownership
    from decoding.missing_constraint import sha256, write_json
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = Path(__file__).resolve().parents[2]
    inputs_path = args.inputs.resolve()
    input_root = inputs_path.parent
    if not (input_root / "complete.json").is_file():
        raise ValueError("O6 main run must finish before the routing follow-up")
    complete = json.loads((input_root / "complete.json").read_text())
    if sha256(input_root / "summary.json") != complete["summary_sha256"]:
        raise ValueError("O6 summary hash mismatch")
    if sha256(input_root / "manifest.json") != complete["manifest_sha256"]:
        raise ValueError("O6 manifest hash mismatch")
    manifest = json.loads((input_root / "manifest.json").read_text())
    if sha256(inputs_path) != manifest["inputs.json"]:
        raise ValueError("O6 input hash mismatch")
    all_inputs = json.loads(inputs_path.read_text())
    names = ("before_only", "after12_aligned")
    selected = {name: next(c for c in all_inputs if c["world"] == name and c["point"] == "uncommitted") for name in names}
    changed = validate_pair(*(selected[name] for name in names))
    groups = query_groups(selected[names[0]]["prompt_length"], len(selected[names[0]]["input_ids"]))
    queries = groups["all"]
    sources = np.flatnonzero(selected[names[0]]["source_mask"]).tolist()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(output).free < 10 * 1024**3:
        raise RuntimeError("less than 10GiB free")
    used = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    if int(used.strip().splitlines()[0]) >= 500:
        raise RuntimeError("GPU occupied; do not interrupt another job")
    model_path = root.parents[1] / "models/Meta-Llama-3.1-8B-Instruct"
    settings = {
        "experiment": "O6_aligned_applicability_routing", "independent_sources": 1,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "constructed_truth": True, "natural_hallucination_labels": False,
        "worlds": list(names), "point": "uncommitted", "changed_source_tokens": changed,
        "query_groups": groups, "source_positions": sources,
        "factors": ["x", "e", "xe", "mlp"], "planned_patches": 32,
        "E_definition": "donor source attention distribution; receiver source mass and current V retained",
        "X_definition": "donor high-dimensional source V; current attention retained",
        "MLP_definition": "entire donor query MLP update; NOT source-only and NOT source-specific attribution",
        "donor_is_oracle_counterfactual": True, "candidate_restriction": False,
        "endpoint": "full next-token vocabulary before duration commitment; no forced number",
        "limitations": "paired-intervention sufficiency, not necessity or automatic constraint recognition",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "model": str(model_path), "seed": 20260913,
        "max_wall_seconds": 1800, "max_output_bytes": 8 * 1024**3,
        "input_sha256": {str(inputs_path): sha256(inputs_path)},
        "code_sha256": {str(p): sha256(p) for p in (Path(__file__), Path(ownership.__file__))},
    }
    write_json(output / "settings.json", settings)
    write_json(output / "inputs.json", selected)
    for p in (Path(__file__), Path(ownership.__file__)):
        shutil.copyfile(p, output / ("executed_" + p.name))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    validate_relation_tokens(*(selected[name] for name in names), tokenizer.decode)
    torch.manual_seed(20260913)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager").to("cuda:0").eval().requires_grad_(False)
    layers = list(range(model.config.num_hidden_layers))
    baseline, factors, records = {}, {}, []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    def describe(logits):
        values, ids = logits.topk(8)
        return {"top_ids": ids.tolist(), "top_text": [tokenizer.decode([int(i)]) for i in ids], "top_logits": values.tolist()}

    def js(a, b):
        p, q = a.double().softmax(-1), b.double().softmax(-1)
        m = (p + q) / 2
        return float((torch.special.xlogy(p, p / m).sum() + torch.special.xlogy(q, q / m).sum()) / 2)

    with torch.inference_mode():
        for name in names:
            baseline[name], factors[name] = ownership.capture_source_factors(model, selected[name]["input_ids"], queries, sources, layers)
            np.save(output / f"{name}_baseline_all_queries.npy", baseline[name].numpy(), allow_pickle=False)
            np.savez(output / f"{name}_factors.npz", **{f"{key}_{layer}": tensor.numpy() for layer, group in factors[name].items() for key, tensor in group.items()})
        write_json(output / "baselines.json", {name: describe(baseline[name][-1]) for name in names})
        for name in names:
            other = names[1] if name == names[0] else names[0]
            plans = [(factor, "all", True) for factor in ("x", "e", "xe", "mlp")]
            plans += [(factor, group, False) for factor in ("x", "e", "xe", "mlp") for group in ("last", "earlier", "all")]
            for factor, group, sham in plans:
                if time.monotonic() - started > 1800:
                    raise RuntimeError("wall budget exceeded")
                selected_queries = groups[group]
                index = [q - queries[0] for q in selected_queries]
                donor = factors[name if sham else other]
                patches = []
                for layer in layers:
                    spec = {"layer": layer, "queries": selected_queries, "factor": factor}
                    if factor == "mlp":
                        spec["mlp_update"] = donor[layer]["mlp_update"][index]
                    else:
                        spec.update(attention=donor[layer]["attention"][:, index], values=donor[layer]["values"])
                    patches.append(spec)
                measured, diagnostics = ownership.intervene_source_factors(model, selected[name]["input_ids"], queries, sources, patches)
                tag = f"{name}__{factor}__{group}__{'sham' if sham else 'donor'}"
                # Persist the complete query/vocabulary trace, including exact sham checks.
                np.save(output / f"{tag}.npy", measured.numpy(), allow_pickle=False)
                pre = min(selected_queries) - queries[0]
                prefix_error = float((measured[:pre] - baseline[name][:pre]).abs().max()) if pre else 0.0
                sham_error = float((measured - baseline[name]).abs().max()) if sham else None
                if prefix_error != 0 or (sham and sham_error != 0):
                    raise ValueError(f"nonzero pre-patch or sham error: {tag}, {prefix_error}, {sham_error}")
                row = {"id": tag, "receiver": name, "donor": name if sham else other, "factor": factor, "group": group, "sham": sham, **describe(measured[-1]), "baseline_js_to_other": js(baseline[name][-1], baseline[other][-1]), "js_to_receiver": js(measured[-1], baseline[name][-1]), "js_to_donor": js(measured[-1], baseline[name if sham else other][-1]), "js_to_opposite_world": js(measured[-1], baseline[other][-1]), "all_query_sham_max_error": sham_error, "earlier_query_max_error": prefix_error, "diagnostics": diagnostics}
                records.append(row)
                write_json(output / "patches.json", records)
                write_json(output / "progress.json", {"completed": len(records), "planned": 32, "last": tag})
                print(json.dumps({"completed": len(records), "id": tag, "top": row["top_text"][0], "js_receiver": row["js_to_receiver"], "js_donor": row["js_to_donor"]}), flush=True)
                if torch.cuda.max_memory_allocated() >= 22 * 1024**3 or sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > 8 * 1024**3:
                    raise RuntimeError("memory or output budget exceeded")
    summary = {"patches": len(records), "model_forwards": len(records) + 2, "independent_sources": 1, "elapsed_seconds": time.monotonic() - started, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "semantic_detector_validated": False}
    write_json(output / "summary.json", summary)
    write_json(output / "manifest.json", {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    write_json(output / "complete.json", {"patches": len(records), "summary_sha256": sha256(output / "summary.json"), "manifest_sha256": sha256(output / "manifest.json")})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
