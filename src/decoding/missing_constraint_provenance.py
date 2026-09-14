"""O6 value provenance: does an inapplicable stage supply the generated value?

This is a one-source constructed intervention, not a detector evaluation.
"""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

VALUES = (12, 16, 18)


def replace_value(prompt, value):
    clause = "cook the onions in the beer mixture for 10 to 12 minutes."
    if value not in VALUES or prompt.count(clause) != 1:
        raise ValueError("expected one inspected duration clause and a preregistered value")
    return prompt.replace(clause, clause.replace("12 minutes", f"{value} minutes"))


def run(args):
    import datetime
    import numpy as np
    import torch
    import transformers
    from route_graph import ownership
    from decoding import missing_constraint, missing_constraint_routing
    from decoding.missing_constraint import sha256, write_json
    from decoding.missing_constraint_routing import query_groups
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = Path(__file__).resolve().parents[2]
    started = time.monotonic()
    input_path = args.inputs.resolve()
    parent = input_path.parent
    complete = json.loads((parent / "complete.json").read_text())
    if sha256(parent / "manifest.json") != complete["manifest_sha256"] or sha256(parent / "summary.json") != complete["summary_sha256"]:
        raise ValueError("upstream completion hash mismatch")
    manifest = json.loads((parent / "manifest.json").read_text())
    if sha256(input_path) != manifest["inputs.json"]:
        raise ValueError("upstream inputs hash mismatch")
    frozen = {c["id"]: c for c in json.loads(input_path.read_text())}
    model_path = root.parents[1] / "models/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    items = {}
    for world in ("before_only", "after12_aligned"):
        for value in VALUES:
            for point in ("uncommitted", "committed"):
                base = frozen[f"{world}__{point}"]
                text = replace_value(base["prompt"], value)
                pids = tokenizer.encode(text, add_special_tokens=False)
                ids = pids + tokenizer.encode(base["prefix"], add_special_tokens=False)
                changed = [i for i, (a, b) in enumerate(zip(base["input_ids"], ids)) if a != b]
                if len(ids) != len(base["input_ids"]) or len(pids) != base["prompt_length"]:
                    raise ValueError("numeric intervention changed token length")
                if value == 12:
                    if changed:
                        raise ValueError("identity numeric edit changed the baseline")
                elif len(changed) != 1 or not base["source_mask"][changed[0]] or changed[0] >= len(pids) or tokenizer.decode([ids[changed[0]]]) != str(value) or tokenizer.decode([base["input_ids"][changed[0]]]) != "12":
                    raise ValueError("numeric intervention is not exactly the declared source numeral")
                name = f"{world}__{value}__{point}"
                items[name] = {**base, "id": name, "prompt": text, "input_ids": ids, "source_value": value, "changed_source_positions": changed, "applicable_duration": [10, value] if world == "after12_aligned" else None}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    used = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    if int(used.strip().splitlines()[0]) >= 500 or shutil.disk_usage(output).free < 10 * 1024**3:
        raise RuntimeError("GPU busy or less than 10GiB free; preserve existing artifacts")
    code_paths = (Path(__file__), Path(ownership.__file__), Path(missing_constraint.__file__), Path(missing_constraint_routing.__file__), root / "scripts/run_missing_constraint_provenance.sh")
    settings = {
        "experiment": "O6_inapplicable_value_provenance", "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "independent_sources": 1, "constructed_truth": True, "values": list(VALUES),
        "free_generations": 6, "max_new_tokens": 64, "candidate_restriction": False,
        "patches": 21, "native_baselines": 3, "patch_endpoint": "already-committed prefix; tests value provenance, NOT refusal before commitment",
        "factors": ["x", "e", "xe"], "groups": ["last", "earlier", "all"],
        "donor": "known same-stage counterfactual source numeral 16 or 18", "receiver": "before_only12",
        "evaluation": "programmatic constructed source relations plus model-assisted semantic audit; no natural-GT or human-eval claim",
        "seed": 20260913, "model": str(model_path), "torch": torch.__version__, "transformers": transformers.__version__,
        "max_wall_seconds": 1800, "max_output_bytes": 8 * 1024**3,
        "input_sha256": {str(input_path): sha256(input_path)},
        "code_sha256": {str(p): sha256(p) for p in code_paths},
    }
    write_json(output / "settings.json", settings)
    write_json(output / "inputs.json", items)
    for p in code_paths:
        shutil.copyfile(p, output / ("executed_" + p.name))
    torch.manual_seed(20260913)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager").to("cuda:0").eval().requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}
    forwards = 0
    behavior, baselines, factors, patches = [], {}, {}, []

    def budget():
        if time.monotonic() - started > 1800 or torch.cuda.max_memory_allocated() >= 22 * 1024**3 or sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > 8 * 1024**3:
            raise RuntimeError("predeclared wall, memory, or output budget exceeded")

    def describe(logits):
        values, ids = logits.topk(8)
        return {"top_ids": ids.tolist(), "top_text": [tokenizer.decode([int(i)]) for i in ids], "top_logits": values.tolist()}

    with torch.inference_mode():
        for item in items.values():
            if item["point"] != "uncommitted":
                continue
            budget()
            tensor = torch.tensor([item["input_ids"]], device="cuda:0")
            first = model(tensor, use_cache=True, logits_to_keep=1)
            repeat = model(tensor, use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu()
            forwards += 2
            current, cache = first.logits[0, -1].float().cpu(), first.past_key_values
            error = float((current - repeat).abs().max())
            if error:
                raise ValueError("same-input first-logit mismatch")
            del first, repeat, tensor
            rows, generated, ended = [], [], False
            for step in range(64):
                rows.append(current.numpy().copy())
                token = int(current.argmax())
                generated.append(token)
                if token in eos:
                    ended = True
                    break
                if step < 63:
                    nxt = model(torch.tensor([[token]], device="cuda:0"), past_key_values=cache, use_cache=True, logits_to_keep=1)
                    forwards += 1
                    current, cache = nxt.logits[0, -1].float().cpu(), nxt.past_key_values
                    del nxt
            del cache
            np.save(output / f"{item['id']}_generation_logits.npy", np.stack(rows), allow_pickle=False)
            np.save(output / f"{item['id']}_generated_ids.npy", np.asarray(generated), allow_pickle=False)
            record = {"id": item["id"], "world": item["world"], "source_value": item["source_value"], "new_text": tokenizer.decode(generated, skip_special_tokens=True), "new_tokens": len(generated), "ended_eos": ended, "same_input_max_logit_error": error, "semantic_judgment": None}
            behavior.append(record)
            write_json(output / "behavior.json", behavior)
            print(json.dumps(record), flush=True)
        source = items["before_only__12__committed"]
        groups = query_groups(source["prompt_length"], len(source["input_ids"]))
        queries, sources = groups["all"], np.flatnonzero(source["source_mask"]).tolist()
        layers = list(range(model.config.num_hidden_layers))
        for value in VALUES:
            item = items[f"before_only__{value}__committed"]
            baselines[value], factors[value] = ownership.capture_source_factors(model, item["input_ids"], queries, sources, layers)
            forwards += 1
            np.save(output / f"baseline_{value}_all_queries.npy", baselines[value].numpy(), allow_pickle=False)
            np.savez(output / f"factors_{value}.npz", **{f"{key}_{layer}": tensor.numpy() for layer, group in factors[value].items() for key, tensor in group.items()})
        write_json(output / "baselines.json", {str(value): describe(logits[-1]) for value, logits in baselines.items()})
        plans = [(12, factor, "all") for factor in ("x", "e", "xe")]
        plans += [(value, factor, group) for value in (16, 18) for factor in ("x", "e", "xe") for group in ("last", "earlier", "all")]
        for value, factor, group in plans:
            budget()
            selected_queries = groups[group]
            index = [q - queries[0] for q in selected_queries]
            specs = [{"layer": layer, "factor": factor, "queries": selected_queries, "attention": factors[value][layer]["attention"][:, index], "values": factors[value][layer]["values"]} for layer in layers]
            measured, diagnostics = ownership.intervene_source_factors(model, source["input_ids"], queries, sources, specs)
            forwards += 1
            tag = f"donor{value}__{factor}__{group}"
            np.save(output / f"{tag}_all_queries.npy", measured.numpy(), allow_pickle=False)
            pre = min(selected_queries) - queries[0]
            prefix_error = float((measured[:pre] - baselines[12][:pre]).abs().max()) if pre else 0.0
            sham_error = float((measured - baselines[12]).abs().max()) if value == 12 else None
            if prefix_error or (value == 12 and sham_error):
                raise ValueError("nonzero prefix or sham error")
            record = {"id": tag, "donor_value": value, "factor": factor, "group": group, "sham": value == 12, **describe(measured[-1]), "all_query_sham_max_error": sham_error, "earlier_query_max_error": prefix_error, "diagnostics": diagnostics}
            patches.append(record)
            write_json(output / "patches.json", patches)
            print(json.dumps({k:v for k,v in record.items() if k != "diagnostics"}), flush=True)
    summary = {"generations": len(behavior), "patches": len(patches), "model_forwards": forwards, "elapsed_seconds": time.monotonic()-started, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "independent_sources": 1, "natural_detector_validated": False}
    write_json(output / "summary.json", summary)
    write_json(output / "manifest.json", {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    write_json(output / "complete.json", {"summary_sha256": sha256(output / "summary.json"), "manifest_sha256": sha256(output / "manifest.json")})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
