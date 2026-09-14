"""O6: inspect applicability before and after committing to a numeric duration.

Constructed source truth is metadata, never a detector score. No training and no
cross-world activation patching: the absence edits are not token-aligned.
"""

import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

OLD_SOURCE = (
    "1 Reduce heat to medium and cook another 10 to 12 minutes. 2  Remove the bratwurst "
    "from the beer mixture; reduce heat to low, and continue cooking the onions."
)
ACTION = "Remove the bratwurst from the grill and cook the onions"
GRAPH_WORLDS = ("original", "before_only", "after12_aligned")


def build_text_worlds(prompt, response):
    if prompt.count(OLD_SOURCE) != 1:
        raise ValueError("inspected source must occur exactly once")
    if response.count(ACTION) != 1:
        raise ValueError("inspected response action must occur exactly once")
    before = "Before removing the bratwurst from the grill, cook the onions in the beer mixture"
    after = "After removing the bratwurst from the grill, cook the onions in the beer mixture"
    worlds = {
        "original": (None, None),
        "both_12_14": (f"Timing rules: {before} for 10 to 12 minutes. {after} for 10 to 14 minutes.", [10, 14]),
        "both_14_12": (f"Timing rules: {before} for 10 to 14 minutes. {after} for 10 to 12 minutes.", [10, 12]),
        "before_only": (f"Timing rules: {before} for 10 to 12 minutes. {after}.", None),
        "after_only": (f"Timing rules: {before}. {after} for 10 to 14 minutes.", [10, 14]),
        "after12_aligned": (
            f"Timing rules: {before.replace('Before', 'After', 1)} for 10 to 12 minutes. "
            f"{after.replace('After', 'Before', 1)}.", [10, 12]
        ),
        "neither": (f"Timing rules: {before}. {after}.", None),
        "wrong_object": (
            f"Timing rules: {before} for 10 to 12 minutes. After removing the bratwurst from the grill, "
            "cook the bratwurst in the beer mixture for 10 to 14 minutes.", None
        ),
        "explicit_unspecified": (
            f"Timing rules: {before} for 10 to 12 minutes. {after}. "
            "The duration of the latter onion-cooking step is unspecified.", None
        ),
    }
    return {
        name: {
            "prompt": prompt if replacement is None else prompt.replace(OLD_SOURCE, replacement),
            "response": response,
            "applicable_duration": duration,
            "truth_provenance": "inspected natural source" if name == "original" else "program-constructed stage/object relation",
            "scope": "onions in beer mixture after removing bratwurst from grill",
            "pool_with_absence": name != "explicit_unspecified",
        }
        for name, (replacement, duration) in worlds.items()
    }


def build_prefixes(response):
    needles = {
        "action": "Remove the bratwurst from the grill and",
        "uncommitted": "Remove the bratwurst from the grill and cook the onions in the beer mixture",
        "committed": "Remove the bratwurst from the grill and cook the onions in the beer mixture for 10 to ",
    }
    result = {}
    for name, needle in needles.items():
        if response.count(needle) != 1:
            raise ValueError(f"ambiguous action prefix: {name}")
        end = response.index(needle) + len(needle)
        result[name] = {"text": response[:end], "response_char_end": end}
    return result


def current_action_text(combined):
    anchor = "Remove the bratwurst from the grill and"
    if anchor not in combined:
        raise ValueError("current action anchor is missing")
    return combined[combined.index(anchor):]


def surface_flags(text, *, truncated):
    mention = bool(re.search(r"\b\d+(?:\s*(?:to|-|–)\s*\d+)?\s*(?:minutes?|hours?|seconds?)\b", text, re.I))
    return {
        "numeric_duration_mention": mention,
        "hallucination_label": None,
        "absence_of_duration_confirmed": False,
        "requires_semantic_review": True,
        "truncated": bool(truncated),
        "note": "Surface cue only; attribution, negation, and current-action scope require semantic review.",
    }


def graph_bytes(n, layers, heads, hidden, kv_heads, head_dim):
    return 4 * ((3 * layers + 1) * n * hidden + layers * n * kv_heads * head_dim + layers * heads * n * n)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def prepare(root, tokenizer):
    import numpy as np

    path = root / "outputs/samples_20260911_145421_235/00012.npz"
    with np.load(path, allow_pickle=False) as saved:
        ids, prompt_length = saved["token_ids"], int(saved["prompt_length"])
    prompt = tokenizer.decode(ids[:prompt_length])
    response = tokenizer.decode(ids[prompt_length:])
    for text, expected in [(prompt, ids[:prompt_length]), (response, ids[prompt_length:])]:
        if not np.array_equal(tokenizer.encode(text, add_special_tokens=False), expected):
            raise ValueError("original text/token round-trip failed")
    conditions = []
    for world, item in build_text_worlds(prompt, response).items():
        pids = tokenizer.encode(item["prompt"], add_special_tokens=False)
        encoded_prompt = tokenizer(item["prompt"], add_special_tokens=False, return_offsets_mapping=True)
        left = item["prompt"].index("passage 1:") + len("passage 1:")
        right = item["prompt"].index("\n\nIn case the passages")
        mask = [a < right and b > left for a, b in encoded_prompt["offset_mapping"]]
        if not any(mask) or len(mask) != len(pids):
            raise ValueError("source mask does not cover passage tokens")
        for point, prefix in build_prefixes(response).items():
            rids = tokenizer.encode(prefix["text"], add_special_tokens=False)
            conditions.append({
                **item, "world": world, "point": point, "id": f"{world}__{point}",
                "prefix": prefix["text"], "input_ids": pids + rids,
                "prompt_length": len(pids), "response_prefix_tokens": len(rids),
                "query_index": len(pids) + len(rids) - 1, "source_mask": mask,
                "next_token_index": len(rids), "response_char_end": prefix["response_char_end"],
            })
    pairs = {c["id"]: c for c in conditions}
    for point in ("action", "uncommitted", "committed"):
        a, b = (pairs[f"{world}__{point}"] for world in ("before_only", "after12_aligned"))
        if len(a["input_ids"]) != len(b["input_ids"]) or a["prompt_length"] != b["prompt_length"]:
            raise ValueError("relation-only applicability control is not length-aligned")
        from collections import Counter
        changed = [i for i, (x, y) in enumerate(zip(a["input_ids"], b["input_ids"])) if x != y]
        if len(changed) != 2 or Counter(a["input_ids"]) != Counter(b["input_ids"]) or any(i >= a["prompt_length"] for i in changed):
            raise ValueError("applicability pair must differ in exactly two source relation tokens")
        a["relation_only_changed_positions"] = changed
        b["relation_only_changed_positions"] = changed
    return conditions, path


def run(args):
    import datetime
    import numpy as np
    import torch
    import transformers
    from route_graph import sample_graph
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    model_path = root.parents[1] / "models/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    conditions, original = prepare(root, tokenizer)
    if args.smoke:
        conditions = [c for c in conditions if c["world"] in ("original", "before_only") and c["point"] == "uncommitted"]
    settings = {
        "experiment": "O6_applicability_before_commitment", "phase": "behavior_and_predeclared_full_graphs",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "planned_conditions": len(conditions), "smoke": args.smoke,
        "model": str(model_path), "torch": torch.__version__, "transformers": transformers.__version__,
        "max_new_tokens": args.max_new_tokens, "seed": 20260913, "decoding": "greedy_full_vocabulary",
        "candidate_restriction": False, "labels_used_to_score_or_select": False, "trained_parameters": 0,
        "query_alignment": "last visible prefix token predicts next token; q=P+t-1",
        "generated_reference_used_as_truth": False,
        "semantic_labels": "pending independent review, surface duration cues are not truth",
        "capture_graphs": args.capture_graphs, "graph_worlds": list(GRAPH_WORLDS),
        "graph_point": "uncommitted", "max_output_gib": 16, "wall_budget_seconds": 1800,
        "source_id": "14375", "original_trace": "00012", "independent_sources": 1,
        "scope": "prefix-conditioned same-model continuations; constructed worlds not natural hallucination labels",
        "input_sha256": {str(original): sha256(original)},
        "code_sha256": {str(p): sha256(p) for p in [Path(__file__), Path(sample_graph.__file__)]},
        "model_files": [{"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in sorted(model_path.iterdir()) if p.is_file()],
    }
    write_json(output / "settings.json", settings)
    write_json(output / "inputs.json", conditions)
    shutil.copyfile(__file__, output / "executed_missing_constraint.py")
    shutil.copyfile(sample_graph.__file__, output / "executed_sample_graph.py")
    if args.prepare_only:
        write_json(output / "prepare_complete.json", {"conditions": len(conditions), "model_forwards": 0})
        print(json.dumps({"prepared": len(conditions), "output": str(output)}), flush=True)
        return
    if not torch.cuda.is_available() or torch.cuda.memory_allocated(0) > 500 * 1024**2:
        raise RuntimeError("CUDA is unavailable or this process already has an unexpected model")
    import subprocess
    used = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    if int(used.strip().splitlines()[0]) >= 500:
        raise RuntimeError("GPU is occupied; do not interrupt another job")
    if shutil.disk_usage(output).free < 20 * 1024**3:
        raise RuntimeError("less than 20GiB free; do not begin bounded capture")
    torch.manual_seed(20260913)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager").to("cuda:0").eval().requires_grad_(False)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    eos.add(tokenizer.eos_token_id)
    records, forwards = [], 0
    torch.cuda.reset_peak_memory_stats(0)
    with torch.inference_mode():
        for item in conditions:
            if time.monotonic() - started > 1800:
                raise RuntimeError("predeclared wall-time budget exceeded")
            dest = output / item["id"]
            dest.mkdir()
            inputs = torch.tensor([item["input_ids"]], device="cuda:0")
            first = model(inputs, use_cache=True, logits_to_keep=1)
            forwards += 1
            baseline = first.logits[0, -1].float().cpu()
            repeat = model(inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu()
            forwards += 1
            error = float((repeat - baseline).abs().max())
            if error != 0:
                raise ValueError(f"same-input first-logit mismatch: {error}")
            logits, generated = [], []
            current, cache = baseline, first.past_key_values
            del first, repeat
            ended = False
            for step in range(args.max_new_tokens):
                logits.append(current.numpy().copy())
                token = int(current.argmax())
                generated.append(token)
                if token in eos:
                    ended = True
                    break
                if step + 1 < args.max_new_tokens:
                    nxt = model(torch.tensor([[token]], device="cuda:0"), past_key_values=cache, use_cache=True, logits_to_keep=1)
                    forwards += 1
                    current, cache = nxt.logits[0, -1].float().cpu(), nxt.past_key_values
                    del nxt
            del cache, inputs
            np.save(dest / "logits.npy", np.stack(logits), allow_pickle=False)
            np.save(dest / "generated_ids.npy", np.asarray(generated, dtype=np.int64), allow_pickle=False)
            text = tokenizer.decode(generated, skip_special_tokens=True)
            combined = item["prefix"] + text
            action_text = current_action_text(combined)
            record = {
                "id": item["id"], "world": item["world"], "point": item["point"],
                "new_text": text, "new_tokens": len(generated), "ended_eos": ended,
                "first_token": tokenizer.decode([generated[0]]), "query_index": item["query_index"],
                "same_input_max_logit_error": error,
                "surface_flags": surface_flags(action_text, truncated=not ended),
                "applicable_duration": item["applicable_duration"], "semantic_judgment": None,
            }
            write_json(dest / "behavior.json", record)
            if args.capture_graphs and item["world"] in GRAPH_WORLDS and item["point"] == "uncommitted":
                all_ids = item["input_ids"] + generated
                estimate = graph_bytes(len(all_ids) - 1, model.config.num_hidden_layers, model.config.num_attention_heads, model.config.hidden_size, model.config.num_key_value_heads, model.config.hidden_size // model.config.num_attention_heads)
                used_bytes = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
                if used_bytes + estimate > 16 * 1024**3 or estimate > 24 * 1024**3:
                    raise RuntimeError("predeclared graph output/RAM budget exceeded; preserve existing artifacts")
                graph = sample_graph.capture_sample_graph(model, all_ids, item["prompt_length"], item["source_mask"], tokenizer.all_special_ids)
                forwards += 1
                first_index = item["response_prefix_tokens"]
                replay_top = graph["top_ids"][first_index:, 0]
                generated_top = np.asarray(generated)
                record["full_replay_cached_argmax_agreement"] = float(np.mean(replay_top == generated_top))
                record["graph_checks"] = sample_graph.save_graph(dest / "graph", graph, {
                    "condition": item["id"], "source_id": "14375", "original_prefix_tokens": item["response_prefix_tokens"],
                    "no_pruning": True, "capture": "complete fixed-continuation replay; not cached generation states",
                    "cached_vs_full_argmax_agreement": record["full_replay_cached_argmax_agreement"],
                })
                del graph
                write_json(dest / "behavior.json", record)
            peak = torch.cuda.max_memory_allocated(0)
            record["peak_cuda_allocated_bytes"] = peak
            records.append(record)
            write_json(output / "progress.json", {"completed": len(records), "planned": len(conditions), "forwards": forwards, "last": record["id"]})
            print(json.dumps({"completed": len(records), "planned": len(conditions), "id": record["id"], "first": record["first_token"], "ended_eos": ended, "peak_gib": peak / 1024**3}), flush=True)
            if peak >= 22 * 1024**3:
                raise RuntimeError("22GiB peak-memory gate exceeded")
    summary = {"status": "complete", "conditions": len(records), "model_forwards": forwards, "full_graphs": sum("graph_checks" in r for r in records), "elapsed_seconds": time.monotonic() - started, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(0), "semantic_accuracy": None, "independent_sources": 1, "records": records}
    write_json(output / "summary.json", summary)
    write_json(output / "manifest.json", {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()})
    write_json(output / "complete.json", {"status": "complete", "conditions": len(records), "summary_sha256": sha256(output / "summary.json"), "manifest_sha256": sha256(output / "manifest.json")})
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--capture-graphs", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 64:
        parser.error("max-new-tokens must be 1..64")
    run(args)


if __name__ == "__main__":
    main()
