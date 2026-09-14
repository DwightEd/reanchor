"""R04: freeze source-disjoint observer inputs before inspecting annotations."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

TASKS = ("QA", "Summary", "Data2txt")
GENERATORS = ("llama-2-7b-chat", "llama-2-13b-chat")
SEED = 20260913
OBSERVER = "Meta-Llama-3.1-8B-Instruct"
EXPECTED_EXCLUSIONS = {"11951", "13514", "13717", "14637", "15220", "15475",
                       "14304", "14315", "14325", "14375"}


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_response_metadata(path):
    # Parse JSON rows, but never retain or pass annotations/content to selection.
    with Path(path).open(encoding="utf-8") as stream:
        return [{k: row[k] for k in ("id", "source_id", "model", "split")}
                for line in stream if line.strip() for row in [json.loads(line)]]


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def select_sources(sources, responses, excluded, generators, task_counts, seed, batch=0):
    """Use only source content/identity and response identity/model/split metadata."""
    from decoding.ragtruth_population_io import source_interval

    if (set(task_counts) != set(TASKS) or any(n < 2 or n % 2 for n in task_counts.values())
            or len(generators) != 2 or len(set(generators)) != 2 or batch not in (0, 1)):
        raise ValueError("require even task quotas, two distinct generators and batch 0/1")
    source_map = {}
    for source in sources:
        sid = str(source["source_id"])
        if sid in source_map or source["task_type"] not in TASKS:
            raise ValueError("duplicate source or unsupported task")
        span = source_interval(source)
        source_map[sid] = {"source_id": sid, "task": source["task_type"],
                           "source_text_sha256": digest(source["prompt"][slice(*span)]),
                           "prompt_sha256": digest(source["prompt"]), "source_span": span}
    seen, pairs, splits = set(), {}, defaultdict(set)
    for row in responses:
        rid, sid = str(row["id"]), str(row["source_id"])
        pair = sid, row["model"]
        if rid in seen or pair in pairs or sid not in source_map:
            raise ValueError("duplicate response identity/model pair or missing source")
        if row["split"] not in ("train", "test"):
            raise ValueError("unknown official split")
        seen.add(rid)
        pairs[pair] = rid
        splits[sid].add(row["split"])
    if any(len(value) != 1 for value in splits.values()):
        raise ValueError("source crosses official split")
    unknown = set(excluded) - source_map.keys()
    if unknown:
        raise ValueError(f"excluded source IDs absent from dataset: {sorted(unknown)}")
    excluded_hashes = {source_map[sid]["source_text_sha256"] for sid in excluded}
    # Exact duplicate evidence cannot enter a different source/split. Evidence
    # found in official-test is not eligible for this official-train roster.
    test_hashes = {source_map[sid]["source_text_sha256"] for sid, split in splits.items()
                   if split != {"train"}}
    candidates = []
    for sid, source in source_map.items():
        if splits.get(sid) != {"train"} or source["source_text_sha256"] in excluded_hashes | test_hashes:
            continue
        key = f"{seed}|{sid}|{source['source_text_sha256']}"
        candidates.append({**source, "official_split": "train",
                           "selection_sha256": digest("R04-v1|select|" + key),
                           "split_sha256": digest("R04-v1|split|" + key)})
    unique = {}
    for source in sorted(candidates, key=lambda s: s["source_id"]):
        unique.setdefault(source["source_text_sha256"], source)
    selected, census = [], {}
    for task in TASKS:
        ordered = sorted((s for s in unique.values() if s["task"] == task),
                         key=lambda s: (s["selection_sha256"], s["source_id"]))
        count = task_counts[task]
        if len(ordered) < (batch + 1) * count:
            raise ValueError(f"not enough eligible unique sources for {task}, batch {batch}")
        chosen = ordered[batch * count:(batch + 1) * count]
        split_order = sorted(chosen, key=lambda s: (s["split_sha256"], s["source_id"]))
        development = {s["source_id"] for s in split_order[:count // 2]}
        selected.extend({**s, "split": "development" if s["source_id"] in development else "validation"}
                        for s in chosen)
        census[task] = {"eligible_unique_sources": len(ordered), "selected": len(chosen)}
    roster, missing = [], []
    for source in selected:
        for generator in generators:
            row = {k: source[k] for k in ("source_id", "task", "split", "official_split")}
            rid = pairs.get((source["source_id"], generator))
            row.update(generator=generator, id=rid, status="available" if rid is not None else "missing")
            roster.append(row)
            if rid is None:
                missing.append(row)
    return {"batch": batch, "sources": selected, "responses": roster,
            "missing_combinations": missing, "census": census}


def graph_array_bytes(length, targets, layers, heads, hidden, kv_heads, head_dim):
    """Exact array payload of sample_graph (not .npy headers or JSON metadata)."""
    return (8 * length + 8 * targets + 8 + length
            + 4 * (3 * layers + 1) * length * hidden
            + 4 * layers * length * kv_heads * head_dim
            + 4 * layers * heads * length * length
            + targets * (8 * 8 + 8 * 4 + 4 + 4))


def label_coverage(records, original_responses):
    """Evaluation-only: call after the label-free roster has been written/hashed."""
    original = {str(row["id"]): row for row in original_responses}
    details, error_sources = [], set()
    for record in records:
        row = original[record["id"]]
        text, offsets = record["response"], record["offsets"]
        if row["response"] != text or str(row["source_id"]) != record["source_id"]:
            raise ValueError("annotation response identity changed")
        if any(not 0 <= a <= b <= len(text) for a, b in offsets):
            raise ValueError("invalid response character offsets")
        labels = row["labels"]
        if not isinstance(labels, list):
            raise ValueError("annotation labels must be a list")
        y = [0] * len(offsets)
        for label in labels:
            a, b = label["start"], label["end"]
            if (type(a) is not int or type(b) is not int or not 0 <= a < b <= len(text)
                    or ("text" in label and text[a:b] != label["text"])):
                raise ValueError("annotation character span/text mismatch")
            covered = [False] * (b - a)
            for index, (left, right) in enumerate(offsets):
                if left < b and right > a:
                    y[index] = 1
                    for pos in range(max(a, left), min(b, right)):
                        covered[pos - a] = True
            if not all(covered):
                raise ValueError("annotation has characters not covered by tokens")
        starts = [i for i, value in enumerate(y) if value and (i == 0 or not y[i - 1])]
        if starts:
            error_sources.add((record["split"], record["source_id"]))
        details.append({k: record[k] for k in ("id", "source_id", "split", "task")}
                       | {"response_tokens": len(y), "error_tokens": sum(y),
                          "annotation_spans": len(labels), "error_token_spans": len(starts),
                          "first_error_token": starts[0] if starts else None, "token_labels": y})
    return {"label_use": "coverage/alignment only, after frozen source selection",
            "validation_error_sources": sum(split == "validation" for split, _ in error_sources),
            "development_error_sources": sum(split == "development" for split, _ in error_sources),
            "response_tokens": sum(r["response_tokens"] for r in details),
            "error_tokens": sum(r["error_tokens"] for r in details),
            "error_token_spans": sum(r["error_token_spans"] for r in details),
            "denominators": {
                split: {"available_sources": len({r["source_id"] for r in details if r["split"] == split}),
                        "available_responses": sum(r["split"] == split for r in details),
                        "response_tokens": sum(r["response_tokens"] for r in details if r["split"] == split),
                        "error_tokens": sum(r["error_tokens"] for r in details if r["split"] == split),
                        "error_responses": sum(r["error_tokens"] > 0 for r in details if r["split"] == split),
                        "error_sources": sum(s == split for s, _ in error_sources)}
                for split in ("development", "validation")},
            "records": details}


def encode_roster(roster, sources, responses, tokenizer):
    source_map = {str(s["source_id"]): s for s in sources}
    response_map = {str(r["id"]): r for r in responses}
    selected = {s["source_id"]: s for s in roster["sources"]}
    special = set(tokenizer.all_special_ids)
    records = []
    for item in roster["responses"]:
        if item["status"] != "available":
            continue
        sid = item["source_id"]
        source, row = source_map[sid], response_map[item["id"]]
        prompt, response = source["prompt"], row["response"]
        p = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        r = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
        if not r["input_ids"]:
            raise ValueError(f"empty response {item['id']} retained as input failure; no replacement")
        bos = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
        left, right = selected[sid]["source_span"]
        source_mask = [False] * len(bos) + [
            a < right and b > left and token not in special
            for token, (a, b) in zip(p["input_ids"], p["offset_mapping"], strict=True)]
        if not any(source_mask):
            raise ValueError(f"source {sid} has no evidence tokens")
        if tokenizer.decode(r["input_ids"], skip_special_tokens=False,
                            clean_up_tokenization_spaces=False) != response:
            raise ValueError(f"response {item['id']} tokenizer roundtrip differs; no silent rewrite")
        prompt_ids = bos + p["input_ids"]
        records.append({k: item[k] for k in ("id", "source_id", "task", "generator", "split", "official_split")}
                       | {"prompt": prompt, "source_span": [left, right], "response": response,
                          "prompt_sha256": digest(prompt), "response_sha256": digest(response),
                          "source_text_sha256": selected[sid]["source_text_sha256"],
                          "prompt_length": len(prompt_ids), "token_ids": prompt_ids + r["input_ids"],
                          "source_mask": source_mask, "offsets": r["offset_mapping"],
                          "observer": OBSERVER, "original_generator_causality": False,
                          "prediction_query_positions": list(range(len(prompt_ids) - 1,
                                                                    len(prompt_ids) + len(r["input_ids"]) - 1))})
    return records


def collect_source_ids(obj):
    if isinstance(obj, dict):
        found = {str(obj["source_id"])} if "source_id" in obj else set()
        for value in obj.values():
            found |= collect_source_ids(value)
        return found
    if isinstance(obj, list):
        return set().union(*(collect_source_ids(value) for value in obj)) if obj else set()
    return set()


def roster_denominators(sources, responses, records):
    result = {"planned_sources": len(sources),
              "available_sources": len({r["source_id"] for r in records}),
              "planned_responses": len(responses), "available_responses": len(records),
              "missing_responses": sum(r["status"] == "missing" for r in responses)}
    result["missing_sources"] = result["planned_sources"] - result["available_sources"]
    for field in ("task", "split"):
        result["by_" + field] = {}
        for value in sorted({s[field] for s in sources}):
            planned = sum(s[field] == value for s in sources)
            available = len({r["source_id"] for r in records if r[field] == value})
            result["by_" + field][value] = {
                "planned_sources": planned, "available_sources": available, "missing_sources": planned - available,
                "planned_responses": sum(r[field] == value for r in responses),
                "available_responses": sum(r[field] == value for r in records),
                "missing_responses": sum(r[field] == value and r["status"] == "missing" for r in responses)}
    return result


def _run(args):
    from transformers import AutoTokenizer

    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    if args.seed != SEED or args.model.name != OBSERVER:
        raise ValueError("R04 preregistration requires seed 20260913 and fixed observer directory")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    exclusions, provenance = set(), []
    for path in args.exclude_roster:
        obj = read_rows(path) if path.suffix == ".jsonl" else json.loads(path.read_text())
        found = collect_source_ids(obj)
        if not found:
            raise ValueError(f"exclusion roster contains no source IDs: {path}")
        exclusions |= found
        provenance.append({"path": str(path.resolve()), "sha256": file_digest(path),
                           "source_ids": sorted(found)})
    if exclusions != EXPECTED_EXCLUSIONS:
        raise ValueError("R04 requires exactly the preregistered 6+4 excluded source IDs")
    sources = read_rows(args.dataset / "source_info.jsonl")
    metadata = read_response_metadata(args.dataset / "response.jsonl")
    quotas = dict(zip(TASKS, (12, 12, 8), strict=True))
    primary = select_sources(sources, metadata, exclusions, GENERATORS, quotas, args.seed, 0)
    reserve = select_sources(sources, metadata, exclusions, GENERATORS, quotas, args.seed, 1)
    frozen = {"schema": "r04-observer-roster@1", "seed": args.seed, "generators": GENERATORS,
              "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "primary": primary, "reserve": reserve, "exclusions": provenance,
              "selection_uses_annotations": False, "selection_uses_response_content_or_length": False,
              "reserve_rule": "use entire next batch only if primary validation error sources <10",
              "protocol_sha256": file_digest(args.protocol),
              "dataset_sha256": {name: file_digest(args.dataset / name)
                                 for name in ("source_info.jsonl", "response.jsonl")}}
    write_json(output / "frozen_roster.json", frozen)
    frozen_sha = file_digest(output / "frozen_roster.json")
    print(json.dumps({"stage": "roster_frozen_before_labels", "sha256": frozen_sha,
                      "sources": len(primary["sources"]), "reserve_sources": len(reserve["sources"])}), flush=True)
    label_join_utc = datetime.now(timezone.utc).isoformat()
    responses = read_rows(args.dataset / "response.jsonl")
    if any(file_digest(args.dataset / name) != expected
           for name, expected in frozen["dataset_sha256"].items()):
        raise ValueError("dataset changed after roster freeze")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError("offset alignment requires a fast tokenizer")
    records = encode_roster(primary, sources, responses, tokenizer)
    if file_digest(output / "frozen_roster.json") != frozen_sha:
        raise ValueError("frozen selection changed before coverage join")
    coverage_primary = label_coverage(records, responses)
    coverage_primary["roster_denominators"] = roster_denominators(primary["sources"], primary["responses"], records)
    write_json(output / "coverage_primary.json", coverage_primary)
    expanded = coverage_primary["validation_error_sources"] < 10
    if expanded:
        records += encode_roster(reserve, sources, responses, tokenizer)
    coverage = label_coverage(records, responses)
    active_sources = primary["sources"] + (reserve["sources"] if expanded else [])
    active_responses = primary["responses"] + (reserve["responses"] if expanded else [])
    denominators = roster_denominators(active_sources, active_responses, records)
    coverage["roster_denominators"] = denominators
    write_json(output / "coverage.json", coverage)
    with (output / "inputs.jsonl").open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    config = json.loads((args.model / "config.json").read_text())
    layers, heads, hidden = (config[k] for k in ("num_hidden_layers", "num_attention_heads", "hidden_size"))
    kv_heads, head_dim = config.get("num_key_value_heads", heads), config.get("head_dim", hidden // heads)
    preflight = []
    for record in records:
        length, targets = len(record["token_ids"]) - 1, len(record["offsets"])
        array_bytes = graph_array_bytes(length, targets, layers, heads, hidden, kv_heads, head_dim)
        preflight.append({"id": record["id"], "source_id": record["source_id"], "split": record["split"],
                          "prompt_tokens": record["prompt_length"], "response_tokens": targets,
                          "input_nodes": length, "graph_array_bytes": array_bytes,
                          "gpu_bf16_retained_attention_bytes_alone": 2 * layers * heads * length**2,
                          "exceeds_model_context": length > config["max_position_embeddings"],
                          "truncated": False})
    cgroup = {}
    for path in (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current"),
                 Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        if path.exists():
            cgroup[str(path)] = path.read_text().strip()
    try:
        probe = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        gpu = {"available": probe.returncode == 0 and bool(probe.stdout.strip()),
               "stdout": probe.stdout.strip(), "stderr": probe.stderr.strip(), "returncode": probe.returncode}
    except (OSError, subprocess.TimeoutExpired) as error:
        gpu = {"available": False, "error": str(error)}
    pilot_slots = active_responses[:2]
    by_id = {r["id"]: r for r in preflight}
    pilot = [by_id[r["id"]] for r in pilot_slots if r["status"] == "available"]
    disk_free = shutil.disk_usage(output).free
    pilot_bytes = sum(r["graph_array_bytes"] for r in pilot)
    pilot_disk_budget = int(pilot_bytes * 1.1) + 256 * 1024**2
    blockers = []
    if any(r["status"] == "missing" for r in pilot_slots):
        blockers.append("pilot_slot_missing")
    if len(pilot) != 2:
        blockers.append("fewer_than_two_available_responses")
    if any(r["exceeds_model_context"] for r in pilot):
        blockers.append("pilot_exceeds_model_context")
    if pilot_disk_budget > disk_free:
        blockers.append("insufficient_filesystem_free_space_for_pilot_budget")
    if not gpu["available"]:
        blockers.append("gpu_probe_unavailable")
    resource = {"records": preflight, "disk_free_bytes": disk_free, "denominators": denominators,
                "personal_quota": "unknown; filesystem free space is not user quota",
                "gpu_snapshot": gpu, "cgroup_memory": cgroup,
                "graph_array_bytes_total": sum(r["graph_array_bytes"] for r in preflight),
                "npy_headers_metadata_extra": True, "gpu_estimate_is_not_peak_measurement": True,
                "pilot_frozen_slots": pilot_slots,
                "pilot_response_ids": [r["id"] for r in pilot_slots],
                "pilot_array_bytes": pilot_bytes, "pilot_disk_budget_bytes": pilot_disk_budget,
                "pilot_disk_margin_bytes": disk_free - pilot_disk_budget,
                "pilot_known_blockers": blockers,
                "pilot_gate": "blocked" if blockers else "eligible_for_resource_pilot_not_confirmed_fit",
                "remaining_r05_checks": ["personal quota", "actual GPU/CPU peak", "causal alignment"],
                "context_exceeded_responses": sum(r["exceeds_model_context"] for r in preflight),
                "model_shape": {"layers": layers, "heads": heads, "hidden": hidden,
                                "kv_heads": kv_heads, "head_dim": head_dim},
                "full_prompt_and_cross_span_edges_required": True}
    write_json(output / "resource_preflight.json", resource)
    model_identity = {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                      for p in sorted(args.model.iterdir()) if p.is_file()}
    for name, info in model_identity.items():
        print(json.dumps({"stage": "hash_model_file", "file": name, "bytes": info["bytes"]}), flush=True)
        info["sha256"] = file_digest(args.model / name)
    if any(p.stat().st_size != model_identity[p.name]["bytes"]
           or p.stat().st_mtime_ns != model_identity[p.name]["mtime_ns"]
           for p in args.model.iterdir() if p.is_file()):
        raise ValueError("model files changed during identity hashing")
    executed = output / "executed_code"
    executed.mkdir()
    from decoding import ragtruth_population_io
    from route_graph import sample_graph
    repo = Path(__file__).resolve().parents[2]
    code_paths = [Path(__file__), Path(ragtruth_population_io.__file__), Path(sample_graph.__file__), args.protocol]
    code_paths += [p for p in (repo / "scripts/run_detection_roster.sh", repo / "tests/test_detection_roster.py")
                   if p.exists()]
    for path in code_paths:
        shutil.copyfile(path, executed / path.name)
    summary = {"status": "complete", "roster_sha256": frozen_sha, "expanded": expanded,
               "source_count": len({r["source_id"] for r in records}), "responses": len(records),
               "planned_sources": len(primary["sources"]) + (len(reserve["sources"]) if expanded else 0),
               "denominators": denominators,
               "missing_combinations": primary["missing_combinations"] + (reserve["missing_combinations"] if expanded else []),
               "by_task": dict(Counter(r["task"] for r in records)),
               "by_split": dict(Counter(r["split"] for r in records)),
               "validation_error_sources": coverage["validation_error_sources"],
               "coverage_gate": "sufficient" if coverage["validation_error_sources"] >= 10 else "insufficient_after_max_extension",
               "r05_gate": resource["pilot_gate"], "r05_known_blockers": blockers,
               "elapsed_seconds": time.monotonic() - started, "model_forwards": 0,
               "started_at_utc": started_utc, "label_join_at_utc": label_join_utc,
               "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "model_path": str(args.model.resolve()), "model_files": model_identity,
               "weight_identity": "SHA256 of every top-level model file including all weight shards",
               "tokenizer_identity": {"class": type(tokenizer).__name__, "is_fast": tokenizer.is_fast,
                                      "bos_token_id": tokenizer.bos_token_id,
                                      "all_special_ids": tokenizer.all_special_ids,
                                      "transformers_version": version("transformers"),
                                      "tokenizers_version": version("tokenizers")},
               "inputs_sha256": file_digest(output / "inputs.jsonl")}
    write_json(output / "summary.json", summary)
    manifest = {"status": "complete", "artifacts": {
        str(p.relative_to(output)): file_digest(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in summary.items() if k != "model_files"}), flush=True)


def run(args):
    if args.output.exists():
        raise FileExistsError(f"refuse to reuse existing output: {args.output}")
    try:
        _run(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / "failure.json", {"status": "failed", "type": type(error).__name__,
                                                     "error": str(error), "preserve_partial_artifacts": True})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--exclude-roster", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260913)
    run(parser.parse_args())
