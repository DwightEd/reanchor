"""Annotation-free RAGTruth inputs and atomic per-response mechanism artifacts."""

import hashlib
import json

import numpy as np

from decoding.adoption_probe import sha256, write_json

TASKS = ("QA", "Summary", "Data2txt")


def input_digest(record):
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def source_interval(source):
    task, info, prompt = source["task_type"], source["source_info"], source["prompt"]
    evidence = info["passages"] if task == "QA" else (str(info) if task == "Data2txt" else info)
    evidence = evidence.strip()
    start = prompt.find(evidence)
    if not evidence or start < 0 or prompt.find(evidence, start + 1) >= 0:
        raise ValueError(
            f"source {source['source_id']}: evidence must match original prompt exactly once"
        )
    return [start, start + len(evidence)]


def prepare_records(dataset, tokenizer, smoke=False):
    sources, encoded = {}, {}
    with (dataset / "source_info.jsonl").open() as stream:
        for line in stream:
            source = json.loads(line)
            sid = str(source["source_id"])
            if sid in sources or source["task_type"] not in TASKS:
                raise ValueError("duplicate source or unsupported task")
            source["source_span"] = source_interval(source)
            sources[sid] = source
    special = set(tokenizer.all_special_ids)
    for sid, source in sources.items():
        tokens = tokenizer(source["prompt"], add_special_tokens=False, return_offsets_mapping=True)
        bos = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
        left, right = source["source_span"]
        mask = [False] * len(bos) + [
            a < right and b > left and token not in special
            for token, (a, b) in zip(tokens["input_ids"], tokens["offset_mapping"], strict=True)
        ]
        if not any(mask):
            raise ValueError(f"source {sid} has no nonspecial evidence token")
        encoded[sid] = (bos + tokens["input_ids"], mask)
    records, seen = [], set()
    with (dataset / "response.jsonl").open() as stream:
        for line in stream:
            # Annotation fields are deliberately not copied, inspected or used for selection.
            raw = json.loads(line)
            rid, sid = str(raw["id"]), str(raw["source_id"])
            if rid in seen or sid not in sources or not rid.isdecimal():
                raise ValueError("invalid response/source identity")
            seen.add(rid)
            source = sources[sid]
            response = tokenizer(
                raw["response"], add_special_tokens=False, return_offsets_mapping=True
            )
            if not response["input_ids"]:
                raise ValueError(f"empty response {rid}")
            prompt_ids, mask = encoded[sid]
            records.append(
                dict(
                    id=rid,
                    source_id=sid,
                    task=source["task_type"],
                    generator=raw["model"],
                    official_split=raw["split"],
                    prompt=source["prompt"],
                    source_span=source["source_span"],
                    response=raw["response"],
                    response_sha256=hashlib.sha256(raw["response"].encode()).hexdigest(),
                    prompt_length=len(prompt_ids),
                    token_ids=prompt_ids + response["input_ids"],
                    source_mask=mask,
                    offsets=response["offset_mapping"],
                )
            )
    records.sort(
        key=lambda r: (
            TASKS.index(r["task"]),
            r["generator"] != "llama-2-7b-chat",
            r["generator"],
            int(r["id"]),
        )
    )
    if smoke:
        selected = []
        for task in TASKS:
            group = [r for r in records if r["task"] == task]
            first, longest = group[0], max(group, key=lambda r: len(r["token_ids"]))
            selected.extend([first, longest] if first["id"] != longest["id"] else group[:2])
        records = selected
    return records


def verify_response(directory, record):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["input_digest"] != input_digest(record):
        raise ValueError(f"completed input changed: {record['id']}")
    for name, expected in manifest["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"completed artifact changed: {directory / name}")


def publish_response(partial, destination, record, metrics, profiles, diagnostics):
    np.savez_compressed(partial / "metrics.npz", **metrics)
    np.savez_compressed(partial / "profiles.npz", **profiles)
    np.savez_compressed(
        partial / "tokens.npz",
        token_ids=np.asarray(record["token_ids"], np.int32),
        source_mask=np.asarray(record["source_mask"], bool),
        offsets=np.asarray(record["offsets"], np.int32),
    )
    metadata = {k: v for k, v in record.items() if k not in {"token_ids", "source_mask", "offsets"}}
    write_json(partial / "record.json", {**metadata, "diagnostics": diagnostics})
    write_json(
        partial / "manifest.json",
        dict(
            input_digest=input_digest(record),
            files={p.name: sha256(p) for p in sorted(partial.iterdir()) if p.is_file()},
        ),
    )
    if destination.exists():
        raise FileExistsError(destination)
    partial.rename(destination)


def append_json(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def roster_summary(records):
    from collections import Counter

    return dict(
        responses=len(records),
        sources=len({r["source_id"] for r in records}),
        by_task=dict(Counter(r["task"] for r in records)),
        by_generator=dict(Counter(r["generator"] for r in records)),
        max_input_tokens=max(len(r["token_ids"]) - 1 for r in records),
        response_tokens=sum(len(r["offsets"]) for r in records),
    )
