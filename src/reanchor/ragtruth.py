"""RAGTruth adapter. Preparation never selects examples using hallucination labels."""

from __future__ import annotations

import json
from pathlib import Path

from reanchor.io import digest, empty_directory, file_digest, read_jsonl, write_json, write_jsonl


def dataset_identity(dataset):
    # Opaque byte hashes bind the later label join; they do not select or score examples.
    return {name: file_digest(dataset / name) for name in ("source_info.jsonl", "response.jsonl")}


def source_content(item):
    info, task = item["source_info"], item["task_type"]
    if task == "QA":
        return str(info["passages"]), f"Answer using the evidence: {info['question']}"
    if task == "Summary":
        return str(info), "Summarize the supplied evidence without adding unsupported facts."
    if task == "Data2txt":
        return json.dumps(info, ensure_ascii=False, sort_keys=True), (
            "Write a factual overview of the business using only the supplied structured evidence."
        )
    raise ValueError(f"unsupported RAGTruth task: {task}")


def prepare_ragtruth(
    dataset: Path,
    output: Path,
    *,
    seed=42,
    limit_sources=32,
    calibration_fraction=0.25,
    model_filter=None,
    exclude_sources=None,
):
    if not 0 < calibration_fraction < 1 or limit_sources < 0:
        raise ValueError("invalid calibration fraction or source limit")
    identity = dataset_identity(dataset)
    sources = {}
    for row in read_jsonl(dataset / "source_info.jsonl"):
        source_id = str(row["source_id"])
        if source_id in sources:
            raise ValueError(f"duplicate source {source_id}")
        sources[source_id] = row
    excluded = set()
    if exclude_sources:
        excluded = set(json.loads(exclude_sources.read_text(encoding="utf-8")))
    # This stage copies only whitelisted text/identity fields. It never reads
    # labels, quality flags, or a model's estimated correctness for selection.
    responses = []
    ids = set()
    for item in read_jsonl(dataset / "response.jsonl"):
        if model_filter is not None and item["model"] != model_filter:
            continue
        source_id = str(item["source_id"])
        if source_id in excluded:
            continue
        if source_id not in sources:
            raise ValueError(f"missing source: {source_id}")
        response_id = str(item["id"])
        if response_id in ids:
            raise ValueError(f"duplicate response: {response_id}")
        ids.add(response_id)
        responses.append(
            {
                "id": response_id,
                "source_id": source_id,
                "response": item["response"],
                "generator": item["model"],
            }
        )
    ordered = sorted({r["source_id"] for r in responses}, key=lambda s: digest([seed, s]))
    if limit_sources:
        ordered = ordered[:limit_sources]
    if len(ordered) < 2:
        raise ValueError("at least two sources required for disjoint calibration/test")
    n_calibration = max(1, min(len(ordered) - 1, round(len(ordered) * calibration_fraction)))
    assignments = {s: "calibration" if i < n_calibration else "test" for i, s in enumerate(ordered)}
    preparation_config = {
        "dataset_identity": identity,
        "seed": seed,
        "limit_sources": limit_sources,
        "calibration_fraction": calibration_fraction,
        "model_filter": model_filter,
        "excluded_sources": sorted(excluded),
        "assignments": assignments,
        "schema": "reanchor/ragtruth-preparation@2",
        "preparer_code_digest": file_digest(Path(__file__)),
    }
    preparation_identity = digest(preparation_config)
    groups = {"calibration": [], "test": []}
    for row in responses:
        if row["source_id"] not in assignments:
            continue
        raw_source = sources[row["source_id"]]
        source, instruction = source_content(raw_source)
        split = assignments[row["source_id"]]
        groups[split].append(
            {
                **row,
                "schema": "reanchor/response@1",
                "split": split,
                "task": raw_source["task_type"],
                "source": source,
                "instruction": instruction,
                "dataset_identity": identity,
                "preparation_identity": preparation_identity,
            }
        )
    empty_directory(output)
    for split, rows in groups.items():
        write_jsonl(output / f"{split}.jsonl", rows)
    summary = {
        "schema": "reanchor/ragtruth-preparation@2",
        "preparation_config": preparation_config,
        "preparation_identity": preparation_identity,
        "seed": seed,
        "dataset_identity": identity,
        "assignments": assignments,
        "responses": {k: len(v) for k, v in groups.items()},
        "excluded_sources": sorted(excluded),
        "model_filter": model_filter,
        "labels_used_for_selection": False,
        "scope": "exploratory observer evaluation; prior discovery sources must be excluded",
        "prompt_policy": "new source/instruction decomposition; not original generator replay",
    }
    write_json(output / "dataset.json", summary)
    return summary
