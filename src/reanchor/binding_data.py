"""Program-verifiable paired worlds; no generated hallucination labels."""

from __future__ import annotations

import random
from pathlib import Path

from reanchor.io import (
    assert_disjoint,
    digest,
    empty_directory,
    read_jsonl,
    write_json,
    write_jsonl,
)

SCHEMA = "reanchor/binding-example@1"
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
SPLITS = ("train", "dev", "calibration", "test")


def paired_examples(source_id: str, split: str, rng: random.Random, records: int = 6):
    if records < 6:
        raise ValueError("at least six records are required")
    # IDs/names do not indicate which record is queried. The order is independent.
    numbers = rng.sample(range(10, 9999), records)
    names = [f"Gallery {number}" for number in numbers]
    values = [rng.choice(DAYS) for _ in names]
    values[1], values[2] = rng.sample(DAYS, 2)  # target and unmentioned donor differ
    order = rng.sample(range(records), records)
    worlds = [values[:], values[:]]
    worlds[1][1], worlds[1][2] = worlds[1][2], worlds[1][1]
    relay_depth = rng.randint(1, 3)
    for world_index, world_values in enumerate(worlds):
        world = "AB"[world_index]
        facts = dict(zip(names, world_values, strict=True))
        if split == "test":
            source = "\n".join(
                f"The opening day for {names[i]} is {world_values[i]}." for i in order
            )
        else:
            source = "\n".join(f"{names[i]} opens on {world_values[i]}." for i in order)
        # Only records 0 and 3 have factual assertions in H; neither is swapped.
        histories = [
            f"{names[0]} opens on {world_values[0]}. ",
            f"{names[3]} opens on {world_values[3]}. {names[0]} opens on {world_values[0]}. ",
        ]
        for role in ("explicit", "referential"):
            for variant in range(2):
                target = 1 if role == "explicit" else 1 + variant
                history = histories[variant] if role == "explicit" else histories[0]
                if role == "explicit":
                    instruction = f"State the opening day of {names[target]}."
                    prefix = history + f"As for {names[target]}, it opens on "
                else:
                    instruction = "State the opening day of the selected gallery."
                    prefix = history + f"The selected gallery is {names[target]}. "
                    prefix += "We continue discussing that same gallery. " * (relay_depth - 1)
                    prefix += "It opens on "
                answer = world_values[target]
                response = prefix + answer + "."
                case = f"{role}-{variant}"
                yield {
                    "schema": SCHEMA,
                    "id": f"{source_id}/{world}/{case}",
                    "source_id": source_id,
                    "split": split,
                    "task": "program-binding",
                    "source": source,
                    "instruction": instruction,
                    "response": response,
                    "supervision": {
                        "kind": "program-target-not-hallucination-label",
                        "binding_spans": [[len(prefix), len(prefix) + len(answer)]],
                        "neutral_spans": [[len(response) - 1, len(response)]],
                    },
                    "program": {
                        "facts": facts,
                        "target": names[target],
                        "answer": answer,
                        "world": world,
                        "role": role,
                        "variant": variant,
                        "pair_id": f"{source_id}/{case}",
                        "relay_depth": relay_depth,
                        "asserted_entities": [names[0]]
                        if variant == 0 or role == "referential"
                        else [names[3], names[0]],
                        "donor": names[2 if target == 1 else 1],
                    },
                }


def generate_dataset(output: Path, sources: int = 64, seed: int = 42, records: int = 6):
    if sources < 8:
        raise ValueError("sources must be >= 8 to populate four source-disjoint splits")
    empty_directory(output)
    rng = random.Random(seed)
    source_numbers = rng.sample(range(sources), sources)
    heldout = max(1, sources // 8)
    counts = [sources - 3 * heldout, heldout, heldout, heldout]
    assignments = {}
    cursor = 0
    for split, count in zip(SPLITS, counts, strict=True):
        examples = []
        assignments[split] = []
        for number in source_numbers[cursor : cursor + count]:
            source_id = f"program-{seed}-{number:06d}"
            assignments[split].append(source_id)
            examples.extend(paired_examples(source_id, split, rng, records))
        write_jsonl(output / f"{split}.jsonl", examples)
        cursor += count
    assert_disjoint({k: set(v) for k, v in assignments.items()})
    summary = {
        "schema": "reanchor/program-dataset@1",
        "seed": seed,
        "sources": sources,
        "records_per_world": records,
        "assignments": assignments,
        "supervision": "program-verifiable targets; no H labels",
        "scope": "opening-day binding; test source wording held out; not natural RAG",
    }
    summary["dataset_digest"] = digest(summary)
    write_json(output / "dataset.json", summary)
    return summary


def load_examples(path: Path):
    result = []
    ids = set()
    groups = {}
    for item in read_jsonl(path):
        if set(item) & {"labels", "hallucination", "is_hallucinated", "label"}:
            raise ValueError("hallucination labels forbidden in measurement input")
        if item.get("schema") not in (SCHEMA, "reanchor/response@1"):
            raise ValueError("expected normalized binding-example@1 or response@1 input")
        for key in ("id", "source_id", "split", "task", "instruction", "response"):
            if not isinstance(item.get(key), str) or not item[key]:
                raise ValueError(f"missing/non-string {key}")
        if not isinstance(item.get("source"), str):
            raise ValueError("source must be a string")
        if item["id"] in ids:
            raise ValueError(f"duplicate response id: {item['id']}")
        ids.add(item["id"])
        groups.setdefault(item["split"], set()).add(item["source_id"])
        if item["schema"] == SCHEMA:
            program = item["program"]
            if program["facts"].get(program["target"]) != program["answer"]:
                raise ValueError("program target does not match source record")
            for start, end in item["supervision"]["binding_spans"]:
                if item["response"][start:end] != program["answer"]:
                    raise ValueError("program answer span mismatch")
        elif "supervision" in item or "program" in item:
            raise ValueError("natural response input must not contain training targets")
        result.append(item)
    assert_disjoint(groups)
    if not result:
        raise ValueError(f"no examples in {path}")
    return result
