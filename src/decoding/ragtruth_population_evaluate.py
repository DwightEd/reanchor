"""Evaluation-only annotation join for frozen RAGTruth mechanism measurements."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from route_graph.metrics import binary_detection_metrics

from decoding.adoption_probe import sha256, write_json
from decoding.ragtruth_population_io import verify_response


def evaluate(run, labels_path, output):
    if output.exists():
        raise FileExistsError(output)
    settings = json.loads((run / "settings.json").read_text())
    if sha256(labels_path) != settings["input_sha256"]["response.jsonl"]:
        raise ValueError("evaluation annotations/text differ from frozen input file")
    roster_path = run / "inputs.jsonl"
    input_manifest = json.loads((run / "input_manifest.json").read_text())
    if sha256(roster_path) != input_manifest["sha256"]:
        raise ValueError("evaluation input roster changed")
    roster = {}
    for line in roster_path.open():
        record = json.loads(line)
        if record["id"] in roster:
            raise ValueError("duplicate input roster identity")
        roster[record["id"]] = record
    labels = {}
    for line in labels_path.open():
        row = json.loads(line)
        if str(row["id"]) in labels:
            raise ValueError("duplicate annotation response identity")
        labels[str(row["id"])] = row
    groups = defaultdict(list)
    completed = []
    for directory in sorted((run / "responses").iterdir()):
        if not directory.is_dir():
            continue
        record = json.loads((directory / "record.json").read_text())
        if directory.name != record["id"] or record["id"] not in roster:
            raise ValueError("completed response directory identity differs from input roster")
        verify_response(directory, roster[record["id"]])
        annotation = labels[record["id"]]
        if (
            str(annotation["source_id"]) != record["source_id"]
            or annotation["model"] != record["generator"]
            or annotation["split"] != record["official_split"]
            or hashlib.sha256(annotation["response"].encode()).hexdigest()
            != record["response_sha256"]
        ):
            raise ValueError("annotation/measurement text, source, split or generator mismatch")
        with np.load(directory / "tokens.npz", allow_pickle=False) as tokens:
            offsets = tokens["offsets"].copy()
            saved_tokens = tokens["token_ids"][record["prompt_length"] :].copy()
        if (
            offsets.shape != (len(saved_tokens), 2)
            or np.any(offsets[:, 0] < 0)
            or np.any(offsets[:, 1] > len(record["response"]))
            or np.any(offsets[:, 0] >= offsets[:, 1])
        ):
            raise ValueError("invalid full response token offsets")
        y = np.zeros(len(offsets), bool)
        for span in annotation["labels"]:
            if not 0 <= span["start"] < span["end"] <= len(record["response"]):
                raise ValueError("invalid annotation span")
            y |= (offsets[:, 0] < span["end"]) & (offsets[:, 1] > span["start"])
        prefix = np.ones(len(y), bool)
        if y.any():
            prefix[np.flatnonzero(y)[0] + 1 :] = False
        with np.load(directory / "metrics.npz", allow_pickle=False) as measured:
            scores = {
                "entropy": measured["base__entropy"].copy(),
                "negative_margin": -measured["base__margin"].copy(),
                "source_sensitivity": measured["source_1__js"].copy(),
                "source_small_sensitivity": measured["source_01__js"].copy(),
                "source_saved_token_support": -measured["source_01__saved_logp_change"].copy(),
                "history_sensitivity": measured["history_1__js"].copy(),
                "history_small_sensitivity": measured["history_01__js"].copy(),
                "topology_sensitivity": measured["source_permute__js"].copy(),
                "mlp_sensitivity": measured["mlp_01__js"].copy(),
            }
            mismatch = measured["base__argmax"] != saved_tokens
        if any(v.shape != y.shape or not np.isfinite(v).all() for v in scores.values()):
            raise ValueError("measurement coverage or finiteness mismatch")
        key = (record["task"], record["generator"], record["official_split"])
        groups[key].append(
            dict(
                id=record["id"],
                source_id=record["source_id"],
                y=y,
                prefix=prefix,
                scores=scores,
                observer_mismatch=mismatch,
            )
        )
        completed.append(record["id"])
    rows = []
    for (task, generator, split), members in sorted(groups.items()):
        y = np.concatenate([r["y"] for r in members])
        source = np.concatenate([np.full(len(r["y"]), r["source_id"]) for r in members])
        _, source_index = np.unique(source, return_inverse=True)
        masks = {
            "all_tokens": np.ones(len(y), bool),
            "through_first_error": np.concatenate([r["prefix"] for r in members]),
        }
        subsets = {}
        for subset, mask in masks.items():
            measures = {}
            for name in members[0]["scores"]:
                values = np.concatenate([r["scores"][name] for r in members])
                # Diagnostic direction is fixed: higher measured value vs annotated error.
                score = (
                    binary_detection_metrics(
                        y[mask],
                        values[mask],
                        source[mask],
                        seed=settings["seed"],
                        bootstrap=0,
                        source_balanced=True,
                    )
                    if len(np.unique(y[mask])) == 2
                    else None
                )
                source_means = {}
                for label, name_label in ((False, "normal"), (True, "error")):
                    selected = mask & (y == label)
                    count = np.bincount(source_index[selected])
                    total = np.bincount(source_index[selected], weights=values[selected])
                    means = total[count > 0] / count[count > 0]
                    source_means[name_label] = dict(
                        sources=len(means), mean=float(np.mean(means)) if len(means) else None
                    )
                measures[name] = dict(
                    descriptive_ranking=score, source_balanced_conditional_mean=source_means
                )
            subsets[subset] = dict(
                tokens=int(mask.sum()), error_tokens=int(y[mask].sum()), metrics=measures
            )
        rows.append(
            dict(
                task=task,
                generator=generator,
                official_split=split,
                responses=len(members),
                sources=len(set(source)),
                observer_saved_token_mismatch=float(
                    np.concatenate([r["observer_mismatch"] for r in members]).mean()
                ),
                subsets=subsets,
            )
        )
    progress = json.loads((run / "progress.json").read_text())
    if progress["status"] in {"evaluating", "complete", "finished_with_failures"}:
        if len(completed) != progress["completed"]:
            raise ValueError("final progress and verified evaluation coverage differ")
    write_json(
        output,
        dict(
            schema="ragtruth-population-evaluation@1",
            scope="observer-replay mechanism sensitivity, not validated ownership detector",
            labels_used_stage="evaluation_only",
            run_settings_sha256=sha256(run / "settings.json"),
            evaluator_sha256=sha256(Path(__file__)),
            coverage=progress,
            verified_completed_responses=len(completed),
            completed_response_ids=completed,
            groups=rows,
            scientific_review="REVIEW_UNAVAILABLE",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.run, args.labels, args.output)
