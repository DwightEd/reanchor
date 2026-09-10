"""Train only on program targets, select only on source-disjoint program dev."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch

from reanchor.features import FeatureStore
from reanchor.io import assert_disjoint, empty_directory, file_digest, write_json
from reanchor.support import SupportHead, support_loss


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_store(store, expected_split):
    if not store.samples or any(r["split"] != expected_split for r in store.samples):
        raise ValueError(f"cache must contain only {expected_split!r} split")
    if any(r["kind"] not in {"binding", "neutral"} for r in store.samples):
        raise ValueError("training/selection requires program targets, not natural responses")
    if {r["kind"] for r in store.samples} != {"binding", "neutral"}:
        raise ValueError("both binding and neutral positions are required")


@torch.no_grad()
def program_metrics(model, store, device, candidate_chunk=256):
    model.eval()
    losses, correct, case_correct, case_pairs = defaultdict(list), defaultdict(list), {}, {}
    for row in store.samples:
        if row["kind"] not in {"binding", "neutral"}:
            raise ValueError("program metrics cannot use unlabeled natural responses")
        values = model(*store.tensors(row, device), candidate_chunk=candidate_chunk)
        loss = float(support_loss(values, row["kind"], row["target_index"]).item())
        losses[row["kind"]].append(loss)
        if row["kind"] == "binding":
            success = int(values.argmax().item()) == row["target_index"]
            role = row["program"]["role"]
            correct[role].append(success)
            response = row["response_id"]
            case_correct[response] = case_correct.get(response, True) and success
            case_pairs[response] = row["program"]["pair_id"]
    pairs = defaultdict(list)
    for response, success in case_correct.items():
        pairs[case_pairs[response]].append(success)
    if any(len(pair) != 2 for pair in pairs.values()):
        raise ValueError("paired-world evaluation needs exactly A and B for each pair")
    result = {f"{kind}_loss": sum(values) / len(values) for kind, values in losses.items()}
    result["objective"] = result["binding_loss"] + result["neutral_loss"]
    result["role_accuracy"] = {key: sum(values) / len(values) for key, values in correct.items()}
    result["world_pair_conjunction"] = sum(all(v) for v in pairs.values()) / len(pairs)
    result["pairs"] = len(pairs)
    result["scope"] = "program target accuracy, not hallucination detection accuracy"
    return result


def train_head(
    train_path: Path,
    dev_path: Path,
    output: Path,
    *,
    device="cuda:0",
    width=128,
    blocks=2,
    epochs=10,
    batch_size=4,
    learning_rate=3e-4,
    seed=42,
    candidate_chunk=256,
):
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    train, dev = FeatureStore(train_path), FeatureStore(dev_path)
    check_store(train, "train")
    check_store(dev, "dev")
    train_sources = {r["source_id"] for r in train.samples}
    dev_sources = {r["source_id"] for r in dev.samples}
    assert_disjoint({"train": train_sources, "dev": dev_sources})
    if train.manifest["extraction_identity"] != dev.manifest["extraction_identity"]:
        raise ValueError("train/dev extraction identity mismatch")
    empty_directory(output)
    seed_all(seed)
    device = torch.device(device)
    model = SupportHead(train.manifest["backbone"]["input_dim"], width, blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    config = {
        "schema": "reanchor/support-training@1",
        "width": width,
        "blocks": blocks,
        "epochs": epochs,
        "batch_size_per_kind": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "device": str(device),
        "candidate_chunk": candidate_chunk,
        "objective": "softmax(f) binding CE + centered neutral variance",
        "train_sources": sorted(train_sources),
        "dev_sources": sorted(dev_sources),
        "train_index_digest": train.manifest["cache_digest"],
        "dev_index_digest": dev.manifest["cache_digest"],
        "backbone": train.manifest["backbone"],
        "extraction_identity": train.manifest["extraction_identity"],
        "selection": "program dev objective only",
    }
    write_json(output / "config.json", config)
    groups = {
        kind: [r for r in train.samples if r["kind"] == kind] for kind in ("binding", "neutral")
    }
    best = math.inf
    best_epoch = None
    with (output / "history.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(epochs):
            model.train()
            rng = random.Random(seed + epoch)
            for rows in groups.values():
                rng.shuffle(rows)
            steps = math.ceil(max(map(len, groups.values())) / batch_size)
            train_loss = 0.0
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0
                for kind, rows in groups.items():
                    batch = [rows[(step * batch_size + i) % len(rows)] for i in range(batch_size)]
                    for row in batch:
                        values = model(*train.tensors(row, device), candidate_chunk=candidate_chunk)
                        loss = support_loss(values, kind, row["target_index"]) / len(batch)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("non-finite support training loss")
                        loss.backward()
                        step_loss += float(loss.detach().item())
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                train_loss += step_loss
            metrics = program_metrics(model, dev, device, candidate_chunk)
            record = {"epoch": epoch + 1, "train_loss": train_loss / steps, "dev": metrics}
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            print(json.dumps(record, allow_nan=False), flush=True)
            if metrics["objective"] < best:
                best, best_epoch = metrics["objective"], epoch + 1
                # This path belongs to this newly created run; never user checkpoints.
                torch.save(
                    {
                        "schema": "reanchor/support-head@1",
                        "architecture": model.config,
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "training": config,
                        "best_epoch": best_epoch,
                        "dev": metrics,
                    },
                    output / "head.pt",
                )
    result = {
        "best_epoch": best_epoch,
        "dev_objective": best,
        "checkpoint": "head.pt",
        "checkpoint_digest": file_digest(output / "head.pt"),
    }
    write_json(output / "summary.json", result)
    return result


def load_head(checkpoint: Path, store, device):
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if artifact["schema"] != "reanchor/support-head@1":
        raise ValueError("unsupported head checkpoint")
    if artifact["training"]["extraction_identity"] != store.manifest["extraction_identity"]:
        raise ValueError("checkpoint/cache extraction identity mismatch")
    model = SupportHead(**artifact["architecture"]).to(device)
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.eval()
    return model, artifact


@torch.no_grad()
def score_head(
    cache: Path, checkpoint: Path, output: Path, *, device="cuda:0", candidate_chunk=256
):
    store = FeatureStore(cache)
    device = torch.device(device)
    model, artifact = load_head(checkpoint, store, device)
    used = set(artifact["training"]["train_sources"]) | set(artifact["training"]["dev_sources"])
    if used & {row["source_id"] for row in store.samples}:
        raise ValueError("scoring/calibration sources overlap training or selection sources")
    empty_directory(output)
    count = 0
    with (output / "scores.jsonl").open("x", encoding="utf-8") as stream:
        for row in store.samples:
            values = model(*store.tensors(row, device), candidate_chunk=candidate_chunk)
            if not torch.isfinite(values).all():
                raise FloatingPointError("non-finite support score")
            best = int(values.argmax().item())
            record = {
                key: row[key]
                for key in (
                    "response_id",
                    "source_id",
                    "split",
                    "task",
                    "token_index",
                    "token_id",
                    "char_start",
                    "char_end",
                    "source_length",
                    "negative_margin",
                    "response_digest",
                    "source_text_digest",
                    "instruction_digest",
                    "response_token_count",
                )
            }
            record.update(
                {
                    "raw_score": float((values[best] - values[row["target_index"]]).item()),
                    "candidate_count": len(row["candidate_ids"]),
                    "best_candidate_id": row["candidate_ids"][best],
                }
            )
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            count += 1
    metadata = {
        "schema": "reanchor/frozen-scores@1",
        "samples": count,
        "head_digest": file_digest(checkpoint),
        "cache_digest": store.manifest["cache_digest"],
        "scores_digest": file_digest(output / "scores.jsonl"),
        "scope": store.manifest["scope"],
        "backbone": store.manifest["backbone"],
        "extraction_identity": store.manifest["extraction_identity"],
        "dataset_identity": store.manifest["dataset_identity"],
        "preparation_identity": store.manifest["preparation_identity"],
        "labels_used": False,
    }
    write_json(output / "manifest.json", metadata)
    if store.manifest["scope"] == "program-selected-slots":
        metrics = program_metrics(model, store, device, candidate_chunk)
        write_json(output / "program_metrics.json", metrics)
    return metadata
