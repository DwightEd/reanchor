"""Local, foreground lookback interventions; use from the reanchor repository root."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .data import encode_pair, load_pairs, make_pairs, symbolic_example
from .patching import capture_states, patch_logits, summarize_logits
from .subspace import fit_mask, source_split, svd_basis


VERSION = "lookback-beliefs-v1"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    with path.with_suffix(".partial").open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    path.with_suffix(".partial").replace(path)


def prepare(model, tokenizer, pair, layers, args):
    device = next(model.parameters()).device
    e = encode_pair(pair, tokenizer, device, args.chat_template)
    if max(e["base"]["input_ids"].numel(), e["donor"]["input_ids"].numel()) > args.max_tokens:
        raise ValueError(f"{pair.id}: prompt exceeds --max-tokens; no truncation is allowed")
    requests = {i: list(e["base_sites"]) for i in layers}
    for i, pos in e["restore"].items():
        requests[i] = list(dict.fromkeys(requests.get(i, []) + pos))
    base_cache, base_logits = capture_states(model, e["base"], requests)
    donor_cache, donor_logits = capture_states(model, e["donor"], {i: e["donor_sites"] for i in layers})
    e["base_states"] = {i: base_cache[i][:, [requests[i].index(p) for p in e["base_sites"]]] for i in layers}
    e["restore_states"] = {i: (pos, base_cache[i][:, [requests[i].index(p) for p in pos]])
                           for i, pos in e["restore"].items()}
    e["donor_states"] = donor_cache
    base = summarize_logits(base_logits, e["candidate_ids"], e["candidates"], tokenizer)
    donor = summarize_logits(donor_logits, e["donor_candidate_ids"], e["candidates"], tokenizer)
    # Same-world replacement is a zero intervention, independent of hypothesis.
    with torch.no_grad():
        check = patch_logits(model, e["base"], layers[0], e["base_sites"], e["base_states"][layers[0]],
                             restore=e["restore_states"])
    error = float((check.detach().cpu() - base_logits).abs().max())
    if not torch.allclose(check.detach().cpu(), base_logits, rtol=1e-3, atol=1e-3):
        raise ValueError(f"{pair.id}: same-world patch changed baseline logits (max error={error})")
    e["baseline"] = dict(base=base, donor=donor, self_patch_max_error=error,
                         both_correct=base["matches"][pair.base_answer] and donor["matches"][pair.donor_answer])
    return e


def summarize(records):
    """IIA is agreement with a named high-level intervention, NOT factual accuracy."""
    cells = {}
    for record in records:
        for measurement in record["measurements"]:
            for hypothesis, target in record["hypotheses"].items():
                key = (measurement["layer"], measurement["arm"], hypothesis)
                cell = cells.setdefault(key, {"all": [], "both_correct": []})
                match = int(measurement["matches"][target])
                cell["all"].append(match)
                if record["baseline"]["both_correct"]:
                    cell["both_correct"].append(match)
    curves = []
    for (layer, arm, hypothesis), cell in sorted(cells.items()):
        curves.append(dict(layer=layer, arm=arm, hypothesis=hypothesis,
                           n_all=len(cell["all"]), iia_all=float(np.mean(cell["all"])),
                           n_both_correct=len(cell["both_correct"]),
                           iia_both_correct=float(np.mean(cell["both_correct"])) if cell["both_correct"] else None))
    return dict(examples=len(records), both_correct=sum(r["baseline"]["both_correct"] for r in records),
                interpretation="Interchange Intervention Accuracy; not RAGTruth detection performance",
                curves=curves)


def save_measurements(out, e, measurements, prefix=""):
    p = e["pair"]
    record = dict(id=p.id, source_id=p.source_id, hypotheses=p.hypotheses, baseline=e["baseline"],
                  base_sites=e["base_sites"], donor_sites=e["donor_sites"], measurements=measurements)
    path = out / "samples" / (prefix + p.id)
    path.parent.mkdir(exist_ok=True)
    save_npz(path.with_suffix(".npz"),
             layers=np.array([r["layer"] for r in measurements]),
             arms=np.array([r["arm"] for r in measurements]), candidates=np.array(e["candidates"]),
             candidate_ids=np.array(e["candidate_ids"]),
             base_token_ids=e["base"]["input_ids"].detach().cpu().numpy(),
             donor_token_ids=e["donor"]["input_ids"].detach().cpu().numpy(),
             base_sites=e["base_sites"], donor_sites=e["donor_sites"],
             patch_logits=np.array([r["candidate_logits"] for r in measurements]),
             patch_probabilities=np.array([r["candidate_probabilities"] for r in measurements]),
             patch_margins=np.array([r["candidate_margin"] for r in measurements]),
             prediction_ids=np.array([r["prediction_id"] for r in measurements]),
             base_probabilities=e["baseline"]["base"]["candidate_probabilities"],
             donor_probabilities=e["baseline"]["donor"]["candidate_probabilities"])
    write_json(path.with_suffix(".json"), record)
    return record


def load_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = args.model
    if path is None:
        settings = Path(args.population) / "settings.json"
        if not settings.exists():
            raise ValueError("provide --model /local/model, or an existing --population with settings.json")
        path = json.loads(settings.read_text())["model"]
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
             torch_dtype=getattr(torch, args.dtype), attn_implementation="sdpa")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    return model.to(args.device).eval().requires_grad_(False), tokenizer, str(Path(path).resolve())


def run(args, model=None, tokenizer=None):
    if args.mode == "symbolic":
        print(json.dumps(symbolic_example(), indent=2))
        return symbolic_example()
    if min(args.samples, args.max_tokens, args.cpu_threads) < 1:
        raise ValueError("positive sample/context/CPU settings required")
    if args.mode == "dcm" and (args.layer is None or args.resume):
        raise ValueError("DCM requires explicit --layer; --resume is for layer scans only")
    if args.mode == "dcm" and (args.rank < 1 or args.epochs < 1 or args.lr <= 0 or args.sparsity < 0):
        raise ValueError("invalid DCM settings")
    pairs = load_pairs(args.pairs) if args.pairs else make_pairs(args.samples, args.seed, args.experiment)
    if not args.output:
        raise ValueError("--output is required for a measured experiment")
    out = Path(args.output)
    if out.exists() and not args.resume:
        raise FileExistsError("output already exists; choose a new directory or --resume for a scan")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    if model is None:
        model, tokenizer, model_name = load_model(args)
    else:
        if tokenizer is None:
            raise ValueError("injected model requires a tokenizer")
        model.eval().requires_grad_(False)
        model_name = "injected_model_for_software_test"
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("supported architecture has model.model.layers (dense Llama-style)")
    layers = [args.layer] if args.mode == "dcm" else (args.layers or list(range(len(model.model.layers))))
    if len(set(layers)) != len(layers) or any(not 0 <= i < len(model.model.layers) for i in layers):
        raise ValueError("distinct zero-based layers within the loaded model required")
    config = {k: v for k, v in vars(args).items() if k not in ("resume", "output")}
    config.update(version=VERSION, resolved_model=model_name, resolved_layers=layers,
                  dataset="independent controlled pairs, not RAGTruth", lm_weights_updated=False)
    if out.exists():
        if not args.resume or not (out / "config.json").exists():
            raise FileExistsError("use a new output directory, or --resume for an identical scan")
        if json.loads((out / "config.json").read_text()) != config:
            raise ValueError("resume settings differ")
        if load_pairs(out / "pairs.jsonl") != pairs:
            raise ValueError("resume input pairs differ")
    else:
        out.mkdir(parents=True)
        write_json(out / "config.json", config)
        (out / "pairs.jsonl").write_text("".join(json.dumps(p.to_dict(), ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
    (out / "complete.json").unlink(missing_ok=True)
    if args.mode == "scan":
        records = []
        for index, pair in enumerate(pairs):
            saved = out / "samples" / (pair.id + ".json")
            if args.resume and saved.exists() and saved.with_suffix(".npz").exists():
                records.append(json.loads(saved.read_text()))
                print(f"reuse {index + 1}/{len(pairs)} {pair.id}", flush=True)
                continue
            e = prepare(model, tokenizer, pair, layers, args)
            measures = []
            with torch.no_grad():
                for layer in layers:
                    z = patch_logits(model, e["base"], layer, e["base_sites"], e["donor_states"][layer],
                                     restore=e["restore_states"])
                    measures.append(dict(layer=layer, arm="full_state",
                                         **summarize_logits(z, e["candidate_ids"], e["candidates"], tokenizer)))
            records.append(save_measurements(out, e, measures))
            print(f"scan {index + 1}/{len(pairs)} {pair.id}: both-correct={e['baseline']['both_correct']}", flush=True)
            del e
        report = summarize(records)
    else:
        # Split BEFORE filtering. No validation activations enter the SVD or mask fit.
        partition = source_split(pairs, seed=args.seed)
        prepared = []
        for index, pair in enumerate(pairs):
            if args.hypothesis not in pair.hypotheses:
                raise ValueError(f"{pair.id}: choose --hypothesis from {list(pair.hypotheses)}")
            e = prepare(model, tokenizer, pair, layers, args)
            e["donor_state"] = e["donor_states"][args.layer]
            prepared.append(e)
            print(f"DCM baseline {index + 1}/{len(pairs)}: {partition[pair.id]} both-correct={e['baseline']['both_correct']}", flush=True)
        train = [e for e in prepared if partition[e["pair"].id] == "train"
                 and (args.include_errors or e["baseline"]["both_correct"])]
        validation = [e for e in prepared if partition[e["pair"].id] == "validation"]
        write_json(out / "split.json", dict(partition=partition, training_ids=[e["pair"].id for e in train],
                                            include_errors=args.include_errors))
        if not train:
            raise ValueError("no both-correct TRAIN pairs; no DCM fit performed (see split.json)")
        basis = svd_basis([state for e in train for state in (e["base_states"][args.layer], e["donor_state"])], args.rank)
        soft_mask, losses = fit_mask(model, train, basis, args.layer, args.hypothesis,
                                     epochs=args.epochs, learning_rate=args.lr, sparsity=args.sparsity, seed=args.seed)
        binary = soft_mask.round()
        generator = torch.Generator().manual_seed(args.seed)
        random_mask = torch.zeros_like(binary)
        random_mask[torch.randperm(len(binary), generator=generator)[:int(binary.sum())]] = 1
        save_npz(out / "subspace.npz", basis=basis.numpy(), mask=soft_mask.numpy(),
                 binary_mask=binary.numpy(), random_rank_matched_mask=random_mask.numpy(), loss=np.array(losses))
        records = []
        with torch.no_grad():
            for e in validation:
                measures = []
                for arm, mask in (("full_state", None), ("dcm", binary), ("zero", torch.zeros_like(binary)),
                                  ("rank_matched_random", random_mask)):
                    z = patch_logits(model, e["base"], args.layer, e["base_sites"], e["donor_state"],
                                     None if mask is None else basis, mask, e["restore_states"])
                    measures.append(dict(layer=args.layer, arm=arm,
                                         **summarize_logits(z, e["candidate_ids"], e["candidates"], tokenizer)))
                records.append(save_measurements(out, e, measures, prefix="validation_"))
        report = summarize(records)
        report.update(train_examples=len(train), validation_examples=len(validation),
                      requested_basis_rank=args.rank, effective_basis_rank=len(basis), selected_rank=int(binary.sum()),
                      training="negative intervention-target logit + L1 mask; semantic supervision, not unsupervised detection")
    write_json(out / "summary.json", report)
    if report["curves"]:
        with (out / "curves.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report["curves"][0]))
            writer.writeheader(); writer.writerows(report["curves"])
    write_json(out / "complete.json", dict(complete=True, mode=args.mode, samples=len(pairs), lm_weights_updated=False))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("symbolic", "scan", "dcm"), default="scan")
    p.add_argument("--experiment", choices=("answer", "binding"), default="answer")
    p.add_argument("--model", help="existing local model directory; nothing is downloaded")
    p.add_argument("--population", default="outputs/ragtruth_population_20260912", help="only used to resolve the observer model path")
    p.add_argument("--pairs", help="custom Pair-schema JSONL; overrides the synthetic generator")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--output")
    p.add_argument("--layers", type=int, nargs="+", help="scan indices; default all physical layers")
    p.add_argument("--layer", type=int, help="one explicitly chosen DCM layer, zero-based")
    p.add_argument("--hypothesis", default="pointer", help="DCM semantic target: pointer/payload/binding_redirect/custom")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=.1)
    p.add_argument("--sparsity", type=float, default=.1)
    p.add_argument("--include-errors", action="store_true", help="DCM ablation: do not filter TRAIN pairs by both-correct")
    p.add_argument("--chat-template", action="store_true", help="explicitly wrap raw prompts using the tokenizer chat template")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--resume", action="store_true", help="reuse completed pairs in an identical scan")
    return p


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
