"""Explicit G0 stages. Only evaluate opens hallucination annotations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from functools import partial
from pathlib import Path


def runtime(parser):
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-chunk", type=int, default=256)


def backbone_options(parser):
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=32)


def training_options(parser):
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)


def training_args(args):
    return {
        key: getattr(args, key)
        for key in (
            "device",
            "width",
            "blocks",
            "epochs",
            "batch_size",
            "learning_rate",
            "seed",
            "candidate_chunk",
        )
    }


def parser():
    root = argparse.ArgumentParser(
        description="Conditional binding G0 support model", allow_abbrev=False
    )
    commands = root.add_subparsers(
        dest="command",
        required=True,
        parser_class=partial(argparse.ArgumentParser, allow_abbrev=False),
    )
    commands.add_parser("audit", help="legacy factorial audit; audit --help for options")
    sample = commands.add_parser("sample", help="generate answers and save states during decoding")
    sample.add_argument("--dataset", type=Path, required=True)
    sample.add_argument("--source-ids", nargs="+", required=True)
    sample.add_argument("--model", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    sample.add_argument("--max-new-tokens", type=int, default=512)
    sample.add_argument("--temperature", type=float, default=0.7)
    sample.add_argument("--top-p", type=float, default=0.9)
    sample.add_argument("--device", default="cuda:0")
    sample.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    attention = commands.add_parser(
        "analyze-attention", help="write per-token/head WAAD and FAI CSV"
    )
    attention.add_argument("--samples", type=Path, required=True)
    attention.add_argument("--output", type=Path, required=True)
    attention.add_argument("--window", type=int, default=10)
    attention.add_argument("--future-range", type=int, nargs=2, default=[10, 100])
    spans = commands.add_parser(
        "evaluate-spans", help="evaluate token scores at onsets and continuations"
    )
    spans.add_argument(
        "--input", type=Path, required=True, help="complete labeled token scores CSV"
    )
    spans.add_argument("--output", type=Path, required=True)
    spans.add_argument("--bootstrap", type=int, default=200)
    spans.add_argument("--seed", type=int, default=42)
    check = commands.add_parser("preflight", help="dependencies/device and seeded witness")
    check.add_argument("--device", default="cuda:0")
    check.add_argument("--model", type=Path)
    check.add_argument("--allow-busy-gpu", action="store_true")
    generate = commands.add_parser("generate", help="source-disjoint paired program tasks")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--sources", type=int, default=64)
    generate.add_argument("--records", type=int, default=6)
    generate.add_argument("--seed", type=int, default=42)
    extract = commands.add_parser("extract", help="frozen causal features without H labels")
    extract.add_argument("--input", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--no-progress", action="store_true")
    runtime(extract)
    backbone_options(extract)
    train = commands.add_parser("train", help="softmax(f) training on program targets")
    train.add_argument("--train-cache", type=Path, required=True)
    train.add_argument("--dev-cache", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    runtime(train)
    training_options(train)
    score = commands.add_parser("score", help="freeze raw G0 scores, no labels")
    score.add_argument("--cache", type=Path, required=True)
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    runtime(score)
    cal = commands.add_parser("calibrate", help="unlabeled mixed-reference ranks")
    cal.add_argument("--reference", type=Path, required=True)
    cal.add_argument("--scores", type=Path, required=True)
    cal.add_argument("--output", type=Path, required=True)
    cal.add_argument("--min-sources", type=int, default=50)
    prepare = commands.add_parser("prepare-ragtruth", help="normalize with labels stripped")
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--limit-sources", type=int, default=32, help="0 means all")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--calibration-fraction", type=float, default=0.25)
    prepare.add_argument("--generator")
    prepare.add_argument("--exclude-sources", type=Path)
    evaluate = commands.add_parser("evaluate", help="ONLY HERE join frozen scores with H spans")
    evaluate.add_argument("--scores", type=Path, required=True)
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--bootstrap", type=int, default=200)
    evaluate.add_argument("--seed", type=int, default=42)
    pipeline = commands.add_parser(
        "pipeline", help="program train/test + optional natural evaluation"
    )
    pipeline.add_argument("--output", type=Path, required=True)
    pipeline.add_argument("--sources", type=int, default=64)
    pipeline.add_argument("--records", type=int, default=6)
    pipeline.add_argument("--ragtruth", type=Path, help="omit for program-only engineering pilot")
    pipeline.add_argument("--ragtruth-sources", type=int, default=32)
    pipeline.add_argument("--generator")
    pipeline.add_argument("--exclude-sources", type=Path)
    pipeline.add_argument("--bootstrap", type=int, default=200)
    pipeline.add_argument("--min-calibration-sources", type=int, default=50)
    pipeline.add_argument("--no-progress", action="store_true")
    pipeline.add_argument("--allow-busy-gpu", action="store_true")
    runtime(pipeline)
    backbone_options(pipeline)
    training_options(pipeline)
    return root


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and (argv[0] == "audit" or argv[0] == "--input"):
        from reanchor.audit_cli import main as audit_main

        return audit_main(argv[1:] if argv[0] == "audit" else argv)
    args = parser().parse_args(argv)
    if args.command == "sample":
        from reanchor.sampling import SamplingConfig, SamplingExperiment

        rows = SamplingExperiment(
            SamplingConfig(
                dataset=args.dataset,
                source_ids=tuple(args.source_ids),
                model=args.model,
                output=args.output,
                seeds=tuple(args.seeds),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=args.device,
                dtype=args.dtype,
            )
        ).run()
        writer = csv.DictWriter(
            sys.stdout, ["source_id", "seed", "tokens", "stop_reason"], extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
        return
    if args.command == "analyze-attention":
        from reanchor.attention import AttentionAnalysis

        count = AttentionAnalysis(
            args.samples, args.output, args.window, tuple(args.future_range)
        ).run()
        print(f"rows\n{count}")
        return
    if args.command == "evaluate-spans":
        from reanchor.span_evaluation import SpanEvaluation

        SpanEvaluation(args.input, args.output, args.bootstrap, args.seed).run()
        print(args.output.read_text(encoding="utf-8"), end="")
        return
    if args.command == "preflight":
        from reanchor.runner import preflight

        result = preflight(args.device, args.model, args.allow_busy_gpu)
    elif args.command == "generate":
        from reanchor.binding_data import generate_dataset

        result = generate_dataset(args.output, args.sources, args.seed, args.records)
    elif args.command == "extract":
        from reanchor.features import FrozenBackbone, extract_features

        result = extract_features(
            args.input,
            args.output,
            FrozenBackbone(args.model, args.device, args.dtype, args.max_tokens),
            args.top_k,
            not args.no_progress,
        )
    elif args.command == "train":
        from reanchor.training import train_head

        result = train_head(args.train_cache, args.dev_cache, args.output, **training_args(args))
    elif args.command == "score":
        from reanchor.training import score_head

        result = score_head(
            args.cache,
            args.checkpoint,
            args.output,
            device=args.device,
            candidate_chunk=args.candidate_chunk,
        )
    elif args.command == "calibrate":
        from reanchor.calibration import calibrate_scores

        result = calibrate_scores(args.reference, args.scores, args.output, args.min_sources)
    elif args.command == "prepare-ragtruth":
        from reanchor.ragtruth import prepare_ragtruth

        result = prepare_ragtruth(
            args.dataset,
            args.output,
            seed=args.seed,
            limit_sources=args.limit_sources,
            calibration_fraction=args.calibration_fraction,
            model_filter=args.generator,
            exclude_sources=args.exclude_sources,
        )
    elif args.command == "evaluate":
        from reanchor.evaluation import evaluate_ragtruth

        result = evaluate_ragtruth(
            args.scores, args.dataset, args.output, bootstrap=args.bootstrap, seed=args.seed
        )
    else:
        from reanchor.runner import run_pipeline

        result = run_pipeline(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
