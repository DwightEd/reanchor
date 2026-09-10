"""One-process feature collection releases the frozen backbone before head training."""

from __future__ import annotations

import gc
import json
import math
import subprocess
import sys
from pathlib import Path

from reanchor.io import digest, empty_directory, file_digest, write_json


def preflight(device, model=None, allow_busy_gpu=False):
    import torch
    import transformers
    from packaging.version import Version

    if not Version("4.46") <= Version(transformers.__version__) < Version("4.58"):
        raise RuntimeError("requires transformers>=4.46,<4.58; install project dependencies")
    if model is not None and not (model / "config.json").is_file():
        raise FileNotFoundError(f"local model config missing: {model}; no download attempted")
    target = torch.device(device)
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; activate the intended GPU Python environment")
        free, total = torch.cuda.mem_get_info(target)
        used_mib = (total - free) / 2**20
        if used_mib >= 500 and not allow_busy_gpu:
            raise RuntimeError(
                f"GPU has {used_mib:.0f} MiB in use; choose a free GPU or explicitly "
                "pass --allow-busy-gpu. No process will be stopped."
            )
    generator = torch.Generator(device=target).manual_seed(42)
    x = torch.randn(8, 8, generator=generator, device=target)
    witness = x @ x
    if not torch.isfinite(witness).all():
        raise RuntimeError("non-finite seeded kernel witness")
    result = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": str(target),
        "witness_shape": list(witness.shape),
    }
    print("WITNESS " + json.dumps(result), flush=True)
    return result


def run_pipeline(args):
    import torch

    from reanchor.binding_data import generate_dataset
    from reanchor.calibration import calibrate_scores
    from reanchor.cli import training_args
    from reanchor.evaluation import evaluate_ragtruth
    from reanchor.features import FrozenBackbone, extract_features
    from reanchor.ragtruth import prepare_ragtruth
    from reanchor.training import score_head, train_head

    positive = (
        "epochs",
        "batch_size",
        "width",
        "blocks",
        "candidate_chunk",
        "max_tokens",
        "top_k",
        "min_calibration_sources",
    )
    if any(getattr(args, key) <= 0 for key in positive):
        raise ValueError(f"these options must be positive: {positive}")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if args.sources < 8 or args.records < 6 or args.ragtruth_sources < 0 or args.bootstrap < 0:
        raise ValueError("sources>=8, records>=6, ragtruth_sources/bootstrap>=0 required")
    environment = preflight(args.device, args.model, args.allow_busy_gpu)
    empty_directory(args.output)
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    repo = Path(__file__).resolve().parents[2]
    code_files = [
        repo / "main.py",
        repo / "pyproject.toml",
        *sorted((repo / "src").rglob("*.py")),
        *sorted((repo / "scripts").glob("*.sh")),
    ]
    code_digests = {p.relative_to(repo).as_posix(): file_digest(p) for p in code_files}
    write_json(
        args.output / "run.json",
        {
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "environment": environment,
            "git_revision": revision,
            "code_digests": code_digests,
            "code_tree_digest": digest(code_digests),
            "determinism": "seeded RNG; CUDA bitwise determinism not guaranteed",
            "method": "G0; no G1",
        },
    )
    generate_dataset(args.output / "program", args.sources, args.seed, args.records)
    stages = {"program": ("train", "dev", "calibration", "test")}
    if args.ragtruth:
        prepare_ragtruth(
            args.ragtruth,
            args.output / "natural",
            seed=args.seed,
            limit_sources=args.ragtruth_sources,
            model_filter=args.generator,
            exclude_sources=args.exclude_sources,
        )
        stages["natural"] = ("calibration", "test")
    backbone = FrozenBackbone(args.model, args.device, args.dtype, args.max_tokens)
    for domain, splits in stages.items():
        for split in splits:
            print(f"STAGE extract {domain}/{split}", flush=True)
            extract_features(
                args.output / domain / f"{split}.jsonl",
                args.output / "cache" / domain / split,
                backbone,
                args.top_k,
                not args.no_progress,
            )
    del backbone
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result = {"method": "G0", "natural_evaluation": "not_requested"}
    print("STAGE train G0", flush=True)
    result["training"] = train_head(
        args.output / "cache/program/train",
        args.output / "cache/program/dev",
        args.output / "head",
        **training_args(args),
    )
    for domain in stages:
        for split in ("calibration", "test"):
            print(f"STAGE score {domain}/{split}", flush=True)
            score_head(
                args.output / "cache" / domain / split,
                args.output / "head/head.pt",
                args.output / "scores" / domain / split,
                device=args.device,
                candidate_chunk=args.candidate_chunk,
            )
        calibrate_scores(
            args.output / "scores" / domain / "calibration",
            args.output / "scores" / domain / "test",
            args.output / "calibrated" / domain,
            args.min_calibration_sources,
        )
    if args.ragtruth:
        print("STAGE frozen-score label join (first label-reading stage)", flush=True)
        evaluate_ragtruth(
            args.output / "calibrated/natural",
            args.ragtruth,
            args.output / "evaluation",
            bootstrap=args.bootstrap,
            seed=args.seed,
        )
        result["natural_evaluation"] = "evaluation/summary.json"
    result["program_metrics"] = "scores/program/test/program_metrics.json"
    result["claim_boundary"] = "completed computation is not evidence of effectiveness"
    write_json(args.output / "summary.json", result)
    return result
