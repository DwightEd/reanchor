"""Command-line entry point for the constraint-control experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reanchor.experiment import ConstraintControlExperiment, ExperimentConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure whether a wrong commitment suppresses source constraint control."
    )
    parser.add_argument("--input", type=Path, required=True, help="paired event JSONL")
    parser.add_argument("--output", type=Path, required=True, help="run directory")
    parser.add_argument("--model", type=Path, required=True, help="local Hugging Face model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = ExperimentConfig(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        device=args.device,
        dtype=args.dtype,
        show_progress=not args.no_progress,
    )
    result = ConstraintControlExperiment(config).run()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
