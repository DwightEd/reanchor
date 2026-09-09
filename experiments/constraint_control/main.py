"""Command-line entry point for free-run constraint-control capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

from .config import ExperimentConfig
from .experiment import ConstraintControlExperiment
from .generation import SamplingConfig


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        prog="constraint-control",
        description="Sample answers freely, replay them, and capture inspectable trajectories.",
    )
    command.add_argument("--input", type=Path, required=True, help="source-grouped JSONL")
    command.add_argument("--output", type=Path, required=True, help="new run directory")
    command.add_argument("--model", required=True, help="Hugging Face model name or path")
    command.add_argument("--revision", help="optional immutable model revision")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--dtype", default="auto")
    command.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    command.add_argument("--max-new-tokens", type=int, default=64)
    command.add_argument("--temperature", type=float, default=0.7)
    command.add_argument("--top-p", type=float, default=0.9)
    command.add_argument("--top-k", type=int, default=20)
    command.add_argument("--trace-top-k", type=int, default=20)
    command.add_argument("--replay-atol", type=float, default=0.05)
    command.add_argument("--max-samples", type=int)
    return command


def main(argv: list[str] | None = None) -> None:
    arguments = parser().parse_args(argv)
    samplings = tuple(
        SamplingConfig(
            seed=seed,
            max_new_tokens=arguments.max_new_tokens,
            temperature=arguments.temperature,
            top_p=arguments.top_p,
            top_k=arguments.top_k,
            trace_top_k=arguments.trace_top_k,
        )
        for seed in arguments.seeds
    )
    config = ExperimentConfig(
        input_path=arguments.input,
        output=arguments.output,
        model=arguments.model,
        samplings=samplings,
        device=arguments.device,
        dtype=arguments.dtype,
        revision=arguments.revision,
        replay_atol=arguments.replay_atol,
        max_samples=arguments.max_samples,
    )

    with tqdm(desc="capture", unit="trajectory", dynamic_ncols=True) as bar:

        def progress(completed: int, total: int, sample: str) -> None:
            if bar.total != total:
                bar.total = total
            if sample:
                bar.set_postfix_str(sample, refresh=False)
            if completed > bar.n:
                bar.update(completed - bar.n)
            else:
                bar.refresh()

        result = ConstraintControlExperiment(config, progress=progress).run()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
