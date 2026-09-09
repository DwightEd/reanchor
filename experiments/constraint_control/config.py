"""Explicit runtime configuration for the constraint-control experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .generation import SamplingConfig


@dataclass(frozen=True)
class ExperimentConfig:
    input_path: Path
    output: Path
    model: str
    samplings: tuple[SamplingConfig, ...]
    input_format: Literal["jsonl", "ragtruth"] = "jsonl"
    task: str = "all"
    split: str = "all"
    device: str = "cuda:0"
    dtype: str = "auto"
    revision: str | None = None
    replay_atol: float = 0.05
    max_samples: int | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be nonempty")
        if not self.samplings:
            raise ValueError("at least one sampling configuration is required")
        if self.input_format not in {"jsonl", "ragtruth"}:
            raise ValueError("input_format must be jsonl or ragtruth")
        if self.input_format == "jsonl" and (self.task != "all" or self.split != "all"):
            raise ValueError("task and split filtering are only available for RAGTruth input")
        if self.task not in {"all", "QA", "Summary", "Data2txt"}:
            raise ValueError("unsupported RAGTruth task")
        if self.split not in {"all", "train", "test"}:
            raise ValueError("RAGTruth split must be all, train, or test")
        seeds = [sampling.seed for sampling in self.samplings]
        if len(seeds) != len(set(seeds)):
            raise ValueError("sampling seeds must be unique")
        if self.replay_atol < 0:
            raise ValueError("replay_atol must be nonnegative")
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError("max_samples must be positive")
