"""Visible orchestration for free-run trajectory capture."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

from reanchor.artifacts.store import ArtifactStore

from .config import ExperimentConfig
from .dataset import SourceDataset
from .generation import GenerationBackend, GenerationRecorder, ReplayMismatchError
from .model import HuggingFaceBackend
from .ragtruth import RagTruthDataset


class ConstraintControlExperiment:
    """Capture every configured source/seed trajectory and publish one manifest."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        backend: GenerationBackend | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ):
        self.config = config
        self.backend = backend
        self.progress = progress

    def run(self) -> dict[str, int]:
        if self.config.input_format == "ragtruth":
            dataset = RagTruthDataset(
                self.config.input_path,
                task=self.config.task,
                split=self.config.split,
            )
        else:
            dataset = SourceDataset(self.config.input_path)
        records = dataset.records[: self.config.max_samples]
        backend = self.backend or HuggingFaceBackend(
            self.config.model,
            device=self.config.device,
            dtype=self.config.dtype,
            revision=self.config.revision,
        )
        recorder = GenerationRecorder(
            backend,
            self.config.output,
            replay_atol=self.config.replay_atol,
        )
        total = len(records) * len(self.config.samplings)
        if self.progress:
            self.progress(0, total, "")
        completed_jobs = 0
        samples = []
        for record in records:
            for sampling in self.config.samplings:
                try:
                    artifact = recorder.run(record, sampling)
                except ReplayMismatchError as error:
                    samples.append(
                        {
                            "sample_key": record.key,
                            "source_id": record.source_id,
                            "seed": sampling.seed,
                            "status": "rejected",
                            "reason": str(error),
                        }
                    )
                else:
                    samples.append(
                        {
                            "sample_key": artifact.sample_key,
                            "source_id": record.source_id,
                            "seed": sampling.seed,
                            "status": "completed",
                            "trajectory": self._relative(artifact.metadata_path),
                            "capture": self._relative(artifact.capture_path),
                            "response_tokens": artifact.response_tokens,
                        }
                    )
                completed_jobs += 1
                if self.progress:
                    self.progress(
                        completed_jobs,
                        total,
                        f"{record.key} seed={sampling.seed}",
                    )

        summary = {
            "planned": len(records) * len(self.config.samplings),
            "completed": sum(sample["status"] == "completed" for sample in samples),
            "rejected": sum(sample["status"] == "rejected" for sample in samples),
        }
        ArtifactStore(self.config.output).write_json(
            self.config.output / "index.json",
            {
                "schema": "constraint_control_run_v1",
                "labels_used_for_capture": False,
                "input": str(self.config.input_path.resolve()),
                "input_format": self.config.input_format,
                "task": self.config.task,
                "split": self.config.split,
                "model": dict(backend.metadata),
                "samplings": [asdict(sampling) for sampling in self.config.samplings],
                "replay_atol": self.config.replay_atol,
                "summary": summary,
                "samples": samples,
            },
        )
        return summary

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.config.output).as_posix()
