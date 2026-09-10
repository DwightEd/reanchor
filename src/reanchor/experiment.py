"""End-to-end factorial constraint-control experiment."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from reanchor import __version__
from reanchor.data import CounterfactualEvent, load_events
from reanchor.effects import BranchMargins, compute_effects
from reanchor.model import CausalLMScorer


@dataclass(frozen=True)
class ExperimentConfig:
    input_path: Path
    output_path: Path
    model_path: Path
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    system_prompt: str = "Answer from the supplied facts and preserve every stated condition."
    show_progress: bool = True


class ConstraintControlExperiment:
    """Measure whether source control survives a conflicting generated commitment."""

    def __init__(self, config: ExperimentConfig, *, scorer=None) -> None:
        self.config = config
        self.scorer = scorer

    def run(self) -> dict:
        if self.config.output_path.is_dir() and any(self.config.output_path.iterdir()):
            raise FileExistsError(f"output directory is not empty: {self.config.output_path}")
        events = load_events(self.config.input_path)
        scorer = self.scorer or CausalLMScorer(
            self.config.model_path,
            device=self.config.device,
            dtype=self.config.dtype,
            system_prompt=self.config.system_prompt,
        )
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.config.input_path, self.config.output_path / "input.jsonl")

        records = []
        output_file = self.config.output_path / "events.jsonl"
        with output_file.open("w", encoding="utf-8") as stream:
            for event in self._progress(events):
                record = self._measure(event, scorer)
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

        summary = {
            "schema": "reanchor/constraint-control-summary@1",
            "software_version": __version__,
            "events": len(records),
            "input": {
                "path": str(self.config.input_path.resolve()),
                "snapshot": "input.jsonl",
            },
            "model": dict(scorer.metadata),
            "settings": {
                "system_prompt": self.config.system_prompt,
                "dtype": self.config.dtype,
                "device": self.config.device,
            },
            "mean_effects": {
                name: sum(record["effects"][name] for record in records) / len(records)
                for name in records[0]["effects"]
            },
        }
        (self.config.output_path / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return summary

    def _measure(self, event: CounterfactualEvent, scorer) -> dict:
        prompt_a = scorer.render_prompt(event.prompt_a) + event.answer_prefix
        prompt_b = scorer.render_prompt(event.prompt_b) + event.answer_prefix
        contexts = (
            prompt_a,
            prompt_b,
            prompt_a + event.option_a + event.followup,
            prompt_a + event.option_b + event.followup,
            prompt_b + event.option_a + event.followup,
            prompt_b + event.option_b + event.followup,
        )
        pairs = [
            (context, option) for context in contexts for option in (event.option_a, event.option_b)
        ]
        scores = scorer.score(pairs)
        branch_names = (
            "onset_a",
            "onset_b",
            "world_a_after_a",
            "world_a_after_b",
            "world_b_after_a",
            "world_b_after_b",
        )
        logp = {
            name: scores[index : index + 2]
            for name, index in zip(branch_names, range(0, len(scores), 2), strict=True)
        }
        margins = BranchMargins(*(values[0] - values[1] for values in logp.values()))
        effects = compute_effects(margins)
        return {
            "schema": "reanchor/constraint-control@2",
            "event_id": event.event_id,
            "source_id": event.source_id,
            "split": event.split,
            "relation": event.relation,
            "options": {"a": event.option_a, "b": event.option_b},
            "onset_preference": {
                "world_a": "a" if margins.onset_a >= 0 else "b",
                "world_b": "a" if margins.onset_b >= 0 else "b",
            },
            "logp": logp,
            "margins": margins.to_dict(),
            "effects": effects.to_dict(),
        }

    def _progress(self, events: tuple[CounterfactualEvent, ...]):
        if not self.config.show_progress:
            return events
        from tqdm.auto import tqdm

        return tqdm(events, desc="constraint-control events", unit="event")
