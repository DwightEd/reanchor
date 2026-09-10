"""Evaluate the same signals for hallucination onset and span continuation."""

from __future__ import annotations

from dataclasses import dataclass

from .binary import BinaryEvaluator


@dataclass(frozen=True)
class DetectionEvaluator:
    bootstrap: int = 1000
    seed: int = 0

    def evaluate(
        self, rows: list[dict], scores: tuple[str, ...], *, phase: str = "phase"
    ) -> dict:
        groups = {"overall": rows}
        for row in rows:
            name = f"{row['split']}/{row['task']}"
            groups.setdefault(name, []).append(row)
        result = {
            name: self._cohorts(group_rows, scores, phase)
            for name, group_rows in groups.items()
        }
        return {
            "evaluation_schema": "reanchor/source-balanced-detection@1",
            "positive_class": "hallucinated target token",
            "score_direction": "larger score predicts the positive class",
            "cohorts": {
                "hallucination": "all known N/H target tokens",
                "onset": "N targets versus N-to-H onset targets",
                "continuing": "N targets versus H-to-H continuing targets",
            },
            "overall": result.pop("overall"),
            "by_split_task": result,
        }

    def _cohorts(self, rows, scores, phase):
        cohorts = {
            "hallucination": [row for row in rows if row.get("label") in (0, 1)],
            "onset": [row for row in rows if row.get(phase) in ("normal", "onset")],
            "continuing": [
                row for row in rows if row.get(phase) in ("normal", "continuing")
            ],
        }
        evaluator = BinaryEvaluator(self.bootstrap, self.seed)
        return {
            cohort: {score: evaluator.evaluate(items, score) for score in scores}
            for cohort, items in cohorts.items()
        }
