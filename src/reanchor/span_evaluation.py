"""Evaluate fixed token scores separately at error starts and continuations."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from reanchor.evaluation import bootstrap_metrics


class SpanEvaluation:
    def __init__(self, input_path: Path, output: Path, bootstrap: int, seed: int):
        self.input_path, self.output = input_path, output
        self.bootstrap, self.seed = bootstrap, seed

    def run(self) -> None:
        if self.bootstrap < 0:
            raise ValueError("bootstrap must be nonnegative")
        grouped = defaultdict(list)
        with self.input_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                for name in ("token_index", "token_count", "label"):
                    row[name] = int(row[name])
                row["score"] = float(row["score"])
                if row["label"] not in (0, 1) or not np.isfinite(row["score"]):
                    raise ValueError("expected binary labels and finite scores")
                grouped[row["response_id"]].append(row)
        if not grouped:
            raise ValueError("no token scores")
        rows = []
        for tokens in grouped.values():
            tokens.sort(key=lambda row: row["token_index"])
            if [r["token_index"] for r in tokens] != list(range(tokens[0]["token_count"])) or any(
                r["token_count"] != len(tokens) for r in tokens
            ):
                raise ValueError("evaluation requires complete response token scores")
            if len({r["source_id"] for r in tokens}) != 1:
                raise ValueError("response belongs to multiple sources")
            previous, seen_error = 0, False
            for row in tokens:
                y = row["label"]
                row["onset"] = bool(y and not previous)
                row["first_error"] = bool(y and not seen_error)
                row["continuation"] = bool(y and previous)
                row["clean_prefix"] = not seen_error
                previous, seen_error = y, seen_error or bool(y)
                rows.append(row)
        cohorts = {
            "all": rows,
            "onsets": [r for r in rows if not r["label"] or r["onset"]],
            "first_error": [r for r in rows if not r["label"] or r["first_error"]],
            "continuations": [r for r in rows if not r["label"] or r["continuation"]],
            "first_error_clean_prefix": [r for r in rows if r["clean_prefix"]],
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "subset",
                    "tokens",
                    "positive",
                    "sources",
                    "prevalence",
                    "auroc",
                    "auroc_low",
                    "auroc_high",
                    "ap",
                    "ap_low",
                    "ap_high",
                ]
            )
            for name, cohort in cohorts.items():
                value = bootstrap_metrics(cohort, "label", "score", self.bootstrap, self.seed)
                writer.writerow(
                    [
                        name,
                        value["tokens"],
                        value["positive"],
                        value.get("sources", 0),
                        value["prevalence"],
                        value["auroc"],
                        *value.get("auroc_ci95", (None, None)),
                        value["ap"],
                        *value.get("ap_ci95", (None, None)),
                    ]
                )
