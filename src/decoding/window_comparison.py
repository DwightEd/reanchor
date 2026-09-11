"""Join independently specified example windows to already computed scores."""

import csv
from pathlib import Path

import numpy as np

from decoding.io import read_jsonl
from decoding.revisits import write_rows


class WindowComparison:
    def __init__(self, analysis: Path, cases: Path, output: Path):
        self.analysis, self.cases, self.output = analysis, cases, output

    def run(self) -> None:
        with self.cases.open(encoding="utf8", newline="") as stream:
            cases = list(csv.DictReader(stream))
        samples = list(read_jsonl(self.analysis / "samples.jsonl"))
        rows = []
        for case in cases:
            matches = [
                s
                for s in samples
                if str(s["source_id"]) == case["source_id"] and s["seed"] == int(case["seed"])
            ]
            if len(matches) != 1:
                raise ValueError(f"{case['case']}: expected exactly one source/seed match")
            rows.extend(self._window(case, matches[0]["trace"]))
        columns = (
            "case source_id seed judgment step offset query token event active revisit "
            "top2_margin logit_entropy hops route_divergence relation_residual "
            "relation_shuffled relation_coverage valid_layers"
        )
        write_rows(self.output, columns, rows)

    def _window(self, case, trace):
        directory = self.analysis / Path(trace).stem
        with (directory / "tokens.csv").open(encoding="utf8", newline="") as stream:
            tokens = list(csv.DictReader(stream))
        start, stop = int(case["start"]), int(case["stop"])
        if not 0 <= start < stop <= len(tokens):
            raise ValueError(f"{case['case']}: window is outside the response")
        if "".join(r["token"] for r in tokens[start:stop]) != case["text"]:
            raise ValueError(f"{case['case']}: window text differs from captured tokens")
        grouped = {}
        with (directory / "paths.csv").open(encoding="utf8", newline="") as stream:
            for row in csv.DictReader(stream):
                t, hop = int(row["step"]), int(row["hops"])
                if start <= t < stop:
                    grouped.setdefault((t, hop), []).append(row)
        for (t, hop), paths in sorted(grouped.items()):
            token = tokens[t]
            values = [
                self._mean(paths, name)
                for name in (
                    "route_divergence",
                    "relation_residual",
                    "relation_shuffled",
                    "relation_coverage",
                )
            ]
            yield [
                case["case"],
                case["source_id"],
                case["seed"],
                case.get("judgment", ""),
                t,
                t - start,
                token["query"],
                token["token"],
                token["event"],
                token["active"],
                token["revisit"],
                token["top2_margin"],
                token["logit_entropy"],
                hop,
                *values,
                sum(bool(r["relation_residual"]) for r in paths),
            ]

    @staticmethod
    def _mean(rows, field):
        values = [float(r[field]) for r in rows if r[field]]
        return float(np.mean(values)) if values else np.nan
