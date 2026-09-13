"""Post-hoc evaluation of whether both constructed values entered native top-k.

This consumes construction truth only AFTER frozen predictions. It does not
change candidates, detector scores, or the original numerical summary.
"""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def inspect(results):
    settings = json.loads((results / "settings.json").read_text())
    cases_path = results / "source/binding_cases.jsonl"
    assert hashlib.sha256(cases_path.read_bytes()).hexdigest() == settings["cases_sha256"]
    cases = [json.loads(line) for line in cases_path.read_text().splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(settings["model"], local_files_only=True)
    rows, groups = [], {}
    for case in cases[: settings["limit"]]:
        values = [tokenizer.encode(value, add_special_tokens=False) for value in case["values"]]
        if any(len(ids) != 1 for ids in values) or values[0] == values[1]:
            raise ValueError("expected distinct single-token constructed values")
        values = [ids[0] for ids in values]
        for layout in settings["layouts"]:
            for subject in range(2):
                for world in range(2):
                    name = f"{case['id']}_{layout}_{subject}_{world}.json"
                    row = json.loads((results / name).read_text())
                    expected = values[(subject + world) % 2]
                    if row["expected_token"] != expected:
                        raise ValueError("construction truth mismatch")
                    rows.append(
                        dict(
                            file=name,
                            layout=layout,
                            both_values_in_top4=set(values) <= set(row["candidates"]),
                            value_tokens=values,
                            candidate_text=row["candidate_text"],
                        )
                    )
    for layout in settings["layouts"]:
        selected = [row for row in rows if row["layout"] == layout]
        groups[layout] = dict(
            conditions=len(selected),
            both_values_in_top4=sum(row["both_values_in_top4"] for row in selected),
        )
    report = dict(
        schema="binding/constructed-value-coverage@1",
        stage="evaluation_only_after_frozen_predictions",
        scope="both constructed values, not merely two source-matching tokens",
        conditions=len(rows),
        both_values_in_top4=sum(row["both_values_in_top4"] for row in rows),
        layouts=groups,
        rows=rows,
    )
    target = results / "constructed_value_coverage.json"
    temporary = target.with_suffix(".partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(target)
    (results / "source/inspect_binding_coverage.py").write_bytes(Path(__file__).read_bytes())
    return {key: value for key, value in report.items() if key != "rows"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    print(json.dumps(inspect(parser.parse_args().results), indent=2))
