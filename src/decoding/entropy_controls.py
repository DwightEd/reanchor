"""Compare first-error entropy with explicitly matched normal tokens."""

import csv
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


def match_onsets(rows: list[dict]) -> list[dict]:
    normal = defaultdict(list)
    for row in rows:
        if not row["label"]:
            normal[row["token_id"]].append(row)
    pairs = []
    for row in rows:
        if not row["first_error"]:
            continue
        controls = normal[row["token_id"]]
        groups = {
            "same_token": controls,
            "same_response_token": [c for c in controls if c["response_id"] == row["response_id"]],
            "same_token_clause_start": [
                c for c in controls if row["boundary"] and c["boundary"] and c["clean_clause"]
            ],
        }
        for name, candidates in groups.items():
            if not candidates:
                continue
            # Equal weight per source prevents long answers dominating the control.
            by_source = defaultdict(list)
            for candidate in candidates:
                by_source[candidate["source_id"]].append(candidate["entropy"])
            baseline = float(np.mean([np.mean(v) for v in by_source.values()]))
            pairs.append(
                dict(
                    control=name,
                    response_id=row["response_id"],
                    source_id=row["source_id"],
                    token_index=row["token_index"],
                    token_id=row["token_id"],
                    error_entropy=row["entropy"],
                    normal_entropy=baseline,
                    difference=row["entropy"] - baseline,
                    normal_tokens=len(candidates),
                    normal_sources=len(by_source),
                )
            )
    return pairs


class EntropyControls:
    def __init__(self, features: Path, labels: Path, output: Path):
        self.features, self.labels, self.output = features, labels, output

    def run(self) -> dict:
        self.output.mkdir(parents=True, exist_ok=False)
        grouped = defaultdict(list)
        opener = gzip.open if self.features.suffix == ".gz" else open
        with opener(self.features, "rt", encoding="utf-8") as stream:
            for line in tqdm(stream, desc="token controls", unit="token"):
                row = json.loads(line)
                if row["schema"] != "route-graph/features@1":
                    raise ValueError("expected native route-graph features, entropy in nats")
                for field in ("observed", "null", "residual", "signal"):
                    row.pop(field, None)
                grouped[row["response_id"]].append(row)
        with self.labels.open(encoding="utf-8") as stream:
            annotations = {str(r["id"]): r for r in map(json.loads, stream)}
        labeled, incomplete = [], 0
        for key, tokens in grouped.items():
            count = tokens[0]["token_count"]
            if len(tokens) != count:
                incomplete += 1
                continue
            if [r["token_index"] for r in tokens] != list(range(count)):
                raise ValueError(f"{key}: duplicate or unordered token positions")
            if tokens[0]["split"] == "test":
                labeled.extend(label_boundaries(tokens, annotations[key]))
        pairs = match_onsets(labeled)
        result = dict(
            first_errors=sum(r["first_error"] for r in labeled),
            incomplete_responses=incomplete,
            entropy_unit="nats",
            controls={},
        )
        for name in ["same_token", "same_response_token", "same_token_clause_start"]:
            selected = [p for p in pairs if p["control"] == name]
            result["controls"][name] = dict(
                matched=len(selected),
                mean_difference=float(np.mean([p["difference"] for p in selected]))
                if selected
                else None,
                fraction_higher=float(np.mean([p["difference"] > 0 for p in selected]))
                if selected
                else None,
            )
        if pairs:
            with (self.output / "pairs.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
                writer.writeheader()
                writer.writerows(pairs)
        (self.output / "numbers.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return result


def label_boundaries(tokens: list[dict], annotation: dict) -> list[dict]:
    text, spans = annotation["response"], annotation["labels"]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    starts = sorted(
        {
            m.end()
            for m in re.finditer(r"(?:^|(?<!\d)[.!?;:](?!\d)|\n)\s*", text)
            if m.end() < len(text)
        }
    )
    ends = starts[1:] + [len(text)]
    result, seen = [], False
    for token in tokens:
        if token["response_sha256"] != digest or token["source_id"] != str(annotation["source_id"]):
            raise ValueError("annotation text or source does not match captured response")
        left, right = token["char_span"]
        if not 0 <= left < right <= len(text) or not np.isfinite(token["entropy"]):
            raise ValueError("invalid token interval or entropy")
        label = any(left < s["end"] and right > s["start"] for s in spans)
        clauses = [(a, b) for a, b in zip(starts, ends) if left <= a < right]
        clean = bool(clauses) and all(
            not any(a < s["end"] and b > s["start"] for s in spans) for a, b in clauses
        )
        result.append(
            {
                **token,
                "label": label,
                "first_error": label and not seen,
                "boundary": bool(clauses),
                "clean_clause": clean,
            }
        )
        seen |= label
    return result
