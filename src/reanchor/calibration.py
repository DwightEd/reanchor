"""Source-balanced mixed-reference ranks; NOT normal-null p-values or FPR guarantees."""

from __future__ import annotations

import bisect
import json
from collections import Counter, defaultdict
from pathlib import Path

from reanchor.io import empty_directory, file_digest, read_jsonl, write_json, write_jsonl


def stratum(row):
    cbin = bisect.bisect_right([128, 512, 2048], row["candidate_count"])
    lbin = bisect.bisect_right([1024, 4096], row["source_length"])
    return row["task"], cbin, lbin


def keys(row):
    task, count, length = stratum(row)
    return [(task, count, length), (task, count, None), (task, None, None), (None, None, None)]


class ReferenceCalibrator:
    def __init__(self, reference, min_sources=50):
        if min_sources <= 0 or not reference:
            raise ValueError("positive min_sources and nonempty calibration required")
        if any(r["split"] != "calibration" for r in reference):
            raise ValueError("reference must use calibration split only")
        self.reference = reference
        self.min_sources = min_sources
        self.source_ids = {r["source_id"] for r in reference}
        self.buckets = defaultdict(list)
        for row in reference:
            for key in keys(row):
                self.buckets[key].append(row)
        self.tables = {}
        for key, rows in self.buckets.items():
            counts = Counter(r["source_id"] for r in rows)
            for score in ("raw_score", "negative_margin"):
                ordered = sorted((r[score], 1 / counts[r["source_id"]]) for r in rows)
                values, cumulative, total = [], [], 0.0
                for value, weight in ordered:
                    total += weight
                    values.append(value)
                    cumulative.append(total)
                self.tables[(key, score)] = (values, cumulative, total, len(counts))

    def transform(self, row):
        if row["source_id"] in self.source_ids:
            raise ValueError("calibration and scored sources must be disjoint")
        result = dict(row)
        chosen = keys(row)[-1]
        level = 3
        for candidate_level, key in enumerate(keys(row)):
            table = self.tables.get((key, "raw_score"))
            if table is not None and table[3] >= self.min_sources:
                chosen, level = key, candidate_level
                break
        for score, output_key in (("raw_score", "score"), ("negative_margin", "margin_rank")):
            values, cumulative, total, source_count = self.tables[(chosen, score)]
            # Strict-left rank: a constant/all-zero detector cannot turn every
            # tied normal observation into a 100th-percentile alarm.
            position = bisect.bisect_left(values, row[score])
            result[output_key] = cumulative[position - 1] / total if position else 0.0
        result["calibration_level"] = level
        result["calibration_sources"] = source_count
        return result


def verified_scores(directory: Path):
    metadata = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if file_digest(directory / "scores.jsonl") != metadata["scores_digest"]:
        raise ValueError("scores changed after freeze")
    return list(read_jsonl(directory / "scores.jsonl")), metadata


def calibrate_scores(reference: Path, scored: Path, output: Path, min_sources=50):
    ref_rows, ref_meta = verified_scores(reference)
    rows, meta = verified_scores(scored)
    if ref_meta["head_digest"] != meta["head_digest"]:
        raise ValueError("calibration and test checkpoints differ")
    if ref_meta["scope"] != meta["scope"]:
        raise ValueError("cannot calibrate natural all-token scores using program-selected slots")
    if ref_meta["extraction_identity"] != meta["extraction_identity"]:
        raise ValueError("calibration/test extraction identity mismatch")
    if ref_meta["dataset_identity"] != meta["dataset_identity"]:
        raise ValueError("calibration/test preparation provenance mismatch")
    if ref_meta["preparation_identity"] != meta["preparation_identity"]:
        raise ValueError("calibration/test preparation split/configuration mismatch")
    calibrator = ReferenceCalibrator(ref_rows, min_sources)
    transformed = [calibrator.transform(row) for row in rows]
    empty_directory(output)
    write_jsonl(output / "scores.jsonl", transformed)
    manifest = {
        **meta,
        "schema": "reanchor/calibrated-scores@1",
        "scores_digest": file_digest(output / "scores.jsonl"),
        "reference_digest": ref_meta["scores_digest"],
        "min_sources": min_sources,
        "reference_sources": sorted(calibrator.source_ids),
        "alarm_quantile": 0.99,
        "tie_policy": "strict-left empirical rank",
        "interpretation": "mixed unlabeled reference percentile; no normal-FPR guarantee",
        "fallback_counts": dict(Counter(r["calibration_level"] for r in transformed)),
    }
    write_json(output / "manifest.json", manifest)
    return manifest
