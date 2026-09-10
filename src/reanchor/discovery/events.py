"""Orchestrate scalable feature extraction, calibration and event freezing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from reanchor.artifacts import ArtifactStore
from reanchor.capture.attention import AttentionReader
from reanchor.capture.protocol import AuditDataset

from .calibration import MaxNullCalibrator, SelectionConfig
from .extractor import TransitionConfig, TransitionExtractor, TransitionFeatures
from .morphology import MorphologyConfig, MorphologyProfiler


@dataclass(frozen=True)
class DiscoveryConfig:
    window: int = 10
    local_floor: float = 0.50
    site_gain_floor: float = 0.10
    broad_gain_floor: float = 0.05
    broad_head_fraction: float = 0.25
    position_bins: int = 4
    family_alpha: float = 0.05
    min_calibration_sources: int = 32
    episode_gap: int = 1
    calibration_split: str = "train"
    device: str = "cuda:0"
    query_chunk: int = 8

    def __post_init__(self):
        self.transition()
        self.selection()
        if self.query_chunk < 1 or not self.device:
            raise ValueError("query_chunk and device must be valid")

    def transition(self) -> TransitionConfig:
        return TransitionConfig(
            window=self.window,
            local_floor=self.local_floor,
            site_gain_floor=self.site_gain_floor,
            broad_head_fraction=self.broad_head_fraction,
            position_bins=self.position_bins,
        )

    def selection(self) -> SelectionConfig:
        return SelectionConfig(
            position_bins=self.position_bins,
            family_alpha=self.family_alpha,
            min_calibration_sources=self.min_calibration_sources,
            sparse_gain_floor=self.site_gain_floor,
            broad_gain_floor=self.broad_gain_floor,
            episode_gap=self.episode_gap,
        )


class EventDiscovery:
    """Discover calibrated token episodes while keeping all labels inaccessible."""

    SCHEMA = "reanchor/max-null-episode@1"

    def __init__(
        self,
        config: DiscoveryConfig = DiscoveryConfig(),
        *,
        extractor=None,
        progress=None,
    ):
        self.config = config
        self.extractor = extractor
        self.progress = progress

    def run(self, dataset: AuditDataset, output: str | Path) -> dict:
        output = Path(output)
        if hasattr(dataset, "root") and output.resolve() == dataset.root.resolve():
            raise ValueError("run output must differ from the immutable capture root")
        samples, coverage = dataset.completed_samples(require_states=False)
        if not samples:
            raise ValueError("no complete attention captures are available")
        extractor = self.extractor
        if extractor is None:
            reader = AttentionReader(
                dataset,
                device=self.config.device,
                query_chunk=self.config.query_chunk,
            )
            extractor = TransitionExtractor(reader, self.config.transition())
        store = ArtifactStore(output)
        settings = json.dumps(asdict(self.config), sort_keys=True)
        records = []
        transition_paths = {}
        resumed_transitions = 0
        transition_samples = (
            self.progress.track(samples, description="discover transitions")
            if self.progress
            else samples
        )
        for sample in transition_samples:
            path = store.sample_path(sample, "transitions.npz")
            if path.is_file():
                values = store.read_npz(path)
                if (
                    str(values.get("method_schema", "")) != self.SCHEMA
                    or str(values.get("settings", "")) != settings
                    or str(values.get("sample_key", "")) != sample.key
                    or bool(values.get("labels_used", True))
                ):
                    raise ValueError(
                        f"{sample.key}: existing transition artifact has different identity"
                    )
                features = TransitionFeatures.from_arrays(sample, values)
                resumed_transitions += 1
            else:
                features = extractor.run(sample)
                store.write_npz(
                    path,
                    **features.arrays(),
                    method_schema=np.array(self.SCHEMA),
                    settings=np.array(settings),
                    sample_key=np.array(sample.key),
                    labels_used=np.array(False),
                )
            transition_paths[sample.key] = path
            records.append(features.token_scores(self.config.transition()))

        calibration_records = [
            record for record in records if record.split == self.config.calibration_split
        ]
        calibrator = MaxNullCalibrator(self.config.selection()).fit(calibration_records)
        store.write_json(output / "calibration.json", calibrator.artifact())
        profiler = MorphologyProfiler(
            MorphologyConfig(
                active_gain_floor=min(self.config.site_gain_floor, self.config.broad_gain_floor),
                local_floor=self.config.local_floor,
            )
        )
        manifest_samples = []
        eligible_count = floor_candidate_count = 0
        significant_count = anchor_count = samples_with_anchors = 0
        event_samples = (
            self.progress.track(
                range(len(samples)),
                description="freeze reanchor events",
                total=len(samples),
            )
            if self.progress
            else range(len(samples))
        )
        for sample_index in event_samples:
            sample, record = samples[sample_index], records[sample_index]
            values = store.read_npz(transition_paths[sample.key])
            if str(values["settings"]) != settings or str(values["sample_key"]) != sample.key:
                raise ValueError(f"{sample.key}: transition artifact identity changed")
            features = TransitionFeatures.from_arrays(sample, values)
            selection = calibrator.select(record)
            morphology = profiler.run(features, selection)
            event_path = store.sample_path(sample, "events.npz")
            store.write_npz(
                event_path,
                schema=np.array(1),
                method_schema=np.array(self.SCHEMA),
                settings=np.array(settings),
                sample_key=np.array(sample.key),
                row_position=record.row_position,
                eligible=record.eligible,
                position_bin=record.position_bin,
                sparse_score=record.sparse_score,
                broad_score=record.broad_score,
                supporting_sites=record.supporting_sites,
                family_p_value=selection.family_p_value,
                channel=selection.channel,
                significant=selection.significant,
                episode_id=selection.episode_id,
                anchor=selection.anchor,
                labels_used=np.array(False),
                **morphology.arrays(),
            )
            significant = int(selection.significant.sum())
            anchors = int(selection.anchor.sum())
            eligible_count += int(record.eligible.sum())
            floor_candidate_count += int((selection.channel != "none").sum())
            significant_count += significant
            anchor_count += anchors
            samples_with_anchors += int(anchors > 0)
            manifest_samples.append(
                {
                    "key": sample.key,
                    "source_id": sample.source_id,
                    "transitions": str(transition_paths[sample.key].relative_to(output)),
                    "events": str(event_path.relative_to(output)),
                    "significant_transitions": significant,
                    "anchors": anchors,
                }
            )
        summary = {
            "method_schema": self.SCHEMA,
            "capture_root": str(getattr(dataset, "root", "in-memory")),
            "coverage": coverage,
            "samples": len(samples),
            "calibration_split": self.config.calibration_split,
            "calibration_sources": len({record.source_id for record in calibration_records}),
            "significant_transitions": significant_count,
            "anchors": anchor_count,
            "selection_funnel": {
                "eligible_tokens": eligible_count,
                "floor_candidates": floor_candidate_count,
                "significant_transitions": significant_count,
                "anchors": anchor_count,
                "samples_with_anchors": samples_with_anchors,
            },
            "resumed_transitions": resumed_transitions,
            "labels_used_for_discovery": False,
            "settings": asdict(self.config),
            "sample_artifacts": manifest_samples,
        }
        store.write_json(output / "index.json", summary)
        return summary
