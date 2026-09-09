"""Independent-source max-null calibration and episode selection."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SelectionConfig:
    position_bins: int = 4
    family_alpha: float = 0.05
    min_calibration_sources: int = 32
    sparse_gain_floor: float = 0.10
    broad_gain_floor: float = 0.05
    episode_gap: int = 1

    def __post_init__(self):
        if self.position_bins < 1 or self.min_calibration_sources < 2:
            raise ValueError("position_bins and calibration source count are invalid")
        if not 0 < self.family_alpha < 1:
            raise ValueError("family_alpha must be in (0, 1)")
        if min(self.sparse_gain_floor, self.broad_gain_floor, self.episode_gap) < 0:
            raise ValueError("gain floors and episode_gap must be nonnegative")


@dataclass(frozen=True)
class TokenScoreRecord:
    split: str
    task: str
    sample_id: str
    source_id: str
    row_position: np.ndarray
    eligible: np.ndarray
    position_bin: np.ndarray
    sparse_score: np.ndarray
    broad_score: np.ndarray
    supporting_sites: np.ndarray

    def __post_init__(self):
        count = len(self.row_position)
        arrays = (
            self.eligible,
            self.position_bin,
            self.sparse_score,
            self.broad_score,
            self.supporting_sites,
        )
        if any(np.asarray(value).shape != (count,) for value in arrays):
            raise ValueError("token score fields must share one row axis")

    @property
    def key(self) -> str:
        return f"{self.split}/{self.task}/{self.sample_id}"


@dataclass(frozen=True)
class CalibratedSelection:
    family_p_value: np.ndarray
    channel: np.ndarray
    significant: np.ndarray
    episode_id: np.ndarray
    anchor: np.ndarray


class MaxNullCalibrator:
    """Calibrate complete token statistics against independent-source maxima."""

    _CHANNELS = ("sparse", "broad")

    def __init__(self, config: SelectionConfig = SelectionConfig()):
        self.config = config
        self._nulls: dict[tuple[str, int, str], np.ndarray] = {}

    def fit(self, records: list[TokenScoreRecord] | tuple[TokenScoreRecord, ...]):
        maxima: dict[tuple[str, int, str, str], float] = {}
        pooled: dict[tuple[str, str, str], float] = {}
        for record in records:
            for channel in self._CHANNELS:
                values = np.asarray(getattr(record, f"{channel}_score"), dtype=float)
                for position_bin in range(self.config.position_bins):
                    take = np.asarray(record.eligible) & (record.position_bin == position_bin)
                    if not take.any():
                        continue
                    key = (record.task, position_bin, channel, record.source_id)
                    maxima[key] = max(maxima.get(key, -np.inf), float(values[take].max()))
                take = np.asarray(record.eligible)
                if take.any():
                    key = (record.task, channel, record.source_id)
                    pooled[key] = max(pooled.get(key, -np.inf), float(values[take].max()))

        tasks = {record.task for record in records}
        family_tests = len(self._CHANNELS) * self.config.position_bins
        resolution_sources = math.ceil(family_tests / self.config.family_alpha) - 1
        required_sources = max(self.config.min_calibration_sources, resolution_sources)
        for task in tasks:
            for channel in self._CHANNELS:
                values = [
                    value
                    for (candidate_task, candidate_channel, _), value in pooled.items()
                    if (candidate_task, candidate_channel) == (task, channel)
                ]
                if len(values) < required_sources:
                    raise ValueError(
                        f"{task}/{channel}: need at least {required_sources} independent "
                        "calibration sources to resolve family alpha"
                    )
                self._nulls[task, -1, channel] = np.sort(values)
                for position_bin in range(self.config.position_bins):
                    exact = [
                        value
                        for (
                            candidate_task,
                            candidate_bin,
                            candidate_channel,
                            _,
                        ), value in maxima.items()
                        if (candidate_task, candidate_bin, candidate_channel)
                        == (task, position_bin, channel)
                    ]
                    if len(exact) >= required_sources:
                        self._nulls[task, position_bin, channel] = np.sort(exact)
        return self

    def select(self, record: TokenScoreRecord) -> CalibratedSelection:
        count = len(record.row_position)
        channel_p = np.ones((2, count), dtype=np.float64)
        family_tests = 2 * self.config.position_bins
        floors = (self.config.sparse_gain_floor, self.config.broad_gain_floor)
        for channel_index, (channel, floor) in enumerate(zip(self._CHANNELS, floors)):
            values = np.asarray(getattr(record, f"{channel}_score"), dtype=float)
            for index in np.flatnonzero(record.eligible & (values >= floor)):
                position_bin = int(record.position_bin[index])
                null = self._nulls.get((record.task, position_bin, channel))
                if null is None:
                    null = self._nulls.get((record.task, -1, channel))
                if null is None:
                    raise ValueError(f"no calibration null for {record.task}/{channel}")
                exceedances = len(null) - int(np.searchsorted(null, values[index], side="left"))
                raw = (1 + exceedances) / (len(null) + 1)
                channel_p[channel_index, index] = min(1.0, family_tests * raw)

        family_p = channel_p.min(0)
        choice = channel_p.argmin(0)
        channel = np.full(count, "none", dtype="<U8")
        tested = record.eligible & (
            (record.sparse_score >= self.config.sparse_gain_floor)
            | (record.broad_score >= self.config.broad_gain_floor)
        )
        channel[tested] = np.asarray(self._CHANNELS)[choice[tested]]
        significant = tested & (family_p <= self.config.family_alpha)
        episode_id = np.full(count, -1, dtype=np.int32)
        anchor = np.zeros(count, dtype=bool)
        episode = -1
        previous = None
        members: list[int] = []

        def close_episode():
            if not members:
                return
            selected = min(
                members,
                key=lambda index: (
                    family_p[index],
                    -max(record.sparse_score[index], record.broad_score[index]),
                    record.row_position[index],
                ),
            )
            anchor[selected] = True

        for index in np.flatnonzero(significant):
            position = int(record.row_position[index])
            if previous is None or position - previous > self.config.episode_gap:
                close_episode()
                episode += 1
                members = []
            members.append(int(index))
            episode_id[index] = episode
            previous = position
        close_episode()
        return CalibratedSelection(family_p, channel, significant, episode_id, anchor)

    def artifact(self) -> dict:
        """Return the complete fitted null needed to audit every empirical p-value."""

        return {
            "calibration_schema": "reanchor/source-max-null@1",
            "labels_used": False,
            "settings": asdict(self.config),
            "strata": [
                {
                    "task": task,
                    "position_bin": position_bin,
                    "channel": channel,
                    "source_maxima": values.tolist(),
                }
                for (task, position_bin, channel), values in sorted(self._nulls.items())
            ],
        }
