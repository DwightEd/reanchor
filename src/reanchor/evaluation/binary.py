"""Source-balanced binary ranking metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryEvaluator:
    """Compute AUROC/AUPR after reducing correlated tokens within each source."""

    bootstrap: int = 1000
    seed: int = 0

    def __post_init__(self) -> None:
        if self.bootstrap < 0:
            raise ValueError("bootstrap must be nonnegative")

    def evaluate(
        self,
        rows: list[dict],
        score: str,
        *,
        label: str = "label",
        source: str = "source_id",
    ) -> dict:
        labels, scores, sources, observations = self._arrays(rows, score, label, source)
        weights = self._source_weights(sources)
        auroc, aupr = self._metrics(labels, scores, weights)
        auroc_draws, aupr_draws = self._cluster_bootstrap(labels, scores, sources)
        auroc_interval = self._interval(auroc, auroc_draws, self.bootstrap)
        return {
            "auroc": auroc,
            "auroc_ci95": auroc_interval,
            "aupr": aupr,
            "aupr_ci95": self._interval(aupr, aupr_draws, self.bootstrap),
            "positive_prevalence": self._prevalence(labels, weights),
            "sources": len(set(map(str, sources))),
            "observations": observations,
            "evaluation_units": len(labels),
            "source_balanced": True,
            "within_source_reduction": "mean by source_id and label",
            "cluster_bootstrap": source,
            "valid_bootstrap_draws": len(auroc_draws),
            "auroc_ci_excludes_0_5": bool(
                auroc is not None
                and auroc_interval[0] is not None
                and (auroc_interval[0] > 0.5 or auroc_interval[1] < 0.5)
            ),
        }

    @staticmethod
    def _arrays(rows, score, label, source):
        grouped = defaultdict(list)
        observations = 0
        for row in rows:
            if row.get(label) not in (0, 1):
                continue
            number = float(row[score])
            if np.isfinite(number):
                grouped[str(row[source]), int(row[label])].append(number)
                observations += 1
        values = [
            (label_value, float(np.mean(items)), source_value)
            for (source_value, label_value), items in grouped.items()
        ]
        if not values:
            empty = np.empty(0)
            return empty.astype(int), empty, empty.astype(str), 0
        labels, scores, sources = zip(*values, strict=True)
        return np.asarray(labels), np.asarray(scores), np.asarray(sources), observations

    @staticmethod
    def _source_weights(sources: np.ndarray) -> np.ndarray:
        if not len(sources):
            return np.empty(0)
        _, inverse, counts = np.unique(sources, return_inverse=True, return_counts=True)
        return 1.0 / counts[inverse]

    @classmethod
    def _metrics(cls, labels, scores, weights):
        positive = labels == 1
        positive_weight = float(weights[positive].sum())
        negative_weight = float(weights[~positive].sum())
        if positive_weight == 0 or negative_weight == 0:
            return None, None

        order = np.argsort(scores, kind="stable")
        sorted_scores = scores[order]
        sorted_labels = labels[order]
        sorted_weights = weights[order]
        concordance = negative_below = 0.0
        for start, stop in cls._tie_blocks(sorted_scores):
            block_positive = float(
                sorted_weights[start:stop][sorted_labels[start:stop] == 1].sum()
            )
            block_negative = float(
                sorted_weights[start:stop][sorted_labels[start:stop] == 0].sum()
            )
            concordance += block_positive * (negative_below + 0.5 * block_negative)
            negative_below += block_negative
        auroc = concordance / (positive_weight * negative_weight)

        order = order[::-1]
        sorted_scores = scores[order]
        sorted_labels = labels[order]
        sorted_weights = weights[order]
        true_positive = false_positive = aupr = 0.0
        for start, stop in cls._tie_blocks(sorted_scores):
            block_positive = float(
                sorted_weights[start:stop][sorted_labels[start:stop] == 1].sum()
            )
            block_negative = float(
                sorted_weights[start:stop][sorted_labels[start:stop] == 0].sum()
            )
            true_positive += block_positive
            false_positive += block_negative
            aupr += (
                block_positive
                / positive_weight
                * true_positive
                / (true_positive + false_positive)
            )
        return float(auroc), float(aupr)

    def _cluster_bootstrap(self, labels, scores, sources):
        if self.bootstrap == 0 or not len(sources):
            return [], []
        unique = np.unique(sources)
        indices = {name: np.flatnonzero(sources == name) for name in unique}
        random = np.random.default_rng(self.seed)
        auroc_draws = []
        aupr_draws = []
        for _ in range(self.bootstrap):
            selected = random.choice(unique, size=len(unique), replace=True)
            draw_indices = np.concatenate([indices[name] for name in selected])
            occurrence = np.concatenate(
                [np.full(len(indices[name]), index) for index, name in enumerate(selected)]
            )
            weights = self._source_weights(occurrence)
            auroc, aupr = self._metrics(labels[draw_indices], scores[draw_indices], weights)
            if auroc is not None:
                auroc_draws.append(auroc)
                aupr_draws.append(aupr)
        return auroc_draws, aupr_draws

    @staticmethod
    def _tie_blocks(scores):
        boundaries = np.flatnonzero(np.diff(scores) != 0) + 1
        edges = np.concatenate(([0], boundaries, [len(scores)]))
        return zip(edges[:-1], edges[1:], strict=True)

    @staticmethod
    def _prevalence(labels, weights):
        return float(weights[labels == 1].sum() / weights.sum()) if len(labels) else None

    @staticmethod
    def _interval(estimate, draws, attempted):
        if estimate is None:
            return [None, None]
        if not draws:
            return [estimate, estimate] if attempted == 0 else [None, None]
        return [float(value) for value in np.quantile(draws, (0.025, 0.975))]
