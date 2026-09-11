"""Read source disagreement and representation relations without relation labels."""

import numpy as np
from tqdm.auto import tqdm

from decoding.attention import entropy
from decoding.reading_graph import normalize


def unit_vectors(values: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, lengths, out=np.zeros_like(values), where=lengths > 0)


class RouteReadout:
    def __init__(self, graph: dict, hidden: np.ndarray | None, prompt: int):
        self.graph, self.hidden, self.prompt = graph, hidden, prompt

    def run(self) -> dict[str, np.ndarray]:
        paths = self.graph["paths"]
        shape = paths.shape[:3]
        result = {
            name: np.full(shape, np.nan)
            for name in ("route_divergence", "relation_residual", "relation_shuffled")
        }
        result["relation_coverage"] = np.zeros(shape)
        if self.hidden is not None:
            embeddings = unit_vectors(self.hidden[0].astype(np.float32))
            queries = embeddings[self.prompt - 1 :]
            lexical = np.exp(queries @ embeddings[self.graph["source_positions"]].T)
        for layer in tqdm(range(shape[1]), desc="route relations", unit="layer", leave=False):
            for hop in range(min(layer + 1, shape[0])):
                if hop:
                    result["route_divergence"][hop, layer] = self._divergence(
                        paths[0, layer], paths[hop, layer]
                    )
                if self.hidden is not None:
                    values = self._relations(layer, paths[hop, layer], lexical)
                    for name, numbers in values.items():
                        result[name][hop, layer] = numbers
        return result

    @staticmethod
    def _divergence(direct, indirect):
        left, right = normalize(direct), normalize(indirect)
        divergence = entropy((left + right) / 2) - (entropy(left) + entropy(right)) / 2
        divergence[(direct.sum(-1) == 0) | (indirect.sum(-1) == 0)] = np.nan
        return np.maximum(0, divergence)

    def _relations(self, layer, paths, lexical):
        states = self.hidden[layer + 1].astype(np.float32)
        source = states[self.graph["source_positions"]]
        states = unit_vectors(states - source.mean(0))
        source = states[self.graph["source_positions"]]
        queries = states[self.prompt - 1 :]
        # Correspondences are fixed by reading paths and lexical content, never
        # fitted to reduce the relation residual.
        transport = normalize(paths * lexical)
        mapped = transport @ source
        observed = queries @ queries.T
        weights = np.tril(self.graph["history"][layer], k=-2).copy()
        valid = (paths.sum(-1) > 0) & (np.linalg.norm(queries, axis=-1) > 0)
        valid &= (transport @ (np.linalg.norm(source, axis=-1) > 0)) > 0
        pair_valid = valid[:, None] & valid[None, 1:]
        pair_weights = weights[:, :-1] * pair_valid
        mass = pair_weights.sum(-1)
        total = weights.sum(-1)
        residual = self._weighted_residual(observed, mapped, pair_weights, mass)
        # A deterministic source reassignment preserves path masses and row
        # entropies. This is a structural control, not a ground-truth negative.
        shuffled = transport @ np.roll(source, 1, axis=0)
        null = self._weighted_residual(observed, shuffled, pair_weights, mass)
        coverage = np.divide(mass, total, out=np.zeros_like(mass), where=total > 0)
        return dict(relation_residual=residual, relation_shuffled=null, relation_coverage=coverage)

    @staticmethod
    def _weighted_residual(observed, mapped, weights, mass):
        residual = np.maximum(0, observed[:, 1:] - (mapped @ mapped.T)[:, 1:])
        return np.divide(
            (residual * weights).sum(-1), mass, out=np.full(len(mass), np.nan), where=mass > 0
        )
