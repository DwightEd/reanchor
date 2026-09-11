"""Content-reading events and layer-ordered attention paths."""

import numpy as np
from tqdm.auto import tqdm

from decoding.attention import entropy


def normalize(weights: np.ndarray) -> np.ndarray:
    mass = weights.sum(-1, keepdims=True)
    return np.divide(weights, mass, out=np.zeros_like(weights), where=mass > 0)


class ReadingGraph:
    def __init__(self, attention: np.ndarray, source_mask: np.ndarray, hops: int):
        self.attention, self.source, self.hops = attention, source_mask, hops

    def run(self) -> dict[str, np.ndarray]:
        layers, _, steps, _ = self.attention.shape
        positions = np.flatnonzero(self.source)
        if self.hops < 1 or not len(positions):
            raise ValueError("hops must be positive and source tokens must exist")
        edges = self.attention.mean(axis=1, dtype=np.float32)
        history = np.tril(edges[:, :, len(self.source) :], k=-1)
        paths = np.zeros((min(self.hops, layers), layers, steps, len(positions)), np.float32)
        paths[0] = edges[:, :, positions]
        for hop in range(1, len(paths)):
            # Row u+1 is the state of generated token u as an INPUT.
            earlier = np.zeros_like(paths[hop - 1])
            earlier[:, :-1] = paths[hop - 1, :, 1:]
            for layer in range(hop, layers):
                paths[hop, layer] = history[layer] @ earlier[layer - 1]
        return dict(source_positions=positions, direct=paths[0], history=history, paths=paths)


class RevisitSignal:
    def __init__(self, attention, source_mask, special_mask, window, quantile):
        self.attention, self.source = attention, source_mask
        self.special, self.window, self.quantile = special_mask, window, quantile

    def run(self) -> dict[str, np.ndarray]:
        layers, heads, steps, _ = self.attention.shape
        names = ("shift", "layer_revisit", "dispersion", "disagreement", "effective_rank")
        values = {name: np.full((steps, layers), np.nan) for name in names}
        values.update({name: np.full(steps, np.nan) for name in ("revisit", "threshold")})
        values.update({name: np.zeros(steps, dtype=bool) for name in ("active", "event")})
        shifts = np.zeros((steps, layers, heads))
        for t in tqdm(range(steps), desc="content reads", unit="token", leave=False):
            current = self._content(t)
            for name, numbers in self._structure(current).items():
                values[name][t] = numbers
            if t:
                # Compare the same keys: the previous query becoming history is
                # not itself evidence of revisiting an older object.
                old = self._content(t - 1)
                common = current[..., : old.shape[-1]].copy()
                common[..., -1] = 0
                shifts[t] = np.abs(np.cumsum(normalize(common) - normalize(old), -1)).sum(-1)
                mass = np.minimum(common.sum(-1), old.sum(-1))
                shifts[t][mass == 0] = 0
                values["shift"][t] = shifts[t].mean(-1)
                if t > 1:
                    past = shifts[max(1, t - self.window) : t]
                    excess = np.maximum(0, shifts[t] - np.median(past, axis=0)) * mass
                    values["layer_revisit"][t] = excess.mean(-1)
                    values["revisit"][t] = excess.mean() if mass.any() else np.nan
            self._event(t, values)
        return values

    def _content(self, step):
        query = len(self.source) + step - 1
        raw = self.attention[:, :, step, : query + 1].astype(np.float64)
        if (
            not np.isfinite(raw).all()
            or (raw < 0).any()
            or not np.allclose(raw.sum(-1), 1, atol=0.005, rtol=0)
            or np.count_nonzero(self.attention[:, :, step, query + 1 :])
        ):
            raise ValueError(f"step {step}: invalid attention or visible future keys")
        allowed = ~self.special[: query + 1].copy()
        allowed[: len(self.source)] &= self.source
        allowed[query] = False
        return raw * allowed

    def _structure(self, current):
        probabilities = normalize(current)
        valid = current.sum(-1) > 0
        count = valid.sum(-1)
        mean = np.divide(
            probabilities.sum(1),
            count[:, None],
            out=np.zeros_like(probabilities[:, 0]),
            where=count[:, None] > 0,
        )
        dispersion = np.divide(
            entropy(probabilities).sum(-1), count, out=np.full(len(count), np.nan), where=count > 0
        )
        root = np.sqrt(probabilities)
        gram = root @ root.swapaxes(-1, -2)
        spectrum = normalize(np.maximum(0, np.linalg.eigvalsh(gram)))
        rank = np.exp2(entropy(spectrum))
        rank[count == 0] = np.nan
        return dict(
            dispersion=dispersion, disagreement=entropy(mean) - dispersion, effective_rank=rank
        )

    def _event(self, step, values):
        history = values["revisit"][max(0, step - self.window) : step]
        history = history[np.isfinite(history)]
        if len(history):
            values["threshold"][step] = np.quantile(history, self.quantile)
        active = step > self.window and values["revisit"][step] > values["threshold"][step] + 1e-10
        values["active"][step] = active
        values["event"][step] = active and not values["active"][step - 1]
