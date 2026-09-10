from types import SimpleNamespace

import numpy as np
import torch

from reanchor.tracing.propagation import trace_events


def test_trace_events_accumulates_many_head_roots_for_one_event(monkeypatch):
    head_count = 17
    token_count = 4

    class FakeLayer:
        def __init__(self, cache, layer, query_chunk):
            self.cache = cache
            self.layer = layer
            self.device = "cpu"
            self.h = head_count
            self.hd = 1
            self.output_gram = torch.ones(head_count, 1, 1)
            self.output_blocks = torch.zeros(head_count, 2, 1)

        def cache_attention(self):
            pass

        def remote_seeds(self, sites, window, *, source_mask=None):
            return {
                (int(head), int(row)): (
                    torch.tensor([float(head + 1), 0.0]),
                    np.full(token_count, head + 1, dtype=np.float32),
                )
                for head, row in sites
            }

        def attention_jvp(self, delta, *, routing):
            codes = torch.zeros(len(delta), head_count, 3, 1)
            return torch.zeros_like(delta), torch.zeros_like(delta), codes

        def mlp_jvp(self, delta):
            return torch.zeros_like(delta)

    cache = SimpleNamespace(
        weights=SimpleNamespace(config={"hidden_size": 2}, device="cpu"),
        rows=3,
        layers=1,
        heads=head_count,
        trace={
            "token_ids": np.arange(token_count),
            "row_position": np.arange(3),
        },
    )
    coordinates = np.array([[0, head, 1] for head in range(head_count)])
    directions = (
        torch.zeros(3, 2),
        np.zeros(2, dtype=int),
        np.ones(2, dtype=int),
        np.zeros(2, dtype=bool),
        np.zeros(2, dtype=np.float32),
    )

    import reanchor.tracing.propagation as module

    monkeypatch.setattr(module, "DifferentialLayer", FakeLayer)
    monkeypatch.setattr(module, "final_directions", lambda cache, contrasts: directions)

    result = trace_events(cache, coordinates, event_batch=1)

    assert len(result) == 1
    np.testing.assert_array_equal(
        result[0]["root_attention_sum"],
        np.full(token_count, sum(range(1, head_count + 1)), dtype=np.float32),
    )
