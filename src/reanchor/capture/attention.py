"""Reconstruct complete native attention behind a streaming layer interface."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch

from .protocol import AuditDataset, AuditSample


class AttentionReader:
    """Yield one complete response-query attention layer at a time."""

    def __init__(self, dataset: AuditDataset, *, device: str = "cpu", query_chunk: int = 32):
        if query_chunk < 1:
            raise ValueError("query_chunk must be positive")
        self.dataset = dataset
        self.device = torch.device(device)
        self.query_chunk = query_chunk

    @torch.inference_mode()
    def iter_layers(self, sample: AuditSample) -> Iterator[tuple[int, np.ndarray]]:
        paths = self.dataset.paths(sample)
        metadata = self.dataset.load_metadata(sample, "row_position")
        rows = np.asarray(metadata["row_position"], dtype=np.int64)
        with (
            np.load(paths.qk, allow_pickle=False) as qk,
            np.load(paths.history, allow_pickle=False) as history,
        ):
            layers = sorted(
                int(name.removeprefix("query_")) for name in qk.files if name.startswith("query_")
            )
            if not layers:
                raise ValueError(f"{sample.key}: Q/K archive contains no layers")
            for layer in layers:
                query_value = qk[f"query_{layer}"]
                key_value = qk[f"key_{layer}"]
                if (
                    query_value.shape[1] != len(rows)
                    or query_value.shape[-1] != key_value.shape[-1]
                ):
                    raise ValueError(f"{sample.key}: incompatible Q/K axes at layer {layer}")
                heads, key_heads = len(query_value), len(key_value)
                if heads % key_heads:
                    raise ValueError(f"{sample.key}: Q heads are not divisible by KV heads")
                dtype_name = str(qk[f"dtype_{layer}"]) if f"dtype_{layer}" in qk else "float32"
                dtype = getattr(torch, dtype_name)
                query = torch.as_tensor(query_value, dtype=dtype, device=self.device)
                key = torch.as_tensor(key_value, dtype=dtype, device=self.device)
                key = key.repeat_interleave(heads // key_heads, dim=0)
                scale = float(qk[f"scale_{layer}"]) if f"scale_{layer}" in qk else 1.0
                output = np.empty((heads, len(rows), key.shape[1]), dtype=np.float32)
                sources = torch.arange(key.shape[1], device=self.device)
                for begin in range(0, len(rows), self.query_chunk):
                    end = min(begin + self.query_chunk, len(rows))
                    scores = (query[:, begin:end] @ key.transpose(-1, -2)) * scale
                    future = (
                        sources[None, None]
                        > torch.as_tensor(rows[begin:end], device=self.device)[None, :, None]
                    )
                    scores.masked_fill_(future, torch.finfo(dtype).min)
                    probability = scores.softmax(-1, dtype=torch.float32).to(dtype).float()
                    output[:, begin:end] = probability.cpu().numpy()
                native = np.asarray(history[f"L{layer}"], dtype=np.float32)
                if native.shape != (heads, len(rows), len(rows)):
                    raise ValueError(f"{sample.key}: incompatible native history at layer {layer}")
                output[..., rows] = native
                yield layer, output
