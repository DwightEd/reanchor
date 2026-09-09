"""Lazy checkpoint access and exact reconstruction shared by tracing operators."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F


class CheckpointWeights:
    """Load only requested tensors from a local bias-free Llama checkpoint."""

    def __init__(self, directory: str | Path, device: str = "cpu"):
        self.directory = Path(directory)
        self.device = device
        self.config = json.loads((self.directory / "config.json").read_text())
        index = self.directory / "model.safetensors.index.json"
        self.files = json.loads(index.read_text())["weight_map"] if index.exists() else {}
        if not self.files and not (self.directory / "model.safetensors").is_file():
            raise ValueError("tracing requires the capture model's local safetensors checkpoint")
        if (
            self.config.get("model_type") != "llama"
            or self.config.get("hidden_act", "silu") != "silu"
            or self.config.get("attention_bias", False)
            or self.config.get("mlp_bias", False)
        ):
            raise ValueError("tracing supports bias-free Llama models with SwiGLU")

    def get(self, name: str) -> torch.Tensor:
        from safetensors import safe_open

        if name == "lm_head.weight" and self.config.get("tie_word_embeddings", False):
            name = "model.embed_tokens.weight"
        filename = self.files.get(name, "model.safetensors")
        with safe_open(self.directory / filename, framework="pt", device="cpu") as archive:
            return archive.get_tensor(name).to(self.device)


def norm_component(part, reference, weight, epsilon):
    """Apply the reference RMS denominator while retaining signed direction."""

    variance = reference.float().square().mean(-1, keepdim=True) + epsilon
    return part * weight.float() * torch.rsqrt(variance)


def readout_directions(trace, states, weights, chunk=16):
    """Construct observed-token versus strongest-runner directions without labels."""

    device = weights.device
    raw = torch.as_tensor(states["final_residual"], device=device)
    count = len(raw) - 1
    rows = trace["row_position"][:count]
    ids = torch.as_tensor(trace["token_ids"][rows + 1], device=device)
    norm = weights.get("model.norm.weight")
    unembed = weights.get("lm_head.weight")
    epsilon = weights.config["rms_norm_eps"]
    runner = torch.empty(count, device=device, dtype=torch.long)
    direction = torch.zeros_like(raw, dtype=torch.float32)
    for begin in range(0, count, chunk):
        end = min(begin + chunk, count)
        x, target = raw[begin:end], ids[begin:end]
        if "readout_runner_id" in trace:
            other = torch.as_tensor(trace["readout_runner_id"][begin:end], device=device)
        else:
            scale = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + epsilon)
            logits = F.linear((x.float() * scale).to(norm.dtype) * norm, unembed).float()
            logits.scatter_(1, target[:, None], -torch.inf)
            other = logits.argmax(-1)
        runner[begin:end] = other
        difference = unembed[target].float() - unembed[other].float()
        direction[begin:end] = norm_component(difference, x, norm, epsilon)
    return direction, runner.cpu().numpy()


def attention_chunks(trace, qk, history, layer, device, chunk):
    """Reconstruct full causal rows and restore the captured response history."""

    query = torch.as_tensor(qk[f"query_{layer}"], device=device)
    key = torch.as_tensor(qk[f"key_{layer}"], device=device)
    dtype = getattr(torch, str(qk[f"dtype_{layer}"]))
    heads, rows_count, _ = query.shape
    key = key.repeat_interleave(heads // len(key), 0).to(dtype)
    rows = torch.as_tensor(trace["row_position"], device=device)
    native_history = torch.as_tensor(history[f"L{layer}"], device=device)
    source = torch.arange(key.shape[1], device=device)
    for begin in range(0, rows_count, chunk):
        end = min(begin + chunk, rows_count)
        scores = (query[:, begin:end].to(dtype) @ key.transpose(-1, -2)) * float(
            qk[f"scale_{layer}"]
        )
        scores.masked_fill_(
            source[None, None] > rows[None, begin:end, None], torch.finfo(dtype).min
        )
        attention = scores.softmax(-1, dtype=torch.float32).to(dtype).float()
        attention[..., rows] = native_history[:, begin:end]
        yield begin, end, attention
