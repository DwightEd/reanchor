"""G0: candidate-conditioned source reads, not a graph-statistic classifier."""

from __future__ import annotations

import math

import torch
from torch import nn


class SourceRead(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.transform = nn.Sequential(
            nn.Linear(width, width, bias=False),
            nn.GELU(),
            nn.Linear(width, width, bias=False),
        )

    def forward(self, query, source):
        weights = self.query(query) @ self.key(source).T / math.sqrt(query.shape[-1])
        evidence = self.out(weights.softmax(-1) @ self.value(source))
        return evidence + self.transform(evidence)


class SupportHead(nn.Module):
    """Address controls routing; only source vectors enter the evidence readout.

    Candidate/address residuals never enter the final value vector. This is an
    architectural constraint, NOT a guarantee that the selected fact is right.
    All input tensors are unpadded [sequence, frozen_hidden_size] matrices.
    """

    def __init__(self, input_dim: int, width: int = 128, blocks: int = 2):
        super().__init__()
        if min(input_dim, width, blocks) <= 0:
            raise ValueError("dimensions and blocks must be positive")
        self.config = {"input_dim": input_dim, "width": width, "blocks": blocks}
        self.input_norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.source_projection = nn.Linear(input_dim, width, bias=False)
        self.address_projection = nn.Linear(input_dim, width, bias=False)
        self.candidate_projection = nn.Linear(input_dim, width, bias=False)
        self.readout_projection = nn.Linear(input_dim, width, bias=False)
        self.reads = nn.ModuleList(SourceRead(width) for _ in range(blocks))

    def forward(self, source, address, candidates, candidate_chunk: int = 256):
        if address.shape[0] == 0 or candidates.shape[0] == 0 or candidate_chunk <= 0:
            raise ValueError("address/candidates must be nonempty and chunk size positive")
        if source.shape[0] == 0:
            # Keep a gradient connection for source-null training/debugging.
            return self.source_projection.weight.sum() * 0 + candidates.new_zeros(len(candidates))
        source = self.source_projection(self.input_norm(source.float()))
        address = self.address_projection(self.input_norm(address.float()))
        pieces = []
        for chunk in candidates.float().split(candidate_chunk):
            normalized = self.input_norm(chunk)
            query = self.candidate_projection(normalized) + address[-1]
            # Prefix information is a routing/address channel, never a value root.
            address_weights = (query @ address.T / math.sqrt(query.shape[-1])).softmax(-1)
            query = query + address_weights @ address
            evidence = None
            for read in self.reads:
                evidence = read(query, source)
                query = query + evidence
            value = self.readout_projection(normalized)
            pieces.append((value * evidence).sum(-1) / math.sqrt(value.shape[-1]))
        return torch.cat(pieces)


def support_loss(energies, kind: str, target_index: int):
    if kind == "binding":
        return -energies.log_softmax(0)[target_index]
    if kind == "neutral":
        return (energies - energies.mean()).square().mean()
    raise ValueError("training requires program binding/neutral targets, never H labels")
