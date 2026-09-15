"""DCM: learn an intervention subspace, NOT an activation reconstruction model.

Only mask parameters are optimized; frozen language-model weights remain fixed.
Semantic counterfactual targets are supervision, even without hallucination labels.
"""
from __future__ import annotations

import random
import torch
from torch import Tensor

from .patching import patch_logits


def source_split(pairs, fraction=.75, seed=20260915):
    sources = sorted({p.source_id for p in pairs})
    if not 0 < fraction < 1 or len(sources) < 2:
        raise ValueError("DCM requires at least two distinct sources and a nonempty holdout")
    random.Random(seed).shuffle(sources)
    cut = min(len(sources) - 1, max(1, int(fraction * len(sources))))
    train = set(sources[:cut])
    return {p.id: "train" if p.source_id in train else "validation" for p in pairs}


def svd_basis(states: list[Tensor], rank: int) -> Tensor:
    """Uncentered TRAIN-state SVD; locally derived basis, not authors' supplied vectors."""
    if not states or rank < 1:
        raise ValueError("nonempty training states and positive rank required")
    x = torch.cat([h.detach().cpu().float().reshape(-1, h.shape[-1]) for h in states])
    if not torch.isfinite(x).all():
        raise ValueError("nonfinite training activations")
    _, singular, vectors = torch.linalg.svd(x, full_matrices=False)
    numerical_rank = int((singular > singular[0] * max(x.shape) * torch.finfo(x.dtype).eps).sum())
    if numerical_rank == 0:
        raise ValueError("training states have zero numerical rank")
    return vectors[:min(rank, numerical_rank)].contiguous()


def dcm_loss(logits: Tensor, target: int, mask: Tensor, sparsity: float):
    if sparsity < 0 or logits.ndim != 2:
        raise ValueError("nonnegative sparsity and [batch,vocab] logits required")
    # Matches official single-layer task objective: negative raw logit, NOT CE.
    return -logits[:, target].mean() + sparsity * mask.abs().sum().to(logits.device)


def fit_mask(model, examples, basis: Tensor, layer: int, hypothesis: str, *,
             epochs=1, learning_rate=.1, sparsity=.1, seed=20260915):
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("freeze all language-model parameters before mask training")
    if not examples or epochs < 1 or learning_rate <= 0:
        raise ValueError("nonempty training cohort and positive optimizer settings required")
    device = next(model.parameters()).device
    basis = basis.to(device)
    mask = torch.nn.Parameter(torch.ones(len(basis), device=device))
    optimizer = torch.optim.Adam([mask], lr=learning_rate)
    rng, losses = random.Random(seed), []
    for epoch in range(epochs):
        order = list(range(len(examples))); rng.shuffle(order)
        total = 0.
        for index in order:
            e = examples[index]
            if hypothesis not in e["pair"].hypotheses:
                raise ValueError("missing intervention target: " + hypothesis)
            target_name = e["pair"].hypotheses[hypothesis]
            target = e["candidate_ids"][e["candidates"].index(target_name)]
            optimizer.zero_grad(set_to_none=True)
            logits = patch_logits(model, e["base"], layer, e["base_sites"], e["donor_state"],
                                  basis, mask, e.get("restore_states"))
            loss = dcm_loss(logits, target, mask, sparsity)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite DCM loss")
            loss.backward(); optimizer.step()
            with torch.no_grad():
                mask.clamp_(0, 1)
            total += float(loss.detach())
            del logits, loss
        losses.append(total / len(order))
        print(f"DCM epoch {epoch + 1}/{epochs}: loss={losses[-1]:.5f}, "
              f"rounded rank={int(mask.detach().round().sum())}/{len(mask)}", flush=True)
    return mask.detach().cpu(), losses
