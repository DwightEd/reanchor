"""Real block-output interventions. No JVP, logit lens, or attention replacement.

Adapted from the independently written lookback_patching_demo.py supplied with
this discussion. Supports dense Llama-style model.model.layers on one device.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import torch
from torch import Tensor


def hidden(output) -> Tensor:
    value = output if isinstance(output, Tensor) else output[0]
    if not isinstance(value, Tensor) or value.ndim != 3 or value.shape[0] != 1:
        raise TypeError("expected one unpadded block output [1, sequence, width]")
    return value


def last_logits(model, inputs: Mapping[str, Tensor]) -> Tensor:
    if model.training:
        raise ValueError("call model.eval() before causal comparisons")
    mask = inputs.get("attention_mask")
    if mask is not None and not bool(mask.all()):
        raise ValueError("this experiment uses one unpadded prompt per forward")
    output = model.model(**inputs, use_cache=False, return_dict=True)
    return model.lm_head(output.last_hidden_state[:, -1]).float()


def check_layers(model, requests):
    layers = model.model.layers
    if any(not isinstance(i, int) or not 0 <= i < len(layers) for i in requests):
        raise ValueError("zero-based layer outside observer")
    return layers


@torch.no_grad()
def capture_states(model, inputs, requests: Mapping[int, Sequence[int]]):
    """One prefill; keep only requested states on CPU, release all hook handles."""
    layers = check_layers(model, requests)
    cached, handles = {}, []
    try:
        for index, positions in requests.items():
            def save(module, args, output, i=index, pos=list(positions)):
                cached[i] = hidden(output)[:, pos].detach().cpu().clone()
            handles.append(layers[index].register_forward_hook(save))
        logits = last_logits(model, inputs).detach().cpu()
    finally:
        for handle in handles:
            handle.remove()
    if set(cached) != set(requests):
        raise RuntimeError("model bypassed requested layer hooks")
    return cached, logits


def subspace_mix(base: Tensor, donor: Tensor, basis: Tensor, mask: Tensor) -> Tensor:
    """P=(diag(m)B)^T diag(m)B; compute factorized, without a dense D x D P.

    This preserves the official soft-mask SQUARED coefficient. For binary m and
    orthonormal rows B, P is an orthogonal projector. Float32 projection prevents
    low-precision mask multiplication from needlessly losing gradient precision.
    """
    if (base.shape != donor.shape or basis.ndim != 2 or basis.shape[1] != base.shape[-1]
            or mask.shape != (basis.shape[0],)):
        raise ValueError("aligned states, basis [K,D] and mask [K] required")
    b, m = basis.to(device=base.device, dtype=torch.float32), mask.to(base.device).float()
    delta = donor.to(base.device).float() - base.float()
    return (base.float() + ((delta @ b.T) * m.square()) @ b).to(base.dtype)


def patch_logits(model, inputs, layer: int, positions: Sequence[int], donor: Tensor,
                 basis: Tensor | None = None, mask: Tensor | None = None,
                 restore: Mapping[int, tuple[Sequence[int], Tensor]] | None = None):
    """Patch specified token positions, then run the ORIGINAL remaining network.

    Optional restore tensors come from the baseline run. Restoration takes
    precedence at its explicitly declared site. Overlap at the injection layer
    is rejected to avoid silently erasing the intervention.
    """
    if (basis is None) != (mask is None):
        raise ValueError("supply both basis and mask, or neither")
    restore = restore or {}
    layers = check_layers(model, [layer, *restore])
    if layer in restore and set(positions) & set(restore[layer][0]):
        raise ValueError("restoration overlaps the patch at the injection layer")
    handles = []
    def edit(module, args, output):
        h = hidden(output)
        original = h[:, list(positions)]
        replacement = donor.to(original)
        if replacement.shape != original.shape:
            raise ValueError("base and donor patch tensors differ in shape")
        if basis is not None:
            replacement = subspace_mix(original, replacement, basis, mask)
        updated = h.clone()
        updated[:, list(positions)] = replacement
        return updated if isinstance(output, Tensor) else (updated, *output[1:])
    try:
        handles.append(layers[layer].register_forward_hook(edit))
        for index, (pos, value) in restore.items():
            def keep(module, args, output, p=list(pos), v=value):
                h = hidden(output)
                if h[:, p].shape != v.shape:
                    raise ValueError("restoration tensor has wrong shape")
                updated = h.clone(); updated[:, p] = v.to(h)
                return updated if isinstance(output, Tensor) else (updated, *output[1:])
            handles.append(layers[index].register_forward_hook(keep))
        return last_logits(model, inputs)
    finally:
        for handle in handles:
            handle.remove()


def summarize_logits(logits: Tensor, candidate_ids: list[int], candidates: list[str], tokenizer):
    z = logits.detach().float().cpu()[0]
    ids = torch.as_tensor(candidate_ids, dtype=torch.long)
    logp = z.log_softmax(-1)
    pred = int(z.argmax())
    text = tokenizer.decode([pred]).strip().lower()
    return dict(prediction_id=pred, prediction=text,
                candidate_logits=z[ids].tolist(), candidate_probabilities=logp[ids].exp().tolist(),
                candidate_margin=[float(z[i] - torch.cat((z[:i], z[i + 1:])).max()) for i in ids],
                matches={name: text == name.strip().lower() for name in candidates})
