"""Actual causal-mask interventions and attention/MLP residual readouts."""

import numpy as np
import torch


def attention_mask(length, prompt_length, blocked=(), window=None, sinks=4):
    query, key = torch.arange(length)[:, None], torch.arange(length)[None, :]
    allowed = key <= query
    if window is not None:
        allowed &= (query - key < window) | (key < sinks)
    if len(blocked):
        cut = torch.zeros(length, dtype=torch.bool)
        cut[list(blocked)] = True
        allowed &= ~((query >= prompt_length - 1) & cut[None, :])
    allowed.fill_diagonal_(True)
    return torch.where(allowed, 0.0, torch.finfo(torch.float32).min)[None, None]


def local_to_remote(attention, queries, source_mask, window=16):
    """Raw attention redistribution; both steps use the current key partition."""
    result = np.zeros(len(queries))
    positions = np.arange(attention.shape[-1])
    source = np.zeros(len(positions), dtype=bool)
    source[: len(source_mask)] = source_mask
    for t in range(1, len(queries)):
        common = positions < queries[t - 1]
        history = (positions >= len(source_mask)) & common
        local = history & (queries[t] - positions < window)
        remote = common & (source | (history & ~local))
        difference = attention[:, :, t].astype(np.float64) - attention[:, :, t - 1]
        gain = np.maximum(0, difference[..., remote].sum(-1))
        loss = np.maximum(0, -difference[..., local].sum(-1))
        result[t] = np.minimum(gain, loss).mean()
    return result


def history_mask(length, prompt_length, window, recent):
    mask = attention_mask(length, prompt_length)
    query, key = torch.arange(length)[:, None], torch.arange(length)[None, :]
    history = (key >= prompt_length) & (key < query)
    distance = query - key < window
    cut = history & (distance if recent else ~distance)
    mask[0, 0][cut] = torch.finfo(torch.float32).min
    return mask


class CausalReadout:
    def __init__(self, model):
        self.model = model

    @torch.inference_mode()
    def run(self, token_ids, queries, candidates, mask=None, mlp_patch=None):
        model = self.model
        queries = np.asarray(queries)
        layers = model.model.layers
        states = np.empty((len(layers), 3, len(queries), model.config.hidden_size), np.float32)
        updates = np.empty_like(states[:, 0])
        handles = []

        def before(layer):
            def hook(module, args, kwargs):
                hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
                states[layer, 0] = hidden[0, queries].float().cpu().numpy()

            return hook

        def after_attention(layer):
            def hook(module, args):
                states[layer, 1] = args[0][0, queries].float().cpu().numpy()

            return hook

        def mlp(layer):
            def hook(module, args, output):
                if mlp_patch is not None and len(layers) // 3 <= layer < 2 * len(layers) // 3:
                    output = output.clone()
                    output[0, queries] = torch.as_tensor(
                        mlp_patch[layer], device=output.device, dtype=output.dtype
                    )
                updates[layer] = output[0, queries].float().cpu().numpy()
                return output

            return hook

        def after(layer):
            def hook(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                states[layer, 2] = hidden[0, queries].float().cpu().numpy()

            return hook

        for index, layer in enumerate(layers):
            handles.extend(
                [
                    layer.register_forward_pre_hook(before(index), with_kwargs=True),
                    layer.post_attention_layernorm.register_forward_pre_hook(
                        after_attention(index)
                    ),
                    layer.mlp.register_forward_hook(mlp(index)),
                    layer.register_forward_hook(after(index)),
                ]
            )
        ids = torch.as_tensor(token_ids, device=model.device)[None]
        kwargs = {} if mask is None else {"attention_mask": mask.to(model.device, model.dtype)}
        try:
            output = model(ids, use_cache=False, output_attentions=True, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        logits = output.logits[0, queries].float()
        logp = logits.log_softmax(-1)
        entropy = -(logp.exp() * logp).sum(-1)
        weights = torch.stack([a[0, :, queries] for a in output.attentions])
        candidate_ids = torch.as_tensor(candidates, device=model.device)
        directions = model.lm_head.weight[candidate_ids[:, 0]].float()
        directions -= model.lm_head.weight[candidate_ids[:, 1]].float()
        directions *= model.model.norm.weight.float()
        normalized = states / np.sqrt(
            np.mean(states**2, axis=-1, keepdims=True) + model.model.norm.variance_epsilon
        )
        contrast = np.einsum("lsqd,qd->lsq", normalized, directions.cpu().numpy())
        return dict(
            logits=logits.cpu().numpy(),
            entropy=entropy.cpu().numpy(),
            attention=weights.to("cpu", dtype=torch.float16).numpy(),
            layer_contrast=contrast,
            mlp_updates=updates,
        )
