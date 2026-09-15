"""Finite message cuts and carrier restoration on the ORIGINAL fixed token stream.

Cut is V-path removal: subtract W_O sum_{j in group} A[q,j] V[j] without
renormalizing attention. QK remains native in that layer; downstream layers adapt.
This is not source deletion, a complete information erasure, JVP, or truth scoring.
"""
import inspect

import numpy as np
import torch
import torch.nn.functional as F

from .patching import hidden


def projected_cut(weights, values, keys, output_weight):
    """weights [H,Q,N], values [H,N,d], W_O [D,H*d]; no head averaging."""
    code = weights[..., keys].to(values.dtype) @ values[:, keys]
    concatenated = code.transpose(0, 1).reshape(code.shape[1], -1)
    return F.linear(concatenated, output_weight, bias=None), code


def _attention_rows(module, storage, args, kwargs, config, queries):
    bound = inspect.signature(module.forward).bind_partial(*args, **kwargs).arguments
    qraw, kraw, vraw = (storage[k] for k in ('q', 'k', 'v'))
    head_count = config.num_attention_heads
    dim = getattr(module, 'head_dim', config.hidden_size//head_count)
    shape = (1, qraw.shape[1], -1, dim)
    q, k, v = [x.reshape(shape).transpose(1, 2) for x in (qraw, kraw, vraw)]
    pos = bound.get('position_embeddings', kwargs.get('position_embeddings'))
    if pos is None:
        position_ids = bound.get('position_ids', kwargs.get('position_ids'))
        if not hasattr(module, 'rotary_emb') or position_ids is None:
            raise ValueError('native RoPE values unavailable; no approximate-position fallback')
        pos = module.rotary_emb(v, position_ids)
    cos, sin = (x.unsqueeze(1) for x in pos)
    def rotate(x):
        a, b = x.chunk(2, -1)
        return torch.cat((-b, a), -1)
    q, k = q*cos+rotate(q)*sin, k*cos+rotate(k)*sin
    repeats = head_count//k.shape[1]
    k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
    qr = torch.as_tensor(queries, device=q.device)
    scale = getattr(module, 'scaling', dim**-.5)
    z = (q[:, :, qr] @ k.transpose(-1, -2))*scale
    mask = torch.arange(k.shape[-2], device=q.device)[None] > qr[:, None]
    a = z.masked_fill(mask, -torch.inf).float().softmax(-1)[0]
    rebuilt = (a.to(v.dtype)@v[0]).transpose(0, 1).reshape(len(qr), -1)
    native = storage['context'][0, qr]
    rel = float((rebuilt.float()-native.float()).norm()/native.float().norm().clamp_min(1e-9))
    if not np.isfinite(rel) or rel > .02:
        raise ValueError(f'QK/OV reconstruction does not match native computation: relative error={rel}')
    return a, v[0], rel


@torch.inference_mode()
def forward(model, item, layers, *, cut_keys=(), cut_start=None, patch=None, strength=1.):
    """One teacher-forced replay, collecting small carrier states and score windows.

    patch=(layer, tensor[1,1,D], optional_site) defaults to item['carrier'].
    A control site precedes the target but lies inside the perturbed region; it is not
    assumed to be semantically irrelevant or exactly distance matched.
    No old KV cache is reused across intervention worlds. Hooks always removed.
    """
    cfg = model.config
    if (cfg.model_type != 'llama' or getattr(cfg, 'pretraining_tp', 1) != 1
            or getattr(cfg, 'sliding_window', None) or not 0 <= strength <= 1):
        raise ValueError('requires standard dense Llama/GQA and intervention strength in [0,1]')
    blocks = model.model.layers
    if any(not 0 <= l < len(blocks)-1 for l in layers):
        raise ValueError('use non-final block layers; saved final hidden is normalized, not block output')
    if patch is not None and patch[0] not in layers:
        raise ValueError('patch layer must be one of the recorded non-final layers')
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([item['ids']], device=device)
    q = np.asarray(item['queries'])
    carrier = int(item['carrier'])
    if carrier >= int(item['prompt_length']+min(item['target_steps'])-1):
        raise ValueError('cannot patch the scored target readout as a carrier')
    if len(cut_keys) and (cut_start is None or np.min(cut_keys)<0 or np.max(cut_keys)>=len(item['ids'])):
        raise ValueError('invalid original cut coordinates')
    if len(cut_keys) and (not 0 <= cut_start < len(item['ids']) or len(set(map(int,cut_keys))) != len(cut_keys)):
        raise ValueError('cut keys must be unique and cut_start in the visible prefix')
    cut_queries = np.arange(cut_start, len(item['ids'])) if len(cut_keys) else np.array([], int)
    handles, states, control_states, replay_errors, removed_codes = [], {}, {}, [], {}
    control_site = item.get('control_carrier', carrier)
    patch_site = patch[2] if patch is not None and len(patch)==3 else carrier
    if patch is not None and not item['prompt_length'] <= patch_site < item['prompt_length']+min(item['target_steps'])-1:
        raise ValueError('patch site must be an earlier emitted token, not a target query')
    try:
        for li, block in enumerate(blocks):
            if len(cut_keys):
                attn, store = block.self_attn, {}
                if getattr(attn, 'q_norm', None) is not None or getattr(attn, 'k_norm', None) is not None:
                    raise ValueError('QK-normalized variants need a verified observer')
                for name in ('q', 'k', 'v'):
                    def collect(module, args, output, s=store, key=name):
                        s[key] = output.detach()
                    handles.append(getattr(attn, name+'_proj').register_forward_hook(collect))
                def context(module, args, s=store):
                    s['context'] = args[0].detach()
                handles.append(attn.o_proj.register_forward_pre_hook(context))
                def remove(module, args, kwargs, output, s=store, index=li):
                    original = output if isinstance(output, torch.Tensor) else output[0]
                    changed = original.clone()
                    codes = []
                    for start in range(0, len(cut_queries), 16):
                        qr = cut_queries[start:start+16]
                        a, v, err = _attention_rows(module, s, args, kwargs, cfg, qr)
                        delta, code = projected_cut(a, v, list(cut_keys), module.o_proj.weight)
                        changed[0, qr] -= strength*delta.to(changed)
                        replay_errors.append(err)
                        # Small exact pre-WO per-head codes at the carrier and first fact query.
                        first = item['prompt_length']+min(item['target_steps'])-1
                        for j, query in enumerate(qr):
                            if query in (carrier, first):
                                codes.append((int(query), code[:, j].cpu().float().numpy()))
                    if codes:
                        removed_codes[index] = codes
                    s.clear()
                    return changed if isinstance(output, torch.Tensor) else (changed, *output[1:])
                handles.append(attn.register_forward_hook(remove, with_kwargs=True))
            if patch is not None and li == patch[0]:
                def replace(module, args, output):
                    h = hidden(output); new = h.clone()
                    if patch[1].shape != h[:, patch_site:patch_site+1].shape:
                        raise ValueError('carrier replacement shape mismatch')
                    new[:, patch_site:patch_site+1] = patch[1].to(h)
                    return new if isinstance(output, torch.Tensor) else (new, *output[1:])
                handles.append(block.register_forward_hook(replace))
            if li in layers:
                def capture(module, args, output, layer=li):
                    states[layer] = hidden(output)[:, carrier:carrier+1].cpu().clone()
                    control_states[layer] = hidden(output)[:, control_site:control_site+1].cpu().clone()
                handles.append(block.register_forward_hook(capture))
        output = model.model(input_ids=ids, use_cache=False, output_hidden_states=False,
                             output_attentions=False, return_dict=True)
        h = output.last_hidden_state[0]
        z = torch.cat([model.lm_head(h[q[i:i+16]]).float().cpu()
                       for i in range(0, len(q), 16)])
        return dict(logits=z, states=states, control_states=control_states, replay_error=max(replay_errors, default=0.),
                    removed_codes=removed_codes)
    finally:
        for handle in handles:
            handle.remove()


def score(logits, saved_ids, alternatives, baseline=None):
    z = logits.float(); lp = z.log_softmax(-1); index = torch.arange(len(z))
    saved_ids, alternatives = torch.as_tensor(saved_ids), torch.as_tensor(alternatives)
    chosen = lp[index, saved_ids]
    result = dict(saved_logp=chosen.numpy(), entropy_bits=(-(lp.exp()*lp).sum(-1)/np.log(2)).numpy(),
                  saved_vs_frozen_alternative=(z[index,saved_ids]-z[index,alternatives]).numpy(),
                  top1=z.argmax(-1).numpy(), alternative_ids=alternatives.numpy())
    if baseline is not None:
        b = baseline.float().log_softmax(-1)
        mixture = torch.logaddexp(lp, b)-np.log(2.)
        result.update(delta_saved_logp=(chosen-b[index,saved_ids]).numpy(),
                      js_to_baseline_bits=(.5*((lp.exp()*(lp-mixture)).sum(-1)
                                             +(b.exp()*(b-mixture)).sum(-1))/np.log(2)).numpy())
    return result


def cache_carrier(trace, layer, carrier, model_layers):
    # HF Llama output_hidden_states stores embeddings, pre-next-block states,
    # and final RMSNorm output. hidden[L] CANNOT patch block L-1.
    if not 0 <= layer < model_layers-1 or trace['hidden'].shape[0] != model_layers+1:
        raise ValueError('cached block index is invalid or refers to normalized final hidden')
    return torch.from_numpy(trace['hidden'][layer+1, carrier].copy())[None, None]
