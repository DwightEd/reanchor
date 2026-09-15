"""Generation-matched finite interventions, with fresh KV memory in EVERY world.

Replay exactly the stored prefix: prompt prefill, then one saved token per call.
Use native eager attention weights rather than rebuilding QK using different
matrix shapes. No sampling, no answer injection, no tolerance increase.
"""
import numpy as np
import torch

from .patching import hidden


@torch.inference_mode()
def forward_incremental(model, item, layers, *, cut_keys=(), cut_start=None,
                        patch=None, strength=1.):
    from .natural_engine import projected_cut

    cfg = model.config
    blocks = model.model.layers
    if model.training:
        raise ValueError('call model.eval() before a causal comparison')
    if (cfg.model_type != 'llama' or getattr(cfg, 'pretraining_tp', 1) != 1
            or getattr(cfg, 'sliding_window', None) or not 0 <= strength <= 1):
        raise ValueError('requires dense Llama/GQA and strength in [0,1]')
    if any(not 0 <= l < len(blocks)-1 for l in layers):
        raise ValueError('use non-final block layers; final saved hidden is normalized')
    if patch is not None and patch[0] not in layers:
        raise ValueError('patch layer must be one of the recorded non-final layers')
    p, n = int(item['prompt_length']), len(item['ids'])
    q = np.asarray(item['queries'], dtype=np.int64)
    if (not 0 < p <= n or q.ndim != 1 or not len(q)
            or q[0] < p-1 or q[-1] >= n or np.any(np.diff(q) <= 0)):
        raise ValueError('queries must be ordered prediction positions in the saved prefix')
    first = p + int(min(item['target_steps'])) - 1
    carrier = int(item['carrier'])
    control = int(item.get('control_carrier', carrier))
    if not p <= carrier < first or not p <= control < first:
        raise ValueError('carriers must precede the target readout, in emitted history')
    keys = np.asarray(cut_keys, dtype=np.int64)
    if len(keys) and (cut_start is None or not p <= cut_start < n
                     or keys.min() < 0 or keys.max() >= n
                     or len(np.unique(keys)) != len(keys)):
        raise ValueError('invalid original cut coordinates')
    site = int(patch[2]) if patch is not None and len(patch) == 3 else carrier
    if patch is not None and not p <= site < first:
        raise ValueError('patch site must precede the target prediction')
    device = model.get_input_embeddings().weight.device
    ids = torch.as_tensor(item['ids'], device=device).reshape(1, -1)
    handles, states, control_states, errors, codes = [], {}, {}, [], {}
    call = {'start': 0, 'stop': p}
    # Extra V buffers only when cutting. Shapes are [KV heads, visible keys, d].
    # The native KV cache remains private to this forward; it is never reused
    # across baseline/cut/restore worlds, even when their token IDs are identical.
    def attach(index, block):
        store = {}
        if len(keys):
            attn = block.self_attn
            if getattr(attn, 'q_norm', None) is not None or getattr(attn, 'k_norm', None) is not None:
                raise ValueError('QK-normalized variants need a verified observer')
            dim = getattr(attn, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
            def read_value(module, args, output):
                value = output.detach().reshape(1, output.shape[1], -1, dim)[0].transpose(0, 1)
                old = store.get('values')
                store['values'] = value.clone() if old is None else torch.cat((old, value), dim=1)
            def read_context(module, args):
                store['context'] = args[0].detach()
            def remove(module, args, output):
                # Current eager Llama returns (attention_output, weights[, cache]).
                if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                    raise ValueError('native eager attention weights unavailable; set attn_implementation=eager')
                original, attention = output[0], output[1]
                if attention.ndim != 4 or attention.shape[0] != 1:
                    raise ValueError('expected batch-one native attention')
                begin, end = call['start'], call['stop']
                if attention.shape[-2:] != (end-begin, end):
                    raise ValueError('native KV attention and absolute query/key coordinates disagree')
                if begin < cut_start:
                    store.pop('context', None)
                    return None  # prefill and pre-cut history are untouched
                visible_keys = keys[keys < end].tolist()
                if not visible_keys:
                    store.pop('context', None)
                    return None
                values = store['values']
                heads = attention.shape[1]
                if values.shape[1] != end or heads % values.shape[0]:
                    raise ValueError('V buffer or GQA groups disagree with the native cache')
                values = values.repeat_interleave(heads // values.shape[0], dim=0)
                weights = attention[0]
                rebuilt = (weights.to(values.dtype) @ values).transpose(0, 1).reshape(1, end-begin, -1)
                context = store.pop('context')
                rel = float((rebuilt.float()-context.float()).norm() / context.float().norm().clamp_min(1e-9))
                if not np.isfinite(rel) or rel > .02:
                    raise ValueError(f'native attention/value reconstruction disagrees at layer {index}: {rel}')
                errors.append(rel)
                delta, code = projected_cut(weights, values, visible_keys, module.o_proj.weight)
                changed = original.clone()
                changed[0] -= strength * delta.to(changed)
                if begin in (carrier, first):
                    codes.setdefault(index, []).append((begin, code[:, 0].cpu().float().numpy()))
                return (changed, *output[1:])
            handles.append(attn.v_proj.register_forward_hook(read_value))
            handles.append(attn.o_proj.register_forward_pre_hook(read_context))
            handles.append(attn.register_forward_hook(remove))
        if patch is not None and index == patch[0]:
            def replace(module, args, output):
                if not call['start'] <= site < call['stop']:
                    return None
                h = hidden(output)
                local = site - call['start']
                if patch[1].shape != h[:, local:local+1].shape:
                    raise ValueError('carrier replacement shape mismatch')
                changed = h.clone()
                changed[:, local:local+1] = patch[1].to(h)
                return changed if isinstance(output, torch.Tensor) else (changed, *output[1:])
            handles.append(block.register_forward_hook(replace))
        if index in layers:
            def capture(module, args, output):
                h = hidden(output)
                for pos, dest in ((carrier, states), (control, control_states)):
                    if call['start'] <= pos < call['stop']:
                        local = pos - call['start']
                        dest[index] = h[:, local:local+1].detach().cpu().clone()
            handles.append(block.register_forward_hook(capture))

    cache = None
    rows, attention_errors = [], []
    wanted = set(q.tolist())
    try:
        for index, block in enumerate(blocks):
            attach(index, block)
        for stop in range(p, int(q[-1])+2):
            start = 0 if stop == p else stop-1
            call.update(start=start, stop=stop)
            output = model(input_ids=ids[:, start:stop], past_key_values=cache,
                           use_cache=True, attention_mask=torch.ones_like(ids[:, :stop]),
                           output_attentions=True, return_dict=True)
            if output.past_key_values is None:
                raise ValueError('model did not return the requested native KV cache')
            query = stop-1
            if query in wanted:
                rows.append(output.logits[0, -1].float().cpu())
                # Old attention was stored as float16. Compare in the SAME
                # storage representation, not float16 old against float32 new.
                if 'cached_attention' in item:
                    current = torch.stack([a[0, :, -1] for a in output.attentions])
                    current = current.to('cpu', dtype=torch.float16).float().numpy()
                    expected = item['cached_attention'][:, :, query-p+1, :stop].astype(np.float32)
                    if current.shape != expected.shape:
                        raise ValueError('cached attention does not align with replay')
                    attention_errors.append(float(np.max(np.abs(current-expected))))
            cache = output.past_key_values
            del output
        if len(rows) != len(q) or set(states) != set(layers) or set(control_states) != set(layers):
            raise ValueError('replay failed to visit every requested query/carrier')
        return dict(logits=torch.stack(rows), states=states, control_states=control_states,
                    replay_error=max(errors, default=0.), removed_codes=codes,
                    execution='incremental_kv',
                    cached_attention_errors=np.asarray(attention_errors, dtype=float))
    finally:
        for handle in handles:
            handle.remove()
        del cache
