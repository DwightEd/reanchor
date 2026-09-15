"""Keep replay mismatch separate from causal effects. No tolerance relaxation."""
import numpy as np
import torch


def compare_replay(result, trace, item):
    z = result['logits'].detach().float().cpu()
    steps = np.asarray(item['steps'], dtype=np.int64)
    saved = torch.as_tensor(trace['top_ids'][steps], dtype=torch.long)
    original = np.asarray(trace['top_logits'][steps])
    current = z.gather(1, saved).numpy()
    dz = current - original
    dlse = z.logsumexp(-1).numpy() - np.asarray(trace['log_normalizer'][steps])
    # Common logit shifts cancel in log probabilities. Report both, without
    # using that invariance to silently accept a failed historical replay.
    dlp = dz - dlse[:, None]
    finite = bool(np.isfinite(dz).all() and np.isfinite(dlse).all())
    att = np.asarray(result.get('cached_attention_errors', []))
    if len(att) not in (0, len(steps)):
        raise ValueError('attention check length does not match saved prediction steps')
    finite = finite and bool(np.isfinite(att).all())
    rows = [dict(step=int(t), query=int(item['queries'][i]),
                 max_top_logit_error=float(np.abs(dz[i]).max()),
                 log_normalizer_delta=float(dlse[i]),
                 max_top_logp_error=float(np.abs(dlp[i]).max()),
                 saved_top1=int(saved[i, 0]), replay_top1=int(z[i].argmax()),
                 attention_error=float(att[i]) if len(att) else None)
            for i, t in enumerate(steps)]
    return dict(execution=result.get('execution', item.get('execution', 'full')),
                finite=finite, saved_top_logit_error=float(np.abs(dz).max()),
                saved_log_normalizer_error=float(np.abs(dlse).max()),
                saved_top_logp_error=float(np.abs(dlp).max()),
                saved_attention_error=float(att.max()) if len(att) else None,
                intervention_complete=False,
                replay_note='mathematically equal prefixes can differ across kernels/batching/dtypes; check the actual replay'), rows
