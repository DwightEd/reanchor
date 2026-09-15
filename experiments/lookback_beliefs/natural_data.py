"""Original sampled RAG trajectories; never borrow labels from a different answer."""
import json
from pathlib import Path

import numpy as np


DEFAULT_SAMPLES = 'outputs/samples_20260911_145421_235'


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def resolve_samples(path, cases):
    path = Path(path)
    if (path / 'samples.jsonl').is_file():
        return path
    required = {(c['source_id'], c['seed']) for c in cases}
    matches = []
    for index in sorted(Path('outputs').glob('samples_*/samples.jsonl')):
        rows = jsonl(index)
        if required <= {(str(r['source_id']), r['seed']) for r in rows}:
            matches.append(index.parent)
    if len(matches) != 1:
        raise FileNotFoundError(f'Specify --samples. Found {len(matches)} possible saved runs: {matches}')
    return matches[0]


def load_trace(samples, states, source_id, seed):
    rows = [r for r in jsonl(Path(samples) / 'samples.jsonl')
            if str(r['source_id']) == str(source_id) and r['seed'] == seed]
    if len(rows) != 1:
        raise ValueError(f'expected exactly one original {source_id}/seed{seed}, found {len(rows)}')
    row = rows[0]
    name = row['trace']
    if Path(name).name != name or not name.endswith('.npz'):
        raise ValueError('trace must be a local NPZ filename')
    with np.load(Path(samples) / name, allow_pickle=False) as f:
        trace = {k: f[k].copy() for k in ('token_ids', 'prompt_length', 'top_ids', 'top_logits',
                                          'log_normalizer', 'attention')}
    with np.load(Path(states) / name, allow_pickle=False) as f:
        for key in ('token_ids', 'hidden', 'source_mask', 'logit_entropy'):
            if key not in f:
                raise ValueError(f'{name}: missing saved state field {key}; use the existing fixed-prefix state cache')
        if not np.array_equal(f['token_ids'], trace['token_ids']):
            raise ValueError('saved states belong to a different generated sequence')
        trace.update(hidden=f['hidden'].copy(), source_mask=f['source_mask'].copy(),
                     logit_entropy=f['logit_entropy'].copy())
    p, n = int(trace['prompt_length']), len(trace['token_ids'])
    if not 0 < p < n or trace['token_ids'].ndim != 1 or not np.issubdtype(trace['token_ids'].dtype, np.integer):
        raise ValueError('original token IDs and prompt boundary must be valid')
    if trace['source_mask'].dtype != np.bool_:
        raise ValueError('saved evidence source_mask must be boolean')
    a, h = trace['attention'], trace['hidden']
    if (a.ndim != 4 or a.shape[2:] != (n-p, n) or h.ndim != 3
            or h.shape[:2] != (a.shape[0]+1, n-1) or trace['source_mask'].shape != (p,)
            or trace['top_ids'].shape != trace['top_logits'].shape
            or trace['top_ids'].shape[0] != n-p or trace['logit_entropy'].shape != (n-p,)):
        raise ValueError('expected original [L,H,response,key] attention and fixed-prefix hidden[L+1,N-1,D]')
    if (not all(np.isfinite(trace[k]).all() for k in ('hidden', 'attention', 'logit_entropy', 'top_logits', 'log_normalizer'))
            or np.any(a < 0)):
        raise ValueError('nonfinite original cache')
    return row, trace


def locate(text, offsets, quote, lower, upper, *, last=False):
    matches, begin = [], lower
    while True:
        start = text.find(quote, begin, upper)
        if start < 0:
            break
        matches.append((start, start+len(quote))); begin = start+1
    if not matches or (len(matches) != 1 and not last):
        raise ValueError(f'quote is missing or ambiguous: {quote!r}; matches={len(matches)}')
    start, end = matches[-1] if last else matches[0]
    hit = np.flatnonzero((offsets[:, 0] < end) & (offsets[:, 1] > start)
                         & (offsets[:, 1] > offsets[:, 0]))
    if not len(hit):
        raise ValueError('quote did not align to original tokens')
    return hit, (start, end)


def prepare_case(case, row, trace, tokenizer, context=8):
    ids, p = trace['token_ids'].tolist(), int(trace['prompt_length'])
    # Recover offsets only if retokenization exactly reproduces ALL saved IDs.
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded['input_ids'] != ids:
        raise ValueError('decoded text does not round-trip to original token IDs; do not silently realign')
    offsets = np.asarray(encoded['offset_mapping'])
    boundary = int(offsets[p, 0])
    observed = tokenizer.decode(ids[p:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if observed.strip() != row['response'].strip():
        raise ValueError('samples.jsonl response does not match saved tokens')
    target, target_chars = locate(text, offsets, case['target'], boundary, len(text))
    carriers, _ = locate(text, offsets, case['carrier'], boundary, target_chars[0], last=True)
    # Only one earlier emitted carrier; never inject the target token or final readout position.
    carrier = int(carriers[-1])
    if carrier >= int(target[0])-1:
        raise ValueError('carrier must be strictly earlier than the first target prediction query')
    seed, _ = locate(text, offsets, case['history_seed'], *target_chars)
    groups = {}
    for name in ('constraint', 'payload', 'control_pool'):
        positions = []
        for quote in case[name]:
            selected, _ = locate(text, offsets, quote, 0, boundary)
            positions.extend(selected.tolist())
        groups[name] = np.unique(positions)
        if not trace['source_mask'][groups[name]].all():
            raise ValueError(f'{name} quote crosses the saved evidence boundary')
    # Drop shared boundary tokens from the role-specific group to keep factors disjoint.
    overlap = np.intersect1d(groups['constraint'], groups['payload'])
    groups['constraint'] = np.setdiff1d(groups['constraint'], overlap)
    if not len(groups['constraint']) or not len(groups['payload']):
        raise ValueError('need nonempty, separable constraint and payload token sets')
    pool = np.setdiff1d(groups.pop('control_pool'), np.union1d(groups['constraint'], groups['payload']))
    for name in ('constraint', 'payload'):
        if len(pool) < len(groups[name]):
            raise ValueError('control pool too short for token-count-matched source removal')
        groups['control_'+name] = pool[:len(groups[name])]
    history = seed[:1]  # one emitted token, not a future-completed phrase
    h_start = int(history[-1]) + 1
    h_control = np.arange(int(target[0])-len(history), int(target[0]))
    if h_control[0] < p:
        raise ValueError('no past response control')
    groups['history_seed'], groups['history_control'] = history, h_control
    start = max(0, int(target[0])-p-context)
    stop = min(len(ids)-p, int(target[-1])-p+1+context)
    steps = np.arange(start, stop)
    # A second site INSIDE the perturbed region, not a trivially unchanged earlier site.
    control_carrier = int(target[0])-2
    if control_carrier <= carrier:
        raise ValueError('need a distinct perturbed carrier-control position before the target query')
    return dict(case=case, record=row, prompt_length=p, ids=ids[:p+stop-1], steps=steps,
                queries=p+steps-1, target_steps=target-p, carrier=carrier, control_carrier=control_carrier,
                history_query_start=h_start-1, source_query_start=carrier,
                groups=groups, token_text=[tokenizer.decode([ids[p+t]]) for t in steps],
                full_text=text, prompt_text=text[:boundary], original_ids=ids, original_response=row['response'])


def cache_measurements(item, trace):
    """Actual cached per-head reads and state geometry; no truth classifier."""
    p, steps, q = item['prompt_length'], item['steps'], item['queries']
    a = trace['attention'][:, :, steps, :].astype(np.float32)
    result = dict(steps=steps, query_positions=q, entropy_bits=trace['logit_entropy'][steps],
                  top_ids=trace['top_ids'][steps], top_logits=trace['top_logits'][steps])
    for name, indices in item['groups'].items():
        result['mass_'+name] = a[..., indices].sum(-1)
    prompt = a[..., :p].copy()
    prompt[..., ~trace['source_mask']] = 0
    mass = prompt.sum(-1)
    prob = np.divide(prompt, mass[..., None], out=np.zeros_like(prompt), where=mass[..., None]>0)
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy = -np.where(prob>0, prob*np.log2(np.maximum(prob, 1e-38)), 0).sum(-1)
    result['effective_source_count'] = np.where(mass>0, 2**entropy, np.nan)
    result['source_mass'] = mass
    # Stored hidden[0] is embedding; hidden[L] is FINAL NORMALIZED, not last block output.
    h = trace['hidden'][:, q].astype(np.float32)
    c = trace['hidden'][:, item['carrier']].astype(np.float32)
    norm = np.linalg.norm(h, axis=-1)*np.linalg.norm(c, axis=-1)[:, None]
    result['carrier_cosine_by_hidden_index'] = np.divide(
        (h*c[:, None]).sum(-1), norm, out=np.full_like(norm, np.nan), where=norm>0)
    result['source_attention_to_history_seed'] = a[..., item['groups']['history_seed']].sum(-1)
    return result
