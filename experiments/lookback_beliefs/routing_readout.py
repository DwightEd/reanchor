"""Read frozen routing-audit logits; no model forward, fitting or new interventions.

Run from reanchor: python -m experiments.lookback_beliefs.routing_readout --decode
Compatible with the original qk-binding-audit-v1 caches, independently of the
current capture code hash. Candidate preference is NOT full-vocabulary accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np


VERSION = 'routing-semantic-readout-v2'
ID = re.compile(r'^[A-Za-z0-9_-]+$')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def logsumexp(x):
    x = np.asarray(x, dtype=np.float64)
    maximum = x.max(axis=-1, keepdims=True)
    return (maximum + np.log(np.exp(x - maximum).sum(-1, keepdims=True))).squeeze(-1)


def log_probabilities(logits):
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or not len(z) or z.shape[1] < 2 or not np.isfinite(z).all():
        raise ValueError('expected finite [recorded sites, vocabulary] logits')
    return z - logsumexp(z)[:, None]


def checked_ids(values, upper, label):
    array = np.asarray(values)
    if (array.ndim != 1 or not len(array) or
            not np.issubdtype(array.dtype, np.integer) or
            (array < 0).any() or (array >= upper).any()):
        raise ValueError(f'invalid integer {label}')
    return array.astype(np.int64)


def read_trace(path, case, *, baseline=None, intervention_query=None):
    """Only load small readouts, not hidden states / all-head attention arrays."""
    with np.load(path, allow_pickle=False) as saved:
        result = {k: saved[k].copy() for k in ('logits', 'queries', 'anchors')}
        if baseline is None and 'case_signature' in saved:
            expected = hashlib.sha256(json.dumps(case, sort_keys=True,
                                                 ensure_ascii=False).encode()).hexdigest()
            if str(saved['case_signature']) != expected:
                raise ValueError('case metadata differs from capture signature')
        if 'query' in saved and intervention_query is not None:
            if int(saved['query']) != intervention_query:
                raise ValueError('intervention query differs from index')
    z = result['logits'] = np.asarray(result['logits'], dtype=np.float64)
    result['lp'] = log_probabilities(z)
    queries = checked_ids(result['queries'], len(case['ids']), 'queries')
    anchors = checked_ids(result['anchors'], int(case['prompt_length']), 'anchors')
    if (not np.array_equal(queries, case['queries']) or
            not np.array_equal(anchors, case['anchors']) or
            len(queries) != len(z) or np.any(np.diff(queries) <= 0) or
            (queries < int(case['prompt_length']) - 1).any()):
        raise ValueError('query/source coordinates differ from original case')
    names = case['site_names']
    if len(names) != len(z) or len(set(names)) != len(names):
        raise ValueError('site names must be unique and aligned')
    candidates = checked_ids(case['candidate_ids'], z.shape[1], 'candidate IDs')
    checked_ids(case['expected_tokens'], z.shape[1], 'expected IDs')
    if len(candidates) != 2 or len(set(candidates.tolist())) != 2:
        raise ValueError('this audit requires two distinct source candidate tokens')
    if len(case['expected_tokens']) != len(z):
        raise ValueError('expected tokens must align with all recorded sites')
    if case['kind'] == 'constructed' and case.get('correct_index') not in (0, 1):
        raise ValueError('constructed case needs its declared correct_index')
    if baseline is not None:
        if z.shape != baseline['logits'].shape:
            raise ValueError('vocabulary / site count differs from frozen baseline')
        for key in ('queries', 'anchors'):
            if not np.array_equal(result[key], baseline[key]):
                raise ValueError(f'{key} differs from frozen baseline')
    return result


def token_stats(z, lp, token, decode):
    if not 0 <= token < len(z):
        raise ValueError('requested token outside vocabulary')
    others = np.concatenate((z[:token], z[token + 1:]))
    return dict(id=int(token), text=decode(int(token)), probability=float(np.exp(lp[token])),
                logp=float(lp[token]), logodds_rest=float(z[token] - logsumexp(others)),
                rank=int(1 + np.count_nonzero(z > z[token])))


def site_stats(trace, case, site, decode):
    z, lp = trace['logits'][site], trace['lp'][site]
    ids = np.asarray(case['candidate_ids'], dtype=int)
    top = int(np.argmax(z)); difference = float(z[ids[1]] - z[ids[0]])
    choice = None if difference == 0 else int(difference > 0)
    truth = case.get('correct_index') if case['kind'] == 'constructed' else None
    row = dict(case=case['id'], group=case.get('group', ''), split=case.get('split', ''),
               kind=case['kind'], world=case.get('world', ''), site=case['site_names'][site],
               query=int(trace['queries'][site]),
               candidate_interpretation=('declared constructed alternatives' if truth is not None
                                         else 'source anchor words, NOT true/false answers'),
               correct_index=truth, candidate_choice=choice, candidate_tie=choice is None,
               candidate_preference_correct=(choice == truth if choice is not None and truth is not None else None),
               global_top1=top, global_top1_text=decode(top), global_top1_probability=float(np.exp(lp[top])),
               top1_outside_candidates=bool(top not in ids),
               global_correct=(top == int(ids[truth]) if truth is not None else None),
               candidate_total=float(np.exp(lp[ids]).sum()),
               logodds_1_over_0=difference,
               correct_candidate_margin=(float(z[ids[truth]] - z[ids[1-truth]]) if truth is not None else None),
               local_claim_supported=case.get('local_claim_supported'),
               entropy_bits=float(-np.sum(np.exp(lp) * lp) / np.log(2)))
    for i, token in enumerate(ids):
        row.update({f'candidate{i}_{k}': v for k, v in token_stats(z, lp, int(token), decode).items()})
    row.update({f'saved_{k}': v for k, v in token_stats(z, lp, int(case['expected_tokens'][site]), decode).items()})
    return row


def top_tokens(trace, site, k, decode):
    z, lp = trace['logits'][site], trace['lp'][site]
    ids = np.argsort(-z, kind='stable')[:k]
    return [dict(id=int(i), text=decode(int(i)), probability=float(np.exp(lp[i]))) for i in ids]


def effect_index(root):
    """Use frozen aggregate index, or per-arm JSON after an interrupted run."""
    file = root / 'intervention_effects.csv'
    if file.exists():
        with file.open(newline='', encoding='utf-8') as stream:
            return list(csv.DictReader(stream)), 'intervention_effects.csv'
    records = [read_json(p) for p in sorted((root / 'interventions').glob('*.json'))]
    return records, 'interventions/*.json'


def reference_donor(case, arm, cases):
    """Describe v1's actual donor, never infer a counterfactual from effect signs."""
    if case['kind'] != 'constructed' or arm not in ('q', 'q_subspace', 'q_visible', 'k', 'k_subspace', 'qk', 'v_content'):
        return {}
    c, b, content = case['c'], case['b'], case.get('content', 0)
    if arm.startswith('q') and arm != 'qk': c ^= 1
    elif arm.startswith('k'): b ^= 1
    elif arm == 'v_content': content = 1
    else: return dict(target_distinguishes_pointer_from_donor_content=None)
    matches = [d for d in cases.values() if d['kind'] == 'constructed' and d['group'] == case['group']
               and (d.get('c'), d.get('b'), d.get('o'), d.get('content', 0), d.get('style', 0))
               == (c, b, case['o'], content, 0)]
    if len(matches) != 1: return {}
    donor = matches[0]
    return dict(donor_case=donor['id'], donor_declared_answer=int(donor['candidate_ids'][donor['correct_index']]))


def control_comparisons(details):
    """Same case/site/channel comparisons, not tests of statistical significance."""
    keys = ('case', 'intervention_site', 'scored_site', 'layer', 'head')
    lookup = {(tuple(r[k] for k in keys), r['arm']): r for r in details}
    pairs = (('q_subspace', 'q_random'), ('k_subspace', 'k_random'),
             ('q', 'q_same_binding'), ('q_subspace', 'q_same_subspace'),
             ('q', 'q_paraphrase'), ('k', 'k_paraphrase'))
    result = []
    for r in details:
        for primary, control in pairs:
            if r['arm'] != primary: continue
            c = lookup.get((tuple(r[k] for k in keys), control))
            if c is None: continue
            row = {k: r[k] for k in keys}
            row.update(primary=primary, control=control, temporal_scope=r['temporal_scope'],
                       interpretation='paired descriptive difference, NOT a p-value or a truth score')
            for field in ('delta_saved_logp', 'delta_saved_logodds_rest',
                          'delta_logodds_1_over_0', 'delta_correct_candidate_margin'):
                row[field + '_primary'] = r[field]
                row[field + '_control'] = c[field]
                row[field + '_difference'] = r[field] - c[field] if r[field] is not None else None
            result.append(row)
    return result



def summarize(baselines, cases, details, missing, invalid, stage_status):
    # Core factorial only: exclude content/style controls. Count queries once, not per layer/head.
    core = [r for r in baselines if r['kind'] == 'constructed' and r['split'] == 'heldout'
            and cases[r['case']].get('content', 0) == 0 and not cases[r['case']].get('style', 0)]
    counts = dict(queries=len(core), groups=len({r['group'] for r in core}),
                  candidate_preference_correct=sum(r['candidate_preference_correct'] is True for r in core),
                  global_correct=sum(r['global_correct'] is True for r in core),
                  top1_outside_candidates=sum(r['top1_outside_candidates'] for r in core),
                  wrong_candidate_top1=sum(not r['top1_outside_candidates'] and r['global_correct'] is False for r in core),
                  candidate_ties=sum(r['candidate_tie'] for r in core))
    frequencies = Counter((r['global_top1'], r['global_top1_text']) for r in core if r['top1_outside_candidates'])
    outside = [dict(token_id=t, text=s, count=n) for (t, s), n in frequencies.most_common()]
    return dict(version=VERSION, heldout_core=counts, outside_top_tokens=outside,
                baseline_site_rows=len(baselines), intervention_site_rows=len(details),
                missing=missing, invalid=invalid, stages=stage_status,
                complete=not missing and not invalid and all(stage_status.values()),
                new_model_forwards=0, trained_parameters=0,
                interpretation=['Candidate preference and full-vocabulary selection are separate.',
                    'Outside-candidate top1 is not automatically a hallucination or failed full answer.',
                    'Natural source anchors are not certified correct/incorrect answers.',
                    'Later scored sites are effects of the original intervention, not new intervention sites.',
                    'These are selected-case diagnostics, not AUROC or a new hallucination detector.'])


def export(audit, decode=False, *, model=None, top_k=8, prefix_atol=1e-5):
    root = Path(audit).resolve()
    if top_k < 1 or not np.isfinite(prefix_atol) or prefix_atol < 0:
        raise ValueError('top-k must be positive and prefix-atol finite/nonnegative')
    if not (root / 'cases.json').is_file():
        raise FileNotFoundError(f'{root}/cases.json missing; use the routing_audit output, not scan/S10')
    config = read_json(root / 'config.json')
    if config.get('version') != 'qk-binding-audit-v1':
        raise ValueError('unsupported producer schema; old natural/scan outputs are different experiments')
    rows = read_json(root / 'cases.json')
    cases = {c['id']: c for c in rows}
    if not rows or len(cases) != len(rows) or any(not ID.fullmatch(cid) for cid in cases):
        raise ValueError('case IDs must be unique safe filenames')
    effects, index_source = effect_index(root)
    decode_status = 'not_requested'; decode_error = None; tokenizer = None
    if decode:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model or config['model'], local_files_only=True)
            decode_status = 'local_tokenizer'
        except (ImportError, OSError, KeyError) as error:
            decode_status = 'unavailable'; decode_error = str(error)
            print(f'Tokenizer unavailable; exporting numeric IDs instead: {error}', flush=True)
    @lru_cache(maxsize=None)
    def text(token):
        return tokenizer.decode([int(token)], clean_up_tokenization_spaces=False) if tokenizer else None

    missing, invalid, baseline_rows, details, world_rows, top_rows = [], [], [], [], [], []
    # Small LRU rather than retaining every full vocabulary array in RAM.
    @lru_cache(maxsize=4)
    def baseline(cid):
        return read_trace(root / 'capture' / (cid + '.npz'), cases[cid])
    valid_cases = set()
    for i, case in enumerate(rows):
        cid = case['id']
        try:
            trace = baseline(cid)
            for site in range(len(trace['queries'])):
                row = site_stats(trace, case, site, text); baseline_rows.append(row)
                top_rows.append(dict(record_type='baseline', case=cid, site=row['site'],
                                     top=top_tokens(trace, site, top_k, text)))
            valid_cases.add(cid)
        except FileNotFoundError:
            missing.append(dict(kind='baseline', case=cid, path=f'capture/{cid}.npz'))
        except (ValueError, KeyError, OSError) as error:
            invalid.append(dict(kind='baseline', case=cid, error=str(error)))
        if (i+1) % 25 == 0 or i+1 == len(rows):
            print(f'Baseline readouts {i+1}/{len(rows)}', flush=True)

    seen = set()
    for i, effect in enumerate(effects):
        cid, arm = effect.get('case', ''), effect.get('arm', '')
        if cid not in cases or not ID.fullmatch(arm):
            invalid.append(dict(kind='index', case=cid, arm=arm, error='unknown case or unsafe arm')); continue
        case = cases[cid]
        if cid not in valid_cases: continue
        try:
            si = case['site_names'].index(effect['site']); query = int(case['queries'][si])
            layer, head = int(effect['layer']), int(effect['head'])
            if min(layer, head) < 0 or int(effect['query']) != query:
                raise ValueError('effect index query / channel is invalid')
            name = f'{cid}_s{si}_l{layer}h{head}_{arm}.npz'
            if name in seen: raise ValueError('duplicate intervention index')
            seen.add(name)
            base = baseline(cid)
            changed = read_trace(root / 'interventions' / name, case,
                                 baseline=base, intervention_query=query)
            before_mask = base['queries'] < query
            prefix_error = float(np.max(np.abs(changed['logits'][before_mask] - base['logits'][before_mask]))) if before_mask.any() else 0.
            if prefix_error > prefix_atol:
                raise ValueError(f'earlier logits changed before intervention: {prefix_error}')
            if arm == 'same_world' and np.max(np.abs(changed['logits'] - base['logits'])) > prefix_atol:
                raise ValueError('same-world transfer is not a no-op')
            target = effect.get('intervention_target')
            target = None if target in (None, '', 'None', 'null') else int(target)
            donor = reference_donor(case, arm, cases)
            arm_rows, arm_top_rows = [], []
            for j, q in enumerate(base['queries']):
                before, after = (site_stats(t, case, j, text) for t in (base, changed))
                row = dict(case=cid, group=case.get('group', ''), kind=case['kind'],
                           arm=arm, layer=layer, head=head, intervention_site=effect['site'],
                           scored_site=case['site_names'][j], query=int(q), intervention_query=query,
                           temporal_scope='before' if q < query else 'at' if q == query else 'after',
                           prefix_max_error=prefix_error,
                           candidate_interpretation=before['candidate_interpretation'],
                           local_claim_supported=case.get('local_claim_supported'))
                fields = ('saved_id', 'saved_text', 'saved_probability', 'saved_logp', 'saved_logodds_rest',
                          'global_top1', 'global_top1_text', 'global_top1_probability', 'top1_outside_candidates',
                          'candidate_total', 'logodds_1_over_0', 'candidate_choice', 'candidate_preference_correct',
                          'correct_candidate_margin', 'global_correct', 'entropy_bits')
                for field in fields:
                    row[field+'_before'], row[field+'_after'] = before[field], after[field]
                for v in range(2):
                    for field in ('id', 'text', 'probability', 'rank'):
                        key = f'candidate{v}_{field}'
                        row[key+'_before'], row[key+'_after'] = before[key], after[key]
                for key in ('saved_logp', 'saved_logodds_rest', 'logodds_1_over_0'):
                    row['delta_'+key] = after[key] - before[key]
                row['delta_correct_candidate_margin'] = (after['correct_candidate_margin'] - before['correct_candidate_margin']
                                                          if before['correct_candidate_margin'] is not None else None)
                lp, b = changed['lp'][j], base['lp'][j]
                mix = np.logaddexp(lp, b) - np.log(2.)
                row['js_bits'] = float(max(0., .5*(np.dot(np.exp(lp), lp-mix) + np.dot(np.exp(b), b-mix))/np.log(2)))
                row['intervention_target'] = target if j == si else None
                # A donor V target can be outside BOTH receiver candidates.
                if target is not None and j == si:
                    for phase, trace in (('before', base), ('after', changed)):
                        stats = token_stats(trace['logits'][j], trace['lp'][j], target, text)
                        row.update({f'target_{key}_{phase}': value for key, value in stats.items()})
                        row['target_hit_'+phase] = int(trace['logits'][j].argmax()) == target
                    row['delta_target_logp'] = row['target_logp_after'] - row['target_logp_before']
                    row.update(donor)
                    row['target_distinguishes_pointer_from_donor_content'] = (target != donor['donor_declared_answer']
                                                                             if 'donor_declared_answer' in donor else None)
                arm_rows.append(row)
                arm_top_rows.append(dict(record_type='intervention', case=cid, arm=arm, layer=layer, head=head,
                    intervention_site=effect['site'], scored_site=case['site_names'][j], temporal_scope=row['temporal_scope'],
                    top_before=top_tokens(base, j, top_k, text), top_after=top_tokens(changed, j, top_k, text)))
            details.extend(arm_rows)
            top_rows.extend(arm_top_rows)
        except FileNotFoundError:
            missing.append(dict(kind='intervention', case=cid, arm=arm, path=f'interventions/{name}'))
        except (ValueError, KeyError, OSError, IndexError) as error:
            invalid.append(dict(kind='intervention', case=cid, arm=arm, error=str(error)))
        if (i+1) % 25 == 0 or i+1 == len(effects):
            print(f'Intervention readouts {i+1}/{len(effects)}', flush=True)

    # Natural whole-world effects are separate from single-head interventions.
    natural = {c['group']: c for c in rows if c['kind'] == 'natural' and c.get('world') == 'original'}
    lookup = {(r['case'], r['site']): r for r in baseline_rows}
    for case in rows:
        if case['kind'] != 'natural' or case.get('world') == 'original' or case['group'] not in natural:
            continue
        original = natural[case['group']]
        if case['id'] not in valid_cases or original['id'] not in valid_cases: continue
        # Prompt positions MAY differ; answer-site roles and actual scored token must not.
        if (case['site_names'] != original['site_names'] or case['expected_tokens'] != original['expected_tokens'] or
                case['ids'][case['prompt_length']:] != original['ids'][original['prompt_length']:]):
            invalid.append(dict(kind='world_alignment', case=case['id'], error='answer suffix / sites / targets differ')); continue
        for site in case['site_names']:
            a, b = lookup[(original['id'], site)], lookup[(case['id'], site)]
            world_rows.append(dict(case=case['group'], world=case['world'], site=site,
                saved_id=a['saved_id'], saved_text=a['saved_text'],
                original_probability=a['saved_probability'], changed_probability=b['saved_probability'],
                delta_saved_logp=b['saved_logp']-a['saved_logp'],
                delta_saved_logodds_rest=b['saved_logodds_rest']-a['saved_logodds_rest'],
                original_claim_supported=original.get('local_claim_supported'),
                changed_claim_supported=case.get('local_claim_supported'),
                interpretation='changed evidence world, NOT repair under original facts'))

    stage_reports = {key: read_json(root/key) if (root/key).exists() else {}
                     for key in ('capture_complete.json', 'interventions_complete.json')}
    stage_status = {key: bool(value.get('complete')) for key, value in stage_reports.items()}
    for marker, field, expected in (('capture_complete.json', 'cases', len(cases)),
                                    ('interventions_complete.json', 'rows', len(effects))):
        value = stage_reports[marker].get(field)
        if value is not None and value != expected:
            invalid.append(dict(kind='completion_index', path=marker, error=f'{field}={value}, expected {expected}'))
    status = summarize(baseline_rows, cases, details, missing, invalid, stage_status)
    status.update(expected_baselines=len(cases), expected_interventions=len(effects), index_source=index_source,
                  tokenizer_status=decode_status, tokenizer_error=decode_error, prefix_atol=prefix_atol,
                  producer_config=config, source_inputs_unchanged=True)
    out = root / 'semantic_readout_review'; out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.semantic-readout-', dir=root) as temporary:
        stage = Path(temporary)
        write_csv(stage/'baseline_readout.csv', baseline_rows)
        write_csv(stage/'candidate_details.csv', details)
        write_csv(stage/'natural_world_effects.csv', world_rows)
        write_csv(stage/'control_comparisons.csv', control_comparisons(details))
        with (stage/'top_tokens.jsonl').open('w', encoding='utf-8') as stream:
            for row in top_rows: stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        write_json(stage/'export_status.json', status)
        write_json(stage/'cases.json', rows)
        for filename in ('hypothesis_counts.json', 'measurement.json', 'subspaces.json',
                         'selected_channel_binding.csv', 'intervention_effects.csv', 'qk_decomposition.csv'):
            source = root / filename
            if source.is_file(): (stage / filename).write_bytes(source.read_bytes())
        core = status['heldout_core']
        text_report = f'''# Frozen routing readout\n\nComplete: {status['complete']}\n\nHeldout core: {core['queries']} queries from {core['groups']} groups.\nCandidate preference correct: {core['candidate_preference_correct']}; full-vocabulary correct: {core['global_correct']}; outside-candidate top1: {core['top1_outside_candidates']}; wrong candidate top1: {core['wrong_candidate_top1']}.\n\nTokenizer: {decode_status}. Missing: {len(missing)}; invalid: {len(invalid)}.\n\nRead baseline_readout.csv before judging interventions. candidate_details.csv separates before/at/after effects and reports candidate mass, odds, target ranks and saved-token support. No new model forwards or labels. Natural anchors are not correct answers; candidate-only accuracy is not model answer accuracy. Next-token outputs cannot establish correctness of a longer completion.\n\nThe v1 Q donor preserves the same payload words. A target equal to the donor answer does not distinguish pointer transfer from content transfer. No new donor experiment is invented by this export.\n'''
        (stage/'README.md').write_text(text_report, encoding='utf-8')
        paths = sorted(stage.iterdir())
        archive = root / 'semantic_readout_review.zip'
        temp_zip = stage / 'review.zip'
        with ZipFile(temp_zip, 'w', ZIP_DEFLATED) as bundle:
            for path in paths: bundle.write(path, f'semantic_readout_review/{path.name}')
        for path in paths: path.replace(out / path.name)
        temp_zip.replace(archive)
    print(json.dumps(dict(complete=status['complete'], heldout_core=core, missing=len(missing),
                         invalid=len(invalid), archive=str(archive)), ensure_ascii=False), flush=True)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', default='outputs/lookback_routing_v1')
    parser.add_argument('--decode', action='store_true', help='load ONLY the existing local tokenizer')
    parser.add_argument('--model', help='override local tokenizer directory, not model weights')
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--prefix-atol', type=float, default=1e-5)
    parser.add_argument('--allow-partial', action='store_true', help='return success on missing files, never invalid data')
    args = parser.parse_args(argv)
    try:
        status = export(args.audit, args.decode, model=args.model, top_k=args.top_k, prefix_atol=args.prefix_atol)
    except (FileNotFoundError, ValueError, KeyError, OSError) as error:
        parser.exit(2, f'Readout error: {error}\n')
    return 0 if not status['invalid'] and (status['complete'] or args.allow_partial) else 2


if __name__ == '__main__':
    raise SystemExit(main())
