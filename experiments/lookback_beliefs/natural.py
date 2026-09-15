"""Analyze the saved cooking/clothing cases, then test finite message mediation.

No training or synthetic-answer substitution. All cases are selected post hoc;
results are mechanism diagnostics, NOT unsupervised detection performance.
"""
import argparse
import csv
import gc
import html
import json
import sys
from pathlib import Path

import numpy as np

from .natural_data import (DEFAULT_SAMPLES, cache_measurements, jsonl,
                           load_trace, prepare_case, resolve_samples)


# Output schema is unchanged; config.execution distinguishes numerical protocols.
VERSION = 'natural-lookback-audit-v1'


def write_json(path, value):
    def safe(x):
        if isinstance(x, dict): return {str(k): safe(v) for k,v in x.items()}
        if isinstance(x, (list, tuple)): return [safe(v) for v in x]
        if isinstance(x, np.ndarray): return safe(x.tolist())
        if isinstance(x, np.generic): return safe(x.item())
        return None if isinstance(x, float) and not np.isfinite(x) else x
    Path(path).write_text(json.dumps(safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def save_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def local_report(directory, detail, effects):
    source = detail['case']
    rows = ''.join('<tr>'+''.join('<td>'+html.escape(str(row.get(k,'')))+'</td>'
                                for k in ('arm','layer','region','mean_delta_saved_logp','mean_js_bits'))+'</tr>'
                   for row in effects)
    response = html.escape(detail['response']).replace(html.escape(source['target']),
                '<mark>'+html.escape(source['target'])+'</mark>', 1)
    body = (f'<h1>{html.escape(source["id"])}</h1><p>{html.escape(source["status"])}</p>'
            f'<p>{html.escape(source["review"])}</p><h2>原回答</h2>'
            f'<pre style="white-space:pre-wrap">{response}</pre><h2>原材料中的限定条件</h2>'
            f'<pre>{html.escape(chr(10).join(source["constraint"]))}</pre>'
            '<h2>有限干预</h2><p>Δ log p 为干预减基线；正值表示干预后更支持原词，'
            '不是正确概率。恢复的是原始状态，不是注入正确答案。层段来自先前探索，未作独立验证。</p>'
            '<table border="1" cellpadding="6"><tr><th>arm</th><th>layer</th><th>region</th>'
            '<th>mean Δ log p</th><th>mean JS bits</th></tr>'+rows+'</table>')
    (directory/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Natural case audit</title>'+body,
                                        encoding='utf8')


def inspect_case(item, trace, directory):
    measured = cache_measurements(item, trace)
    np.savez_compressed(directory/'cached_observations.npz', **measured)
    rows = []
    for i,t in enumerate(item['steps']):
        rows.append(dict(step=int(t), query=int(item['queries'][i]), token=item['token_text'][i],
                         in_target=bool(t in item['target_steps']),
                         entropy_bits=float(measured['entropy_bits'][i])))
    save_csv(directory/'tokens.csv', rows)
    role_rows = []
    for name in item['groups']:
        values = measured['mass_'+name]
        for region, mask in (('target', np.isin(item['steps'],item['target_steps'])),
                             ('before_target', item['steps']<min(item['target_steps'])),
                             ('after_target', item['steps']>max(item['target_steps']))):
            if not mask.any(): continue
            for l in range(values.shape[0]):
                for h in range(values.shape[1]):
                    role_rows.append(dict(region=region, layer=l, head=h, group=name,
                                          mass=float(values[l,h,mask].mean())))
    save_csv(directory/'head_reads.csv', role_rows)
    detail = dict(version=VERSION, case=item['case'], record=item['record'],
                  response=item['original_response'], original_prompt=item['prompt_text'],
                  coordinates=dict(prompt_length=item['prompt_length'], steps=item['steps'],
                                   target_steps=item['target_steps'], carrier_absolute=item['carrier'],
                                   control_carrier_absolute=item['control_carrier'], groups=item['groups']),
                  scope='RAGTruth-source regenerated sample, manual case diagnosis; no borrowed gold labels',
                  cached_hidden_convention='hidden[0]=embedding; hidden[l+1]=block l for l<L-1; hidden[L]=final normalized',
                  geometry='cosine diagnostic, NOT a pointer decoder or causal contribution')
    write_json(directory/'case.json', detail)
    local_report(directory,detail,[])
    return detail


def mechanism_case(model, tokenizer, item, trace, directory, args):
    import torch
    from .natural_engine import cache_carrier, forward, score
    from .replay_checks import compare_replay
    layers = args.layers
    print(f'  baseline: {item["execution"]}; exact saved token IDs', flush=True)
    check_item = dict(item)
    if item['execution'] == 'incremental_kv':
        check_item['cached_attention'] = trace['attention']
    base = forward(model, check_item, layers)
    z0 = base['logits']
    steps, p = item['steps'], item['prompt_length']
    originals = np.asarray(item['original_ids'])[p+steps]
    top = z0.topk(2, -1).indices.numpy()
    alternatives = np.where(top[:,0] == originals, top[:,1], top[:,0])
    diagnostics, token_checks = compare_replay(base, trace, item)
    checks = dict(**diagnostics, replay_atol=args.replay_atol,
                  replay_attention_atol=args.replay_attention_atol, layers=layers,
                  no_answer_injection=True, original_tokens_preserved=True,
                  history_is_teacher_forced=True, torch_version=torch.__version__,
                  cuda_version=torch.version.cuda,
                  transformers_version=getattr(sys.modules.get('transformers'), '__version__', None),
                  model_dtype=str(next(model.parameters()).dtype),
                  attention_implementation=getattr(model.config, '_attn_implementation', None))
    save_csv(directory/'replay_tokens.csv', token_checks)
    write_json(directory/'checks.json', checks)
    if (not diagnostics['finite'] or max(diagnostics['saved_top_logit_error'],
            diagnostics['saved_log_normalizer_error']) > args.replay_atol
            or (diagnostics['saved_attention_error'] is not None
                and diagnostics['saved_attention_error'] > args.replay_attention_atol)):
        raise ValueError(f'{item["case"]["id"]}: {item["execution"]} replay failed original-cache checks; '
                         'see checks.json and replay_tokens.csv. No intervention executed. '
                         'Verify weights/dtype/PyTorch/Transformers; do not raise tolerances to bypass this check.')
    scores, effects = {}, []
    def save_arm(name, result, layer=None):
        measured = score(result['logits'], originals, alternatives, z0)
        scores[name] = measured
        np.savez_compressed(directory/(name+'.npz'), steps=steps, queries=item['queries'],
                            saved_token_ids=originals, **measured)
        if result.get('removed_codes'):
            codes = {}
            for l, entries in result['removed_codes'].items():
                codes[f'queries_l{l}'] = np.asarray([e[0] for e in entries])
                codes[f'pre_WO_codes_l{l}'] = np.stack([e[1] for e in entries]).astype(np.float16)
            np.savez_compressed(directory/(name+'.messages.npz'), **codes)
        for region, mask in (('target', np.isin(steps,item['target_steps'])),
                             ('after_target',steps>max(item['target_steps']))):
            if not mask.any(): continue
            effects.append(dict(arm=name, layer=layer, region=region, tokens=int(mask.sum()),
                                mean_delta_saved_logp=float(measured['delta_saved_logp'][mask].mean()),
                                mean_js_bits=float(measured['js_to_baseline_bits'][mask].mean()),
                                attention_reconstruction_error=result['replay_error']))
        save_csv(directory/'effects.csv', effects)
    save_arm('baseline', base)
    g = item['groups']; qstart = item['source_query_start']
    all_source = np.union1d(g['constraint'], g['payload'])
    sham = forward(model,item,layers,cut_keys=all_source,cut_start=qstart,strength=0.)
    checks['zero_cut_max_logit_error'] = float((sham['logits']-z0).abs().max())
    if checks['zero_cut_max_logit_error'] > 1e-5:
        raise ValueError('zero-strength message intervention changed the output')
    del sham
    cut_constraint = None
    for name, keys in (('constraint',g['constraint']), ('payload',g['payload']),
                       ('control_constraint',g['control_constraint']), ('control_payload',g['control_payload']),
                       ('constraint_and_payload',all_source),
                       ('history_seed',g['history_seed']), ('history_control',g['history_control'])):
        print(f'  cut {name}', flush=True)
        start = item['history_query_start'] if name.startswith('history') else qstart
        result = forward(model,item,layers,cut_keys=keys,cut_start=start)
        save_arm('cut_'+name,result)
        if name == 'constraint':
            cut_constraint = dict(states=result['states'], control_states=result['control_states'])
        if name.startswith('history'):
            pre = item['queries'] < start
            checks[name+'_earlier_logit_error'] = float((result['logits'][pre]-z0[pre]).abs().max()) if pre.any() else 0.
            if checks[name+'_earlier_logit_error'] > 1e-5:
                raise ValueError('history intervention changed a prediction before the history was available')
        del result
    checks['carrier_cache'] = {}
    for layer in layers:
        print(f'  carrier restoration layer {layer}', flush=True)
        live_sham = forward(model,item,layers,patch=(layer,base['states'][layer]))
        live_error = float((live_sham['logits']-z0).abs().max())
        if live_error > 1e-5: raise ValueError('same-world live carrier patch is not a no-op')
        saved = cache_carrier(trace,layer,item['carrier'],len(model.model.layers))
        cached_sham = forward(model,item,layers,patch=(layer,saved))
        cache_error = float((cached_sham['logits']-z0).abs().max())
        use_cache = cache_error <= args.cache_atol
        state = saved if use_cache else base['states'][layer]
        checks['carrier_cache'][layer] = dict(max_logit_error=cache_error, reused=use_cache,
                                              fallback='none' if use_cache else 'same-world live state',
                                              live_sham_error=live_error)
        del live_sham,cached_sham
        rescued = forward(model,item,layers,cut_keys=g['constraint'],cut_start=qstart,patch=(layer,state))
        save_arm(f'restore_carrier_l{layer}',rescued,layer)
        reverse = forward(model,item,layers,patch=(layer,cut_constraint['states'][layer]))
        save_arm(f'cut_carrier_into_base_l{layer}',reverse,layer)
        control = forward(model,item,layers,cut_keys=g['constraint'],cut_start=qstart,
                          patch=(layer,base['control_states'][layer],item['control_carrier']))
        save_arm(f'restore_control_site_l{layer}',control,layer)
        checks['carrier_cache'][layer]['cut_control_state_l2_change'] = float(
            (base['control_states'][layer]-cut_constraint['control_states'][layer]).float().norm())
        del rescued,reverse,control
    b = scores['baseline']['saved_logp']
    c,v,cv = (scores['cut_'+k]['saved_logp'] for k in ('constraint','payload','constraint_and_payload'))
    interaction = b-c-v+cv
    np.savez_compressed(directory/'interaction.npz', steps=steps, finite_nonadditivity_nats=interaction)
    floor = max(checks['zero_cut_max_logit_error'], *(x['max_logit_error'] if x['reused'] else x['live_sham_error']
                                                     for x in checks['carrier_cache'].values()))*2
    checks['diagnostic_numeric_floor_logp'] = floor
    checks['intervention_complete'] = True
    comparisons = []
    focus = np.isin(steps,item['target_steps'])
    cdelta = scores['cut_constraint']['saved_logp']-b
    vdelta = scores['cut_payload']['saved_logp']-b
    for layer in layers:
        restored = scores[f'restore_carrier_l{layer}']['saved_logp']
        cut = scores['cut_constraint']['saved_logp']
        denom = float((b-cut)[focus].mean())
        recovery = float((restored-cut)[focus].mean())
        comparisons.append(dict(layer=layer, constraint_cut_mean_delta=float(cdelta[focus].mean()),
                                payload_cut_mean_delta=float(vdelta[focus].mean()),
                                carrier_recovery_delta=recovery,
                                control_site_recovery_delta=float((scores[f'restore_control_site_l{layer}']['saved_logp']-cut)[focus].mean()),
                                recovery_fraction=recovery/denom if abs(denom)>max(floor,1e-3) else None,
                                interpretation='signed recovery of original preference, NOT recovery of truth'))
    save_csv(directory/'mediation.csv',comparisons)
    write_json(directory/'checks.json',checks)
    write_json(directory/'mechanism_summary.json',dict(
        case_status=item['case']['status'], comparisons=comparisons,
        constraint_minus_control_delta=float((cdelta-scores['cut_control_constraint']['delta_saved_logp'])[focus].mean()),
        payload_minus_control_delta=float((vdelta-scores['cut_control_payload']['delta_saved_logp'])[focus].mean()),
        mean_finite_nonadditivity_nats=float(interaction[focus].mean()),
        scope='finite direct-message and carrier mediation in a fixed prefix; not pure QK/address identification',
        interpretation='constraint-cut positive delta can indicate opposition to the ORIGINAL output; '
                       'payload-cut negative delta indicates support. Neither alone proves a binding mechanism.',
        no_truth_margin=True, both_conditions_must_be_compared_to_controls=True))
    return effects


def run(args):
    if (args.context < 0 or not args.layers or len(set(args.layers)) != len(args.layers)
            or any(l < 0 for l in args.layers)
            or not np.isfinite([args.replay_atol,args.cache_atol,args.replay_attention_atol]).all()
            or min(args.replay_atol,args.cache_atol,args.replay_attention_atol) < 0):
        raise ValueError('invalid context, layers or numeric tolerance')
    cases = json.loads(Path(args.cases).read_text(encoding='utf8'))
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('duplicate case IDs')
    if args.select and not set(args.select) <= {c['id'] for c in cases}:
        raise ValueError('unknown selected case')
    if args.select: cases = [c for c in cases if c['id'] in args.select]
    if not cases: raise ValueError('no selected case')
    samples = resolve_samples(args.samples,cases)
    states = Path(args.states) if args.states else samples.parent/('states_'+samples.name)
    if not states.is_dir(): raise FileNotFoundError(f'original fixed-prefix states missing: {states}')
    settings = json.loads((samples/'settings.json').read_text())
    state_settings = json.loads((states/'settings.json').read_text())
    if (Path(settings['model']).resolve()!=Path(state_settings['model']).resolve()
            or settings['dtype']!=state_settings['dtype']):
        raise ValueError('sample and state cache model/dtype disagree')
    selected = [r for r in jsonl(samples/'samples.jsonl') if (str(r['source_id']),r['seed']) in
                {(c['source_id'],c['seed']) for c in cases}]
    files = [samples/'samples.jsonl',samples/'settings.json',states/'settings.json']
    files += [folder/r['trace'] for r in selected for folder in (samples,states)]
    stamps = [[str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns] for p in files]
    config = dict(version=VERSION, input_stamps=stamps, device=args.device, samples=str(samples.resolve()), states=str(states.resolve()),
                  model=settings['model'], dtype=settings['dtype'], cases=cases, layers=args.layers,
                  context=args.context, replay_atol=args.replay_atol, cache_atol=args.cache_atol,
                  execution=args.execution, replay_attention_atol=args.replay_attention_atol,
                  inputs='regenerated answers from RAGTruth source passages; manually selected diagnosis',
                  trained_parameters=0, use_gold_hallucination_labels=False)
    output = Path(args.output)
    if output.exists():
        if not args.resume or not (output/'config.json').is_file() or json.loads((output/'config.json').read_text())!=config:
            raise ValueError('use a new output directory or --resume with the same case/configuration; v1 full and v2 KV results cannot be mixed')
    else:
        output.mkdir(parents=True);write_json(output/'config.json',config)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(settings['model'],local_files_only=True)
    model = None
    reports = []
    try:
        for case in cases:
            directory=output/case['id']; directory.mkdir(exist_ok=True)
            marker=directory/('mechanism_complete.json' if args.phase in ('all','mechanism') else 'inspect_complete.json')
            if marker.exists() and args.resume:
                reports.append(dict(case=case['id'],resumed=True));continue
            print(f'{case["id"]}: source={case["source_id"]}, seed={case["seed"]}',flush=True)
            row,trace=load_trace(samples,states,case['source_id'],case['seed'])
            item=prepare_case(case,row,trace,tokenizer,args.context)
            item['execution'] = args.execution
            detail=inspect_case(item,trace,directory)
            write_json(directory/'inspect_complete.json',dict(complete=True,new_model_forwards=0))
            if args.phase in ('all','mechanism'):
                if model is None:
                    import torch
                    from transformers import AutoModelForCausalLM
                    model=AutoModelForCausalLM.from_pretrained(settings['model'],local_files_only=True,
                        torch_dtype=getattr(torch,settings['dtype']),attn_implementation='eager').to(args.device).eval().requires_grad_(False)
                effects=mechanism_case(model,tokenizer,item,trace,directory,args)
                local_report(directory,detail,effects)
                write_json(marker,dict(complete=True,trained_parameters=0,method='finite message cuts + carrier swaps', execution=args.execution))
            reports.append(dict(case=case['id'],trace=row['trace'],status=case['status']))
            del trace,item;gc.collect()
        write_json(output/'summary.json',dict(cases=reports,phase=args.phase,independent_sources=len({c['source_id'] for c in cases}),
                                              scope='selected-case diagnosis, no AUROC/AP or generalization claims'))
        links=''.join(f'<li><a href="{html.escape(c["id"])}/report.html">{html.escape(c["id"])}</a></li>' for c in cases)
        (output/'index.html').write_text('<meta charset="utf-8"><h1>Cooking / clothing case audit</h1><ul>'+links+'</ul>',encoding='utf8')
        print(f'Case reports: {output / "index.html"}',flush=True)
    finally:
        del model;gc.collect()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',default=DEFAULT_SAMPLES)
    p.add_argument('--states',default=None,help='default: outputs/states_<sample directory name>')
    p.add_argument('--cases',default=str(Path(__file__).with_name('natural_cases.json')))
    p.add_argument('--select',nargs='+')
    p.add_argument('--output',default='outputs/lookback_beliefs_natural_v2')
    p.add_argument('--phase',choices=('inspect','mechanism','all'),default='all')
    p.add_argument('--layers',type=int,nargs='+',default=[15,19,22,26],help='predeclared exploratory layers, not universally valid pointer layers')
    p.add_argument('--execution',choices=('incremental_kv','full'),default='incremental_kv',help='match original generation schedule; full is legacy diagnostic only')
    p.add_argument('--context',type=int,default=8)
    p.add_argument('--replay-atol',type=float,default=.1,help='unchanged absolute tolerance for original top logits/log-normalizer')
    p.add_argument('--replay-attention-atol',type=float,default=1e-4,help='old float16 attention vs identically quantized KV replay')
    p.add_argument('--cache-atol',type=float,default=.02,help='max same-world patch logit error to reuse quantized carrier')
    p.add_argument('--device',default='cuda:0');p.add_argument('--resume',action='store_true')
    return p


def main():
    run(parser().parse_args())


if __name__=='__main__':main()
