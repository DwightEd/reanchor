"""Identifiable selector/address/usage audit. No hallucination classifier.

Run: python -m experiments.lookback_beliefs.routing_audit --resume
Default: reference-frozen relation controls AND the four saved natural windows.
"""
import argparse
import csv
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit,logsumexp

from . import routing_data as data
from .routing_math import (channel_information,categorical,fit_constraint_space,fit_probe,
                           probe_logits,accessible_bits,cluster_interval,decompose_qk,receiver_coordinates,address_components)

VERSION='qk-binding-audit-v1'


def write(path,value):
    def safe(x):
        if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [safe(v) for v in x]
        if isinstance(x,np.ndarray):return safe(x.tolist())
        if isinstance(x,np.generic):return safe(x.item())
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(safe(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');tmp.replace(path)


def table(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def load_trace(root,case):
    with np.load(Path(root)/'capture'/(case['id']+'.npz'),allow_pickle=False) as f:
        if str(f['case_signature'])!=signature(case):raise ValueError('cached case changed')
        return {k:f[k].copy() for k in f.files if k!='case_signature'}


def save_trace(root,case,trace):
    file=Path(root)/'capture'/(case['id']+'.npz');file.parent.mkdir(exist_ok=True)
    partial=file.with_suffix('.partial.npz')
    np.savez_compressed(partial,case_signature=signature(case),**trace);partial.replace(file)


def check_original(trace,expected,tolerance=.1):
    z=trace['logits'];a=trace['attention'];ids=expected['top_ids']
    scores=np.take_along_axis(z,ids,axis=1)
    errors=np.max(np.abs(scores-expected['top_logits']),axis=1)
    lse=np.abs(logsumexp(z.astype(float),axis=1)-expected['log_normalizer'])
    # Original attention was serialized in float16.
    ae=np.max(np.abs(a.astype(np.float16).astype(float)-expected['attention'].astype(float)),axis=(1,2,3))
    report=dict(max_top_logit_error=float(errors.max()),max_log_normalizer_error=float(lse.max()),
                max_attention_error=float(ae.max()),per_query_logit_error=errors,
                per_query_log_normalizer_error=lse,per_query_attention_error=ae,
                tolerance=tolerance,valid=bool(max(errors.max(),lse.max())<=tolerance and ae.max()<=1e-4))
    return report


def route_channel(trace,layer,head,site=0):
    row=trace['attention'][site,layer,head].astype(float)
    probs=row[trace['anchors']]
    # Native BF16 row sums may not be exactly one: normalize the WHOLE row only.
    total=row.sum()
    if total<=0:raise ValueError('empty attention row')
    probs=probs/total
    return np.r_[probs,max(0.,1-probs.sum())]


def grouped(cases):
    groups={}
    for c in cases:
        if c['kind']=='constructed' and c['content']==0 and not c.get('style',0):
            groups.setdefault(c['group'],{})[(c['c'],c['b'],c['o'])]=c
    for name,g in groups.items():
        if len(g)!=8:raise ValueError('incomplete intervention factorial: '+name)
    return groups


def fit_and_measure(root,cases,args):
    """Reference fitting is isolated; natural cases are NEVER probe training data."""
    groups=grouped(cases)
    ref=[g for g in groups.values() if next(iter(g.values()))['split']=='reference']
    held=[g for g in groups.values() if next(iter(g.values()))['split']=='heldout']
    if not ref or not held:raise ValueError('separate reference/heldout relation groups required')
    reference=[c for g in ref for c in g.values()];test=[c for g in held for c in g.values()]
    # Only compact query/source states live in RAM; full attention is loaded per group.
    compact={}
    for c in reference+test:
        t=load_trace(root,c)
        compact[c['id']]={k:t[k] for k in ('h','x','q_raw','k_raw','key_x','logits')}
        a=t['attention'][0].astype(float); total=a.sum(-1,keepdims=True)
        probs=a[:,:,t['anchors']]/total
        compact[c['id']]['route']=np.concatenate((probs,np.maximum(0,1-probs.sum(-1,keepdims=True))),-1)
    first=compact[reference[0]['id']];L=first['x'].shape[1];H=first['q_raw'].shape[2]
    models={};measure=[];ranking=[];states=[]
    for l in range(L):
        selector=[];nuisance=[];address=[];key_nuisance=[]
        for g in ref:
            get=lambda c,b,o:compact[g[(c,b,o)]['id']]
            for b in (0,1):
                for o in (0,1):selector.append(get(1^b,b,o)['x'][0,l]-get(b,b,o)['x'][0,l])
            for c in (0,1):
                for o in (0,1):nuisance.append(get(c^1,1,o)['x'][0,l]-get(c,0,o)['x'][0,l])
                for b in (0,1):nuisance.append(get(c,b,1)['x'][0,l]-get(c,b,0)['x'][0,l])
            for o in (0,1):address.extend(get(0,1,o)['key_x'][l]-get(0,0,o)['key_x'][l])
            for b in (0,1):key_nuisance.extend(get(0,b,1)['key_x'][l]-get(0,b,0)['key_x'][l])
        qs=fit_constraint_space(selector,nuisance,args.rank,args.nuisance_rank)
        ks=fit_constraint_space(address,key_nuisance,args.rank,args.nuisance_rank)
        xref=np.stack([compact[c['id']]['x'][0,l] for c in reference]);yref=np.array([c['c'] for c in reference])
        xtest=np.stack([compact[c['id']]['x'][0,l] for c in test]);ytest=np.array([c['c'] for c in test])
        hp=fit_probe(xref,yref,rank=args.probe_rank)
        qpred=probe_logits(hp,xtest)
        binding_ref=np.array([c['correct_index'] for c in reference])
        binding_test=np.array([c['correct_index'] for c in test])
        bp=fit_probe(xref,binding_ref,rank=args.probe_rank)
        binding_pred=probe_logits(bp,xtest)
        kref=[];ky=[];ktest=[]
        for c in reference:
            if c['c']:continue
            kref.extend(compact[c['id']]['key_x'][l]);ky.extend([c['b'],1^c['b']])
        kp=fit_probe(kref,ky,rank=args.probe_rank)
        for c in test:ktest.append(probe_logits(kp,compact[c['id']]['key_x'][l]))
        ktest=np.asarray(ktest)
        models[l]=dict(query_space=qs['basis'],address_space=ks['basis'],query_probe=hp,binding_probe=bp,key_probe=kp)
        bytest={c['id']:i for i,c in enumerate(test)}
        for i,c in enumerate(test):
            z=compact[c['id']]['logits'][0];candidate=np.array(c['candidate_ids']);truth=c['correct_index']
            pred=int(np.argmax(z[candidate]));top=int(np.argmax(z));gating=top not in candidate
            inferred=(ktest[i]>=0)==(qpred[i]>=0)
            unique=bool(inferred.sum()==1)
            inferred_index=int(np.flatnonzero(inferred)[0]) if unique else None
            states.append(dict(case=c['id'],group=c['group'],layer=l,selector=c['c'],binding=c['b'],
                order=c['o'],global_correct=top==candidate[truth],candidate_correct=pred==truth,gating_failure=gating,
                selector_decoded_correct=bool((qpred[i]>=0)==c['c']),
                binding_decoded_correct=bool((binding_pred[i]>=0)==truth),
                native_candidate_confidence=float(np.exp(z[candidate[pred]]-logsumexp(z.astype(float)))),
                native_entropy_bits=float(-np.sum(np.exp(z-logsumexp(z))*(z-logsumexp(z)))/np.log(2)),
                decoded_address=inferred_index,decoded_address_correct=unique and inferred_index==truth,
                decoded_binding_correct_but_wrong_choice=bool((binding_pred[i]>=0)==truth and pred!=truth and not gating),
                independently_decoded_selector_address_agreement=unique and inferred_index==truth,
                selector_logit=float(qpred[i]),constraint_rank=len(qs['basis']),address_rank=len(ks['basis'])))
        # Per physical channel; do NOT select the maximum on heldout errors.
        for h in range(H):
            ref_info=[];test_info=[];output_info=[];fidelity=[];outside=[];case_details=[]
            for group_list,results in ((ref,ref_info),(held,test_info)):
                for g in group_list:
                    vals=[];outs=[]
                    for b in (0,1):
                        for o in (0,1):
                            pair=[compact[g[(c,b,o)]['id']] for c in (0,1)]
                            ch=np.stack([t['route'][l,h] for t in pair]);vals.append(channel_information(ch))
                            if group_list is held:
                                op=np.stack([categorical(pair[c]['logits'][0],g[(c,b,o)]['candidate_ids']) for c in (0,1)])
                                outs.append(channel_information(op))
                                for c in (0,1):
                                    truth=c^b;fidelity.append(ch[c,truth]);outside.append(ch[c,-1])
                                    case_details.append((g[(c,b,o)]['id'],ch[c]))
                    results.append(float(np.mean(vals)))
                    if group_list is held:output_info.append(float(np.mean(outs)))
            # q probe assesses selector decodability AFTER WQ, not answer correctness.
            hbits=accessible_bits(qpred,ytest);qbits=None
            channelrow=dict(layer=l,head=h,reference_route_information_bits=float(np.mean(ref_info)),
                heldout_route_information_bits=float(np.mean(test_info)),
                heldout_output_information_bits=float(np.mean(output_info)),
                available_selector_lower_bits=hbits,available_binding_lower_bits=accessible_bits(binding_pred,binding_test),
                available_key_phase_lower_bits=accessible_bits(ktest.ravel(),np.array([[c['b'],1^c['b']] for c in test]).ravel()),q_binding_lower_bits=qbits,
                selector_probe_accuracy=float(np.mean((qpred>=0)==ytest)),q_probe_accuracy=None,
                mean_correct_anchor_mass=float(np.mean(fidelity)),mean_outside_anchor_mass=float(np.mean(outside)),
                # This empirical difference is NOT exact information loss/certificate.
                available_lower_minus_route_bits=accessible_bits(binding_pred,binding_test)-float(np.mean(test_info)),
                query_rank=len(qs['basis']),address_rank=len(ks['basis']),
                route_group_summary=cluster_interval(test_info,args.seed))
            measure.append(channelrow);ranking.append((np.mean(ref_info),l,h))
        print(f'fit/measure layer {l+1}/{L}; selector rank={len(qs["basis"])} address rank={len(ks["basis"])}',flush=True)
    # Freeze reference-only selection BEFORE interventions or natural comparisons.
    points=[dict(layer=int(l),head=int(h)) for _,l,h in sorted(ranking,reverse=True)[:args.channels]]
    point_states=[]
    for pt in points:
        l,h=pt['layer'],pt['head']
        qp=fit_probe(np.stack([compact[c['id']]['q_raw'][0,l,h] for c in reference]),binding_ref,rank=args.probe_rank)
        qlogit=probe_logits(qp,np.stack([compact[c['id']]['q_raw'][0,l,h] for c in test]))
        kr=[];ky=[];kt=[];kty=[]
        for c in reference:
            if c['c']:continue
            kr.extend(compact[c['id']]['k_raw'][0,l,h]);ky.extend([c['b'],1^c['b']])
        projected_key_probe=fit_probe(kr,ky,rank=args.probe_rank)
        for c in test:
            kt.extend(compact[c['id']]['k_raw'][0,l,h]);kty.extend([c['b'],1^c['b']])
        klogit=probe_logits(projected_key_probe,np.asarray(kt)).reshape(len(test),2)
        models[l][f'q_probe_h{h}']=qp;models[l][f'k_probe_h{h}']=projected_key_probe
        for i,c in enumerate(test):
            row=next(x for x in states if x['case']==c['id'] and x['layer']==l)
            point_states.append({**row,'head':h,'q_binding_decoded_correct':bool((qlogit[i]>=0)==c['correct_index']),
                'k_phase_decoded_correct_count':int(np.sum((klogit[i]>=0)==np.array([c['b'],1^c['b']]))),
                'native_anchor_choice_correct':bool(np.argmax(compact[c['id']]['route'][l,h,:2])==c['correct_index']),
                'native_anchor_mass':float(compact[c['id']]['route'][l,h,:2].sum())})
        for row in measure:
            if row['layer']==l and row['head']==h:
                row['q_binding_lower_bits']=accessible_bits(qlogit,binding_test)
                row['q_probe_accuracy']=float(np.mean((qlogit>=0)==binding_test))
                row['k_phase_lower_bits']=accessible_bits(klogit.ravel(),np.asarray(kty))
                row['k_phase_probe_accuracy']=float(np.mean((klogit.ravel()>=0)==kty))
    write(Path(root)/'measurement.json',dict(version=VERSION,points=points,reference_groups=len(ref),heldout_groups=len(held),
         supervision='selector and source-role labels from constructed facts; NO natural hallucination labels',
         measures=measure,information_scope='balanced selector intervention channel; outside bucket retained',
         warning='probe lower estimates and channel MI are not semantic truth; high MI permits systematic inversion'))
    save_models(root,models)
    table(Path(root)/'selected_channel_binding.csv',point_states)
    layer_counts=[]
    for l in range(L):
        subset=[x for x in states if x['layer']==l]
        wrong=[x for x in subset if not x['candidate_correct'] and not x['gating_failure']]
        layer_counts.append(dict(layer=l,n=len(subset),binding_errors=len(wrong),
            gating_failures=sum(x['gating_failure'] for x in subset),
            correct=sum(x['global_correct'] for x in subset),
            decoded_binding_accuracy_among_binding_errors=(float(np.mean([x['binding_decoded_correct'] for x in wrong])) if wrong else None),
            decoded_correct_but_wrong_count=sum(x['decoded_binding_correct_but_wrong_choice'] for x in subset)))
    write(Path(root)/'hypothesis_counts.json',dict(layers=layer_counts,
        unit='heldout constructed query; repeated layer measurements are NOT independent examples',
        zero_errors='No evidence for encoded-correct-but-wrong when no binding errors occur; do not fabricate them',
        natural='No validated natural binding decoder is assumed; natural cases use targeted Q/K/V comparisons'))
    table(Path(root)/'channel_measures.csv',measure);table(Path(root)/'heldout_binding_readout.csv',states)
    natural_geometry(root,cases)
    return models,points


def natural_geometry(root,cases):
    """All heads/all three decision sites, not selected by a test effect.

    Label of the original CLAIM stays separate from route identity. There is no
    invented unique correct onion time, or ordinal-probe transfer claim.
    """
    lookup={c['id']:c for c in cases};rows=[]
    for c in cases:
        if c['kind']!='natural' or c['world']!='original':continue
        base=load_trace(root,c)
        for world in ('constraint_flip','paraphrase_control'):
            donor_case=lookup[c['group']+'__'+world];donor=load_trace(root,donor_case)
            for si,site in enumerate(c['site_names']):
                z0,z1=base['logits'][si].astype(float),donor['logits'][si].astype(float);y=c['expected_tokens'][si]
                lp0=z0[y]-logsumexp(z0);lp1=z1[y]-logsumexp(z1)
                for l in range(base['q'].shape[1]):
                    for h in range(base['q'].shape[2]):
                        q,k=receiver_coordinates(base,donor,si,l,h)
                        q0,k0=base['q'][si,l,h].astype(float),base['k'][si,l,h].astype(float)
                        dq=q-q0;parallel,_=address_components(dq,k0)
                        norm=float(np.linalg.norm(dq))
                        rows.append(dict(case=c['group'],site=site,layer=l,head=h,world=world,
                            original_claim_supported=c['local_claim_supported'],donor_claim_supported=donor_case['local_claim_supported'],
                            normalized_state_delta_norm=float(np.linalg.norm(donor['x'][si,l]-base['x'][si,l])),
                            query_delta_norm=norm,query_address_visible_fraction=float(np.linalg.norm(parallel)/norm) if norm else None,
                            original_saved_probability=float(np.exp(lp0)),donor_saved_probability=float(np.exp(lp1)),
                            full_world_delta_logp=float(lp1-lp0),
                            **decompose_qk(q0,q,k0,k,q.shape[-1]**-.5),
                            semantics='world intervention observation; QK decomposition is frozen-local, not a causal effect by itself'))
    table(Path(root)/'natural_pair_geometry.csv',rows)


def save_models(root,models):
    arrays={}
    for layer,model in models.items():
        for name,value in model.items():
            if isinstance(value,dict):
                for k,v in value.items():arrays[f'l{layer}__{name}__{k}']=np.asarray(v)
            else:arrays[f'l{layer}__{name}']=np.asarray(value)
    np.savez_compressed(Path(root)/'subspaces.npz',**arrays)
    write(Path(root)/'subspaces.json',dict(file='subspaces.npz',layers=list(models),
        scope='reference-only semantic decoders and contrast subspaces; not a hallucination classifier'))


def load_models(root):
    models={}
    with np.load(Path(root)/'subspaces.npz',allow_pickle=False) as f:
        for key in f.files:
            fields=key.split('__');model=models.setdefault(int(fields[0][1:]),{})
            if len(fields)==2:model[fields[1]]=f[key].copy()
            else:model.setdefault(fields[1],{})[fields[2]]=f[key].copy()
    return models


def effect_row(base,changed,case,si,layer,head,arm,expected_token=None):
    z0,z=base['logits'][si].astype(float),changed['logits'][si].astype(float)
    y=case['expected_tokens'][si]
    # Saved-token log odds against all other tokens avoids probability-ceiling floor.
    odds=lambda a:a[y]-logsumexp(np.delete(a,y))
    p0,p=route_channel(base,layer,head,si),route_channel(changed,layer,head,si)
    routeodds=lambda t:float((t['q'][si,layer,head]@(t['k'][si,layer,head,1]-t['k'][si,layer,head,0]))/
                             np.sqrt(t['q'].shape[-1]))
    return dict(case=case['id'],group=case['group'],site=case['site_names'][si],query=int(case['queries'][si]),
        layer=layer,head=head,arm=arm,delta_saved_logp=float(z[y]-logsumexp(z)-z0[y]+logsumexp(z0)),
        delta_saved_logodds=float(odds(z)-odds(z0)),
        delta_anchor_logodds=routeodds(changed)-routeodds(base),
        anchor0_delta=float(p[0]-p0[0]),anchor1_delta=float(p[1]-p0[1]),outside_delta=float(p[2]-p0[2]),
        post_WO_delta_norm=float(np.linalg.norm(changed['attn_write'][si,layer]-base['attn_write'][si,layer])),
        top_before=int(np.argmax(z0)),top_after=int(np.argmax(z)),
        intervention_target=expected_token,
        target_hit=None if expected_token is None else bool(np.argmax(z)==expected_token),
        interpretation='semantic interchange target, not natural truth repair' if expected_token else 'no truth target assigned')


def interventions(root,cases,model,models,points,args):
    from .routing_engine import Intervention,replay,snapshot
    lookup={c['id']:c for c in cases};groups=grouped(cases);jobs=[]
    held=[g for g in groups.values() if next(iter(g.values()))['split']=='heldout'][:args.pairs]
    for g in held:
        base=g[(0,0,0)];qdon=g[(1,0,0)];kdon=g[(0,1,0)]
        vdon=lookup[base['id'].replace('v0','v1')]
        jobs.append((base,qdon,kdon,vdon,None,lookup[base['id']+'_same']))
    for c in cases:
        if c['kind']=='natural' and c['world']=='original':
            jobs.append((c,lookup[c['group']+'__constraint_flip'],lookup[c['group']+'__constraint_flip'],
                         lookup[c['group']+'__constraint_flip'],lookup[c['group']+'__paraphrase_control'],None))
    out=Path(root)/'interventions';out.mkdir(exist_ok=True);rows=[];algebra=[]
    for case,qcase,kcase,vcase,shamcase,samecase in jobs:
        base=load_trace(root,case);qd=load_trace(root,qcase);kd=load_trace(root,kcase);vd=load_trace(root,vcase)
        sham=load_trace(root,shamcase) if shamcase else None
        same=load_trace(root,samecase) if samecase else None
        for point in points:
            l,h=point['layer'],point['head'];mo=models[l] if l in models else models[str(l)]
            for si in args.site_indices:
                if si>=len(case['queries']):continue
                qpos=case['queries'][si]
                qs,ks,vs=[snapshot(t,si,l) for t in (qd,kd,vd)]
                original=snapshot(base,si,l)
                qb=np.asarray(mo['query_space']);kb=np.asarray(mo['address_space'])
                specs=[('same_world','q',original,None,None),('q','q',qs,None,None),('k','k',ks,None,None),
                       ('qk','qk',qs,ks,None),('v_content','v',vs,None,None)]
                if qb.size:
                    specs.extend((a,a,qs,None,qb) for a in ('q_subspace','q_visible','q_null','q_random'))
                if kb.size:specs.extend((a,a,ks,None,kb) for a in ('k_subspace','k_random'))
                if same is not None:
                    ss=snapshot(same,si,l);specs.append(('q_same_binding','q',ss,None,None))
                    if qb.size:specs.extend([('q_same_subspace','q_subspace',ss,None,qb),('q_same_random','q_random',ss,None,qb)])
                if sham is not None:
                    sh=snapshot(sham,si,l);specs.extend([('q_paraphrase','q',sh,None,None),('k_paraphrase','k',sh,None,None)])
                # Upstream writer tests are separate, using the SAME query and a layer before the read.
                if l>0:
                    specs.extend((a,a,snapshot(qd,si,l-1),None,None) for a in ('attn_write','mlp_write'))
                for arm,kind,don,keydon,basis in specs:
                    key=f'{case["id"]}_s{si}_l{l}h{h}_{arm}'
                    path=out/(key+'.json')
                    if args.resume and path.exists():
                        rows.append(json.loads(path.read_text()));continue
                    patch=Intervention(l-1 if kind.endswith('_write') else l,h,qpos,kind,don,keydon,basis,args.seed)
                    print(f'  {case["id"]} site={si} L{l}H{h} {arm}',flush=True)
                    changed=replay(model,case,patch)
                    # Same-world and prefix invariance are correctness checks, not research findings.
                    prior=np.asarray(case['queries'])<qpos
                    prefix_error=float(np.max(np.abs(changed['logits'][prior]-base['logits'][prior]))) if prior.any() else 0.
                    if prefix_error>1e-5:raise ValueError('intervention leaked into earlier predictions')
                    if arm=='same_world' and np.max(np.abs(changed['logits']-base['logits']))>1e-5:
                        raise ValueError('same-world Q transfer failed: native computation mismatch')
                    target=None
                    if case['kind']=='constructed':
                        if arm in ('q','k','q_subspace','k_subspace','q_visible'):target=case['candidate_ids'][1]
                        elif arm in ('same_world','qk','q_same_binding','q_same_subspace'):target=case['candidate_ids'][0]
                        elif arm=='v_content':target=vcase['candidate_ids'][0]
                    r=effect_row(base,changed,case,si,l,h,arm,target);r.update(prefix_max_error=prefix_error)
                    # Preserve full non-averaged route, Q,K and all actual post-WO change vectors at this site.
                    np.savez_compressed(out/(key+'.npz'),query=qpos,
                        q=changed['q'][si,l],k=changed['k'][si,l],v=changed['v'][si,l],
                        attention=changed['attention'][si,l],post_WO_delta=changed['post_WO_delta'][si],
                        logits=changed['logits'],queries=case['queries'],anchors=case['anchors'],
                        read_error=changed['read_error'],attn_write=changed['attn_write'],mlp_write=changed['mlp_write'])
                    write(path,r);rows.append(r);del changed;gc.collect()
                # Use actual operator outputs in RECEIVER RoPE coordinates, never raw
                # donor rotated keys (donor sequence lengths/record positions may differ).
                prefix=f'{case["id"]}_s{si}_l{l}h{h}_'
                with np.load(out/(prefix+'q.npz')) as qr,np.load(out/(prefix+'k.npz')) as kr,np.load(out/(prefix+'qk.npz')) as both:
                    decomposition=decompose_qk(base['q'][si,l,h].astype(float),qr['q'][h].astype(float),
                        base['k'][si,l,h].astype(float),kr['k'][h].astype(float),base['q'].shape[-1]**-.5)
                    bq,bk=both['q'][h].astype(float),both['k'][h].astype(float)
                    actual=(bq@(bk[1]-bk[0])-base['q'][si,l,h].astype(float)@(
                        base['k'][si,l,h,1].astype(float)-base['k'][si,l,h,0].astype(float)))*base['q'].shape[-1]**-.5
                    algebra.append(dict(case=case['id'],site=si,layer=l,head=h,**decomposition,
                        qk_intervention_total=float(actual),coordinate_check_error=float(abs(actual-decomposition['total'])),
                        note='single-head simultaneous frozen QK in receiver coordinates; algebraic validation, not a discovered mechanism'))
    table(Path(root)/'intervention_effects.csv',rows);table(Path(root)/'qk_decomposition.csv',algebra)
    write(Path(root)/'interventions_complete.json',dict(complete=True,rows=len(rows),jobs=len(jobs),
        hypothesis='reference-decoded constraints vs head matching vs semantic output',
        no_automatic_natural_hallucination_labels=True))
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',default=data.DEFAULT_SAMPLES);p.add_argument('--states',default=None)
    p.add_argument('--output',default='outputs/lookback_routing_v1')
    p.add_argument('--phase',choices=('all','capture','analyze','intervene'),default='all')
    p.add_argument('--suite',choices=('both','controlled'),default='both')
    p.add_argument('--groups',type=int,default=16);p.add_argument('--filler-records',type=int,default=8)
    p.add_argument('--rank',type=int,default=4);p.add_argument('--nuisance-rank',type=int,default=2)
    p.add_argument('--probe-rank',type=int,default=16)
    p.add_argument('--channels',type=int,default=1,help='reference-only highest MI channels; no test head selection')
    p.add_argument('--pairs',type=int,default=2,help='first heldout groups, including successes and failures')
    p.add_argument('--site-indices',type=int,nargs='+',default=[0],help='0=commitment; 1/2=entry/completion')
    p.add_argument('--select',nargs='+');p.add_argument('--model');p.add_argument('--dtype',default=None)
    p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=20260915)
    p.add_argument('--replay-atol',type=float,default=.1);p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv)
    if (min(args.rank,args.nuisance_rank,args.probe_rank,args.channels,args.pairs)<1 or min(args.site_indices)<0
            or len(set(args.site_indices))!=len(args.site_indices) or args.replay_atol<0 or not np.isfinite(args.replay_atol)):
        p.error('positive ranks/channel/pair counts and nonnegative sites required')
    root=Path(args.output);settings_path=Path(args.samples)/'settings.json'
    settings=json.loads(settings_path.read_text()) if settings_path.exists() else {}
    model_path=args.model or settings.get('model');dtype=args.dtype or settings.get('dtype','bfloat16')
    if not model_path:raise ValueError('--model is needed when original sample settings are unavailable')
    parameters={k:v for k,v in vars(args).items() if k not in ('phase','device','resume','output','model','dtype')}
    code_hash=hashlib.sha256(b''.join((Path(__file__).parent/n).read_bytes() for n in
              ('routing_math.py','routing_engine.py','routing_data.py','routing_audit.py'))).hexdigest()
    config=dict(version=VERSION,code_hash=code_hash,model=str(Path(model_path).resolve()),dtype=dtype,parameters=parameters,
                semantic_targets='constructed selector and binding, not hallucination annotations')
    if root.exists():
        if not args.resume or not (root/'config.json').exists() or json.loads((root/'config.json').read_text())!=config:
            raise ValueError('choose a new output, or --resume with identical experimental parameters')
    else:root.mkdir(parents=True);write(root/'config.json',config)
    if args.phase in ('all','capture'):
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        cases=data.controlled(tokenizer,args.groups,args.filler_records);original_checks={}
        if args.suite=='both':
            states=args.states or str(Path(args.samples).parent/('states_'+Path(args.samples).name))
            natural,original_checks=data.natural(tokenizer,args.samples,states,args.select);cases+=natural
        write(root/'cases.json',cases)
        model=None
        try:
            for i,c in enumerate(cases):
                path=root/'capture'/(c['id']+'.npz')
                if path.exists() and args.resume:
                    t=load_trace(root,c)
                    if c['kind']=='natural' and c['world']=='original':
                        checked=check_original(t,original_checks[c['group']],args.replay_atol)
                        if not checked['valid']:raise ValueError('resumed original replay no longer matches')
                    continue
                if model is None:model=load_model(model_path,dtype,args.device)
                from .routing_engine import replay
                print(f'capture {i+1}/{len(cases)} {c["id"]}',flush=True)
                t=replay(model,c)
                if c['kind']=='natural' and c['world']=='original':
                    checked=check_original(t,original_checks[c['group']],args.replay_atol)
                    write(root/(c['group']+'.replay_check.json'),checked)
                    if not checked['valid']:raise ValueError('original-cache replay differs; stop before interpretation')
                save_trace(root,c,t);del t
            write(root/'capture_complete.json',dict(complete=True,cases=len(cases),new_responses_generated=0))
        finally:
            del model;gc.collect()
            import torch
            if torch.cuda.is_available():torch.cuda.empty_cache()
    cases=json.loads((root/'cases.json').read_text())
    if args.phase in ('all','analyze'):
        models,points=fit_and_measure(root,cases,args)
    if args.phase in ('all','intervene'):
        if args.phase=='intervene':
            models=load_models(root);points=json.loads((root/'measurement.json').read_text())['points']
        model=load_model(model_path,dtype,args.device)
        try:interventions(root,cases,model,models,points,args)
        finally:del model;gc.collect()
    write(root/'run_status.json',dict(phase=args.phase,finished=True,trained_llm_parameters=0,
        note='This is a semantic mechanism audit, not a validated unsupervised hallucination detector.'))
    print('Results:',root/'channel_measures.csv',root/'intervention_effects.csv',flush=True)


def load_model(path,dtype,device):
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=getattr(torch,dtype),
             attn_implementation='eager').to(device).eval().requires_grad_(False)


if __name__=='__main__':main()
