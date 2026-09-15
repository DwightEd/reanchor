"""Readout regression tests; fabricated logits are software fixtures, NOT model results."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
import pytest

from experiments.lookback_beliefs.routing_readout import (
    export, log_probabilities, main, read_trace, reference_donor, site_stats,
    token_stats,
)


def dump(path, data):
    path.write_text(json.dumps(data), encoding='utf-8')


def npz(path, case, logits, *, query=None, signature=True):
    extra = {}
    if signature:
        extra['case_signature'] = hashlib.sha256(json.dumps(case, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if query is not None: extra['query'] = query
    np.savez(path, logits=logits, queries=case['queries'], anchors=case['anchors'], **extra)


def case(cid='base', kind='constructed'):
    c = dict(id=cid, group='g0', kind=kind, split='heldout' if kind=='constructed' else 'natural',
             c=0, b=0, o=0, content=0, prompt_length=3, ids=[0,1,2,3,4,5],
             queries=[2,3,4], anchors=[0,1], candidate_ids=[1,2], expected_tokens=[1,1,2],
             correct_index=0 if kind=='constructed' else None, site_names=['before','commitment','completion'])
    if kind=='natural': c.update(world='original', local_claim_supported=False)
    return c


def make_audit(tmp_path, kind='constructed', target='2', arm='q', site=1):
    root=tmp_path/'audit'; (root/'capture').mkdir(parents=True); (root/'interventions').mkdir()
    c=case(kind=kind); dump(root/'cases.json',[c]); dump(root/'config.json',dict(version='qk-binding-audit-v1',model='/missing/local/model'))
    z=np.array([[1.,4.,2.,0.,5.,-1.],[0.,4.,3.,1.,5.,-1.],[0.,2.,4.,1.,5.,-1.]])
    changed=z.copy()
    if arm!='same_world': changed[site:,2]+=2
    npz(root/'capture/base.npz',c,z)
    name=f'base_s{site}_l1h2_{arm}'
    npz(root/'interventions'/f'{name}.npz',c,changed,query=c['queries'][site],signature=False)
    effect=dict(case='base',group='g0',site=c['site_names'][site],query=c['queries'][site],layer=1,head=2,arm=arm,intervention_target=target)
    with (root/'intervention_effects.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=effect); w.writeheader();w.writerow(effect)
    dump(root/'capture_complete.json',dict(complete=True,cases=1)); dump(root/'interventions_complete.json',dict(complete=True,rows=1))
    return root,c,z,changed,effect,name


def read_csv(root, name):
    with (root/'semantic_readout_review'/name).open(newline='',encoding='utf8') as f:
        return list(csv.DictReader(f))


def test_log_softmax_shift_invariance_and_saturation():
    z=np.array([[10000.,9999.,9998.],[80.,0.,-80.]])
    np.testing.assert_allclose(log_probabilities(z),log_probabilities(z-100000),atol=1e-12)
    lp=log_probabilities(z)
    a=token_stats(z[1],lp[1],0,lambda _:None)
    assert a['probability']==1.
    assert a['logodds_rest']==pytest.approx(80.)  # no 1-p cancellation


@pytest.mark.parametrize('z',[[[np.nan,0]],[[np.inf,0]],[1,2],[[]],[[1.]]])
def test_invalid_logits_rejected(z):
    with pytest.raises(ValueError):log_probabilities(z)


def test_global_gate_and_candidate_preference_are_distinct(tmp_path):
    root,c,z,_,_,_=make_audit(tmp_path)
    trace=read_trace(root/'capture/base.npz',c)
    r=site_stats(trace,c,0,lambda i:str(i))
    assert r['candidate_preference_correct'] is True
    assert r['global_correct'] is False and r['top1_outside_candidates'] is True
    assert r['candidate_total']<1
    assert r['candidate0_rank']==2


def test_natural_anchors_are_not_truth_labels(tmp_path):
    root,_,_,_,_,_=make_audit(tmp_path,kind='natural',target='')
    status=export(root)
    assert status['complete']
    rows=read_csv(root,'candidate_details.csv')
    assert all(r['global_correct_after']=='' and r['delta_correct_candidate_margin']=='' for r in rows)
    assert 'NOT true/false' in rows[0]['candidate_interpretation']


def test_exports_all_baselines_not_only_intervention_jobs(tmp_path):
    root,c,z,_,_,_=make_audit(tmp_path)
    other=copy.deepcopy(c);other.update(id='without_intervention',group='g1')
    dump(root/'cases.json',[c,other]);npz(root/'capture/without_intervention.npz',other,z)
    dump(root/'capture_complete.json',dict(complete=True,cases=2))
    status=export(root)
    assert status['baseline_site_rows']==6 and status['heldout_core']['groups']==2
    assert len(read_csv(root,'baseline_readout.csv'))==6


def test_temporal_scope_and_target_only_at_intervention(tmp_path):
    root,_,_,_,_,_=make_audit(tmp_path)
    status=export(root);assert status['complete']
    rows=read_csv(root,'candidate_details.csv')
    assert [r['temporal_scope'] for r in rows]==['before','at','after']
    assert float(rows[0]['delta_saved_logp'])==0
    assert [r['intervention_target'] for r in rows]==['','2','']
    assert float(rows[2]['delta_logodds_1_over_0'])==pytest.approx(2.)
    assert float(rows[1]['target_logp_after'])>float(rows[1]['target_logp_before'])


def test_foreign_payload_target_has_actual_probability_and_rank(tmp_path):
    root,_,_,_,_,_=make_audit(tmp_path,target='5',arm='v_content')
    export(root);row=read_csv(root,'candidate_details.csv')[1]
    assert row['intervention_target']=='5' and row['target_id_after']=='5'
    assert float(row['target_probability_after'])>0
    assert row['target_hit_before']=='False'


def test_wrong_candidate_target_not_applied_to_future_site(tmp_path):
    root,c,z,_,_,_=make_audit(tmp_path)
    c['correct_index']=1;dump(root/'cases.json',[c]);npz(root/'capture/base.npz',c,z)
    export(root);rows=read_csv(root,'baseline_readout.csv')
    assert rows[1]['candidate_preference_correct']=='False'
    assert float(rows[1]['correct_candidate_margin'])==-1


def test_ties_are_not_arbitrarily_counted_correct(tmp_path):
    root,c,z,_,_,_=make_audit(tmp_path);z[:,2]=z[:,1]
    npz(root/'capture/base.npz',c,z)
    r=site_stats(read_trace(root/'capture/base.npz',c),c,0,lambda _:None)
    assert r['candidate_tie'] and r['candidate_choice'] is None
    assert r['candidate_preference_correct'] is None


def test_prefix_leakage_invalidates_whole_arm(tmp_path):
    root,c,_,changed,_,name=make_audit(tmp_path);changed[0,2]+=1
    npz(root/'interventions'/f'{name}.npz',c,changed,query=3,signature=False)
    status=export(root)
    assert status['invalid'] and status['intervention_site_rows']==0
    assert main(['--audit',str(root),'--allow-partial'])==2


def test_same_world_check_and_wrong_npz_query(tmp_path):
    root,c,z,_,_,name=make_audit(tmp_path,arm='same_world')
    bad=z.copy();bad[1,0]+=1
    npz(root/'interventions'/f'{name}.npz',c,bad,query=3,signature=False)
    status=export(root);assert 'no-op' in status['invalid'][0]['error']
    npz(root/'interventions'/f'{name}.npz',c,z,query=4,signature=False)
    status=export(root);assert 'query' in status['invalid'][0]['error']


def test_missing_npz_is_not_an_effect_of_zero(tmp_path):
    root,_,_,_,_,name=make_audit(tmp_path)
    (root/'interventions'/f'{name}.npz').unlink()
    status=export(root)
    assert not status['complete'] and len(status['missing'])==1 and status['intervention_site_rows']==0
    assert main(['--audit',str(root)])==2
    assert main(['--audit',str(root),'--allow-partial'])==0
    assert (root/'semantic_readout_review.zip').exists()


def test_reads_individual_records_after_interruption(tmp_path):
    root,_,_,_,effect,name=make_audit(tmp_path)
    (root/'intervention_effects.csv').unlink();dump(root/'interventions'/f'{name}.json',effect)
    (root/'interventions_complete.json').unlink()
    status=export(root)
    assert status['index_source']=='interventions/*.json' and status['intervention_site_rows']==3
    assert not status['complete']


def test_detects_truncated_index(tmp_path):
    root,_,_,_,_,_=make_audit(tmp_path)
    dump(root/'interventions_complete.json',dict(complete=True,rows=2))
    status=export(root);assert status['invalid'][0]['kind']=='completion_index'


def test_rejects_metadata_changed_after_capture(tmp_path):
    root,c,_,_,_,_=make_audit(tmp_path);c['candidate_ids']=[2,3]
    dump(root/'cases.json',[c]);status=export(root)
    assert 'signature' in status['invalid'][0]['error']
    assert not status['baseline_site_rows']


def test_no_outputs_or_input_hash_changes_on_repeat(tmp_path):
    root,_,_,_,_,_=make_audit(tmp_path)
    raw={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    first=export(root);second=export(root)
    assert first==second
    for p,b in raw.items():assert p.read_bytes()==b
    with ZipFile(root/'semantic_readout_review.zip') as bundle:
        assert 'semantic_readout_review/baseline_readout.csv' in bundle.namelist()
        assert not any(n.endswith('.npz') for n in bundle.namelist())


def test_decode_loads_only_local_tokenizer(tmp_path,monkeypatch):
    from types import SimpleNamespace
    root,_,_,_,_,_=make_audit(tmp_path);calls=[]
    class Tokenizer:
        def decode(self, ids, **kwargs):return 'T'+str(ids[0])
    def load(path,**kwargs):calls.append((path,kwargs));return Tokenizer()
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load)))
    status=export(root,decode=True)
    assert calls[0][1]=={'local_files_only':True}
    assert status['outside_top_tokens'][0]['text']=='T4'
    assert status['new_model_forwards']==0


def test_absent_tokenizer_falls_back_honestly(tmp_path,monkeypatch):
    root,_,_,_,_,_=make_audit(tmp_path);monkeypatch.setitem(sys.modules,'transformers',None)
    status=export(root,decode=True)
    assert status['tokenizer_status']=='unavailable'
    assert status['outside_top_tokens'][0]['text'] is None


def test_target_is_not_evidence_for_pure_pointer_with_same_payload():
    a=case();b=copy.deepcopy(a);b.update(id='donor',c=1,correct_index=1)
    d=reference_donor(a,'q',{'base':a,'donor':b})
    assert d['donor_declared_answer']==a['candidate_ids'][1]


def test_style_content_controls_not_counted_as_core(tmp_path):
    root,c,z,_,_,_=make_audit(tmp_path)
    controls=[]
    for field in ('style','content'):
        d=copy.deepcopy(c);d.update(id=field);d[field]=1;controls.append(d)
        npz(root/'capture'/f'{field}.npz',d,z)
    dump(root/'cases.json',[c]+controls);dump(root/'capture_complete.json',dict(complete=True,cases=3))
    status=export(root)
    assert status['baseline_site_rows']==9 and status['heldout_core']['queries']==3


def test_natural_world_alignment_accepts_prompt_shift_not_answer_change(tmp_path):
    root,c,z,_,_,_=make_audit(tmp_path,kind='natural',target='')
    d=copy.deepcopy(c);d.update(id='flip',world='constraint_flip',prompt_length=4,queries=[3,4,5],ids=[0,1,2,0,3,4,5])
    dump(root/'cases.json',[c,d]);npz(root/'capture/flip.npz',d,z)
    dump(root/'capture_complete.json',dict(complete=True,cases=2))
    status=export(root);assert status['complete']
    assert len(read_csv(root,'natural_world_effects.csv'))==3
    d['ids'][-1]=0;dump(root/'cases.json',[c,d]);npz(root/'capture/flip.npz',d,z)
    status=export(root);assert any(x['kind']=='world_alignment' for x in status['invalid'])


def test_both_cli_entrypoints_work_without_torch(tmp_path):
    root,_,_,_,_,_=make_audit(tmp_path)
    repo=Path(__file__).resolve().parents[3]
    for entry in [['-m','experiments.lookback_beliefs.routing_readout'],['export_semantic_readout.py']]:
        result=subprocess.run([sys.executable,*entry,'--audit',str(root)],cwd=repo,capture_output=True,text=True)
        assert result.returncode==0,result.stderr


def test_random_control_comparison_same_case_site_channel():
    from experiments.lookback_beliefs.routing_readout import control_comparisons
    r=dict(case='x',intervention_site='a',scored_site='b',layer=1,head=2,temporal_scope='after',
           delta_saved_logp=1.,delta_saved_logodds_rest=2.,delta_logodds_1_over_0=3.,
           delta_correct_candidate_margin=None,arm='q_subspace')
    c={**r,'arm':'q_random','delta_saved_logp':.3}
    rows=control_comparisons([r,c])
    assert len(rows)==1 and rows[0]['delta_saved_logp_difference']==pytest.approx(.7)
    assert rows[0]['delta_correct_candidate_margin_difference'] is None
    c['head']=3
    assert not control_comparisons([r,c])


def test_late_bad_target_does_not_leave_partial_arm_rows(tmp_path):
    root,_,_,_,_,_=make_audit(tmp_path,target='999')
    status=export(root)
    assert status['invalid'] and status['intervention_site_rows']==0


def test_export_does_not_import_torch_when_decode_disabled(tmp_path):
    root,_,_,_,_,_=make_audit(tmp_path)
    repo=Path(__file__).resolve().parents[3]
    program="import sys;sys.modules['torch']=None;sys.modules['transformers']=None;from experiments.lookback_beliefs.routing_readout import export;assert export(sys.argv[1])['complete']"
    r=subprocess.run([sys.executable,'-c',program,str(root)],cwd=repo,capture_output=True,text=True)
    assert r.returncode==0,r.stderr
