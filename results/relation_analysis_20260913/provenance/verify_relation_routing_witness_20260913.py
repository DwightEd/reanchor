from pathlib import Path
import hashlib, json, time
import numpy as np
root=Path('/share/home/tm902089733300000/a903202310/lys/research')
o=root/'reanchor/outputs/relation_routing_20260913'
old=root/'reanchor/outputs/relation_only_20260913'
run=root/'reanchor/outputs/ragtruth_population_20260912'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert (o/'COMPLETE').is_file()
manifest=json.loads((o/'manifest.json').read_text())
for name,want in manifest.items():
    assert sha(o/name)==want,name
settings=json.loads((o/'settings.json').read_text())
for name,want in settings['code_sha256'].items():
    assert sha(Path(name))==want,name
for name,want in settings['inherited_code_sha256'].items():
    assert sha(Path(name))==want,name
rows=json.loads((o/'patches.json').read_text())
expected={f'query_{t}' for t in range(132)}|{f'layer_{i}' for i in range(32)}|{'all_queries','without_final','sham'}
assert len(rows)==167 and {r['condition'] for r in rows}==expected
summary=json.loads((o/'summary.json').read_text())
assert summary['conditions']==167 and summary['baseline_max_error']==0
baseline=np.load(old/'real_after_0_logits.npy',allow_pickle=False)
shapes=set();max_prefix=0;max_mass=0
for r in rows:
    v=np.load(o/f"{r['condition']}_logits.npy",allow_pickle=False)
    shapes.add(v.shape)
    assert v.ndim==1 and v.size==128256 and np.isfinite(v).all()
    assert int(v.argmax())==r['argmax_id']
    assert float(v[975]-v[717])==r['margin_14_minus_12']
    assert r['earlier_query_max_logit_error']==0
    max_prefix=max(max_prefix,r['earlier_query_max_logit_error'])
    assert np.isfinite(r['effect']['js_nats']).all()
    max_mass=max(max_mass,max(d['source_mass_max_error'] for d in r['diagnostics']))
assert np.array_equal(np.load(o/'query_131_logits.npy'),np.load(old/'real_after_0_e_endpoint.npy'))
assert np.array_equal(np.load(o/'sham_logits.npy'),baseline[-1])
assert next(r for r in rows if r['condition']=='sham')['all_query_max_logit_error']==0
snapshot=json.loads(Path('/tmp/relation_routing_interleave_witness_20260913_before.json').read_text())
assert sha(run/'settings.json')==snapshot['settings_sha256']
for name,want in snapshot['code_sha256'].items():
    assert sha(Path(name))==want,name
for name,want in snapshot['manifests_sha256'].items():
    assert sha(run/name)==want,name
result={'verified_unix':time.time(),'manifest_file_count':len(manifest),'conditions':len(rows),'endpoint_shapes':[list(s) for s in shapes],'exact_final_E_reproduction':True,'exact_sham_endpoint_reproduction':True,'runner_exact_complete_baseline_assertion_passed':True,'all_query_sham_error':0,'maximum_earlier_query_error':max_prefix,'maximum_source_mass_error':max_mass,'old_population_manifests_verified':len(snapshot['manifests_sha256']),'old_settings_and_code_unchanged':True,'summary':summary}
Path('/tmp/relation_routing_interleave_witness_20260913_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
