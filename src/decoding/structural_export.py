"""Export label-free causal scalar features from the frozen RAGTruth observer cache."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2))


def cross_head_spread(source):
    """Per-layer population std across heads, then equal layer mean."""
    return np.asarray(source,dtype=float).std(axis=2).mean(axis=0)


def export(population,output,limit=None):
    population,output=Path(population),Path(output)
    if (population/'COMPLETE').read_text().strip()!='completed=17790 failed=0 total=17790':
        raise ValueError('complete observer population required')
    settings=json.loads((population/'settings.json').read_text())
    if settings['history_window']!=16:raise ValueError('unexpected remote-history definition')
    roster=[json.loads(s) for s in (population/'inputs.jsonl').open()]
    rows=[r for r in roster if r['task']=='QA' and r['generator']=='llama-2-7b-chat']
    if len(rows)!=989:raise ValueError('expected frozen QA train839 + test150 roster')
    if limit is not None:rows=rows[:limit]
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,output/'executed_structural_export.py')
    manifest=dict(complete=False,planned=len(rows),completed=0,labels_read=False,model_forwards=0,
        observer='Llama3.1 replay of original Llama2 responses',history_exclusion=16,
        parent_settings_sha256=sha(population/'settings.json'),parent_inputs_sha256=sha(population/'inputs.jsonl'),
        code_sha256=sha(__file__),parent_hashes={},files={})
    write(output/'manifest.json',manifest)
    exported=[]
    for i,r in enumerate(rows):
        rid=str(r['id']);root=population/'responses'/rid
        parent=json.loads((root/'manifest.json').read_text())
        for name in ['record.json','metrics.npz','profiles.npz','tokens.npz']:
            if sha(root/name)!=parent['files'][name]:raise ValueError('parent artifact hash: '+rid+'/'+name)
        record=json.loads((root/'record.json').read_text())
        for key in ['source_id','response_sha256','prompt_length','official_split']:
            if record[key]!=r[key]:raise ValueError('parent input identity mismatch')
        with np.load(root/'tokens.npz') as tok:
            if tok['token_ids'].tolist()!=r['token_ids'] or tok['offsets'].tolist()!=r['offsets']:
                raise ValueError('token identity mismatch')
        with np.load(root/'profiles.npz') as p,np.load(root/'metrics.npz') as m:
            source=p['source_attention'].astype(float);remote=p['remote_history_attention'].astype(float)
            if source.ndim!=3 or source.shape!=remote.shape:raise ValueError('profile shape')
            values=np.column_stack([m['base__entropy'],-m['base__margin'],source.mean((0,2)),
                                    remote.mean((0,2)),cross_head_spread(source)]).astype(np.float32)
            if len(values)!=len(r['offsets']) or not np.isfinite(values).all():raise ValueError('invalid features')
        artifact=output/(rid+'.npz');np.savez_compressed(artifact,values=values,offsets=r['offsets'])
        exported.append(dict(id=rid,source_id=str(r['source_id']),official_split=r['official_split'],
            response_sha256=r['response_sha256'],tokens=len(values),artifact_sha256=sha(artifact)))
        manifest['parent_hashes'][rid]=sha(root/'manifest.json');manifest['completed']=i+1
        if i%100==0:print(json.dumps(dict(exported=i+1,planned=len(rows))),flush=True)
    with (output/'records.jsonl').open('x') as f:
        for r in exported:f.write(json.dumps(r)+'\n')
    manifest.update(complete=True,records_sha256=sha(output/'records.jsonl'),
        columns=['entropy','negative_margin','source_mass','remote_history_excl16_mass','source_head_std'])
    write(output/'manifest.json',manifest)
    print(json.dumps(dict(complete=True,exported=len(rows),model_forwards=0)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--population',required=True)
    p.add_argument('--output',required=True);p.add_argument('--limit',type=int)
    a=p.parse_args();export(a.population,a.output,a.limit)
