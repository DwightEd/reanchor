"""Small real-PyTorch checks; none are natural-data or hypothesis success claims."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.lookback_beliefs.natural_data import load_trace, prepare_case, cache_measurements
from experiments.lookback_beliefs.natural_engine import cache_carrier, forward, projected_cut, score
from experiments.lookback_beliefs.natural import run, parser


torch.set_num_threads(1)


class CharTokenizer:
    def __call__(self,text,**kw):
        return dict(input_ids=[ord(c) for c in text],offset_mapping=[(i,i+1) for i in range(len(text))])
    def decode(self,ids,**kw): return ''.join(chr(int(i)) for i in ids)


class Attn(nn.Module):
    def __init__(self,d=16,heads=4,kv=2):
        super().__init__();self.head_dim=d//heads;self.heads=heads;self.scaling=self.head_dim**-.5
        self.q_proj=nn.Linear(d,d);self.k_proj=nn.Linear(d,kv*self.head_dim)
        self.v_proj=nn.Linear(d,kv*self.head_dim);self.o_proj=nn.Linear(d,d)
    def forward(self,hidden_states,position_embeddings):
        x=hidden_states; n=x.shape[1]
        q=self.q_proj(x).reshape(1,n,self.heads,-1).transpose(1,2)
        k=self.k_proj(x).reshape(1,n,2,-1).transpose(1,2)
        v=self.v_proj(x).reshape(1,n,2,-1).transpose(1,2)
        cos,sin=(a[:,None] for a in position_embeddings)
        def rot(a):
            l,r=a.chunk(2,-1);return torch.cat((-r,l),-1)
        q,k=q*cos+rot(q)*sin,k*cos+rot(k)*sin
        k=k.repeat_interleave(2,1);v=v.repeat_interleave(2,1)
        logits=(q@k.transpose(-1,-2))*self.scaling
        weights=logits.masked_fill(torch.ones(n,n,dtype=torch.bool).triu(1),-torch.inf).softmax(-1)
        context=(weights@v).transpose(1,2).reshape(1,n,-1)
        return self.o_proj(context),weights


class Block(nn.Module):
    def __init__(self):
        super().__init__();self.self_attn=Attn();self.norm=nn.LayerNorm(16);self.ff=nn.Linear(16,16)
    def forward(self,x,position_embeddings):
        a,_=self.self_attn(self.norm(x),position_embeddings=position_embeddings)
        x=x+a;return (x+torch.tanh(self.ff(x)),)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__();self.embed=nn.Embedding(512,16);self.layers=nn.ModuleList([Block() for _ in range(4)])
        self.norm=nn.LayerNorm(16)
    def forward(self,input_ids,**kw):
        x=self.embed(input_ids);n=x.shape[1]
        phase=torch.arange(n).float()[:,None]*torch.tensor([.07,.03,.07,.03])[None]
        position=(phase.cos()[None],phase.sin()[None])
        for layer in self.layers:x=layer(x,position)[0]
        return SimpleNamespace(last_hidden_state=self.norm(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.model=Decoder();self.lm_head=nn.Linear(16,512,bias=False)
        self.config=SimpleNamespace(model_type='llama',num_attention_heads=4,hidden_size=16,num_hidden_layers=4)
    def get_input_embeddings(self):return self.model.embed


def fixture(tmp_path):
    torch.manual_seed(12);model=Model().eval().requires_grad_(False);tok=CharTokenizer()
    prompt='Only ruler: feathers. Control words of adequate length.\nANSWER:'
    response='They say the feathers are bright. Other text.'
    ids=tok(prompt+response)['input_ids'];p=len(prompt);steps=len(response)
    source=np.ones(p,bool)
    cfg=dict(id='fixture',source_id='14315',seed=2,status='synthetic_test_not_ragtruth',
             expected_trace='00006.npz',target='feathers are bright.',carrier='They',history_seed='feathers',
             constraint=['Only ruler'],payload=['feathers'],control_pool=['Control words of adequate length.'],
             review='synthetic software fixture')
    row=dict(trace='00006.npz',source_id='14315',seed=2,response=response)
    activations={};attentions={};handles=[]
    for li,l in enumerate(model.model.layers):
        def before(m,args,index=li):activations[index]=args[0][0].detach().numpy().copy()
        def attention(m,args,output,index=li):attentions[index]=output[1][0].detach().numpy().copy()
        handles.extend([l.register_forward_pre_hook(before),l.self_attn.register_forward_hook(attention)])
    with torch.no_grad():
        full=model.model(torch.tensor([ids[:-1]])).last_hidden_state[0]
        logits=model.lm_head(full[p-1:]).float()
    for h in handles:h.remove()
    hidden=np.stack([activations[i] for i in range(4)]+[full.numpy()]).astype(np.float16)
    att=np.zeros((4,4,steps,len(ids)),np.float16)
    for li in range(4):att[li,:,:,:-1]=attentions[li][:,p-1:]
    top=logits.topk(5,-1)
    trace=dict(token_ids=np.asarray(ids),prompt_length=np.asarray(p),top_ids=top.indices.numpy(),
               top_logits=top.values.numpy(),log_normalizer=logits.logsumexp(-1).numpy(),attention=att,
               hidden=hidden,source_mask=source,
               logit_entropy=(-(logits.softmax(-1)*logits.log_softmax(-1)).sum(-1)/np.log(2)).numpy())
    item=prepare_case(cfg,row,trace,tok,context=4)
    samples,states=tmp_path/'samples',tmp_path/'states';samples.mkdir();states.mkdir()
    (samples/'samples.jsonl').write_text(json.dumps(row)+'\n')
    settings=dict(model=str(tmp_path/'model'),dtype='float32',device='cpu')
    for folder in (samples,states):(folder/'settings.json').write_text(json.dumps(settings))
    np.savez(samples/row['trace'],**{k:v for k,v in trace.items() if k not in ('hidden','source_mask','logit_entropy')},
             labels=np.array([dict(do_not_read=1)],dtype=object))
    np.savez(states/row['trace'],**{k:trace[k] for k in ('hidden','source_mask','token_ids','logit_entropy')})
    return model,tok,item,trace,cfg,samples,states


def test_load_actual_pair_and_ignore_pickle_labels(tmp_path):
    _,_,_,trace,_,samples,states=fixture(tmp_path)
    row,read=load_trace(samples,states,'14315',2)
    assert row['trace']=='00006.npz'
    np.testing.assert_array_equal(read['token_ids'],trace['token_ids'])


def test_wrong_token_identity_fails(tmp_path):
    _,_,_,trace,_,s,st=fixture(tmp_path)
    trace['token_ids'][0]+=1
    np.savez(st/'00006.npz',**{k:trace[k] for k in ('hidden','source_mask','token_ids','logit_entropy')})
    with pytest.raises(ValueError,match='different generated'):load_trace(s,st,'14315',2)


def test_quote_validation_and_target_not_used_as_carrier(tmp_path):
    _,tok,item,trace,c,_,_=fixture(tmp_path)
    assert item['carrier']<min(item['target_steps'])+item['prompt_length']-1
    c['constraint']=['unobserved quote']
    with pytest.raises(ValueError,match='missing or ambiguous'):prepare_case(c,item['record'],trace,tok)


def test_clothes_case_not_assigned_false_label():
    path=Path(__file__).parents[1]/'natural_cases.json'
    cases=json.loads(path.read_text())
    scope=[c for c in cases if c['id']=='clothes_lengths_scope_control'][0]
    assert scope['status']=='scope_ambiguous_not_gold_error'
    assert 'replacement time' in cases[0]['review']


def test_cache_only_keeps_head_and_layer_axes(tmp_path):
    _,_,item,trace,_,_,_=fixture(tmp_path)
    r=cache_measurements(item,trace)
    assert r['mass_constraint'].shape==(4,4,len(item['steps']))
    assert r['carrier_cosine_by_hidden_index'].shape==(5,len(item['steps']))
    np.testing.assert_array_equal(r['query_positions'],item['prompt_length']+item['steps']-1)


def test_cached_final_hidden_cannot_be_block_patched(tmp_path):
    _,_,item,trace,_,_,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='normalized'):cache_carrier(trace,3,item['carrier'],4)
    assert cache_carrier(trace,1,item['carrier'],4).shape==(1,1,16)


def test_per_head_message_sum_equals_output_projection():
    torch.manual_seed(3)
    a=torch.rand(4,3,8).softmax(-1);v=torch.rand(4,8,4);wo=torch.rand(16,16)
    whole,code=projected_cut(a,v,list(range(8)),wo)
    pieces=[projected_cut(a,v,[j],wo)[0] for j in range(8)]
    torch.testing.assert_close(whole,sum(pieces))
    assert code.shape==(4,3,4)


def test_cut_is_not_local_renormalization():
    a=torch.tensor([[[.2,.8]]]);v=torch.tensor([[[1.],[2.]]]);wo=torch.ones(1,1)
    cut,_=projected_cut(a,v,[0],wo)
    assert cut.item()==pytest.approx(.2)


def test_gqa_rope_zero_cut_and_sham_do_not_modify_predictions(tmp_path):
    m,_,item,_,_,_,_=fixture(tmp_path)
    base=forward(m,item,[1,2]);zero=forward(m,item,[1,2],cut_keys=item['groups']['constraint'],
                                        cut_start=item['carrier'],strength=0)
    torch.testing.assert_close(base['logits'],zero['logits'],rtol=0,atol=0)
    same=forward(m,item,[1,2],patch=(1,base['states'][1]))
    torch.testing.assert_close(same['logits'],base['logits'],rtol=0,atol=0)
    assert zero['replay_error']<1e-6


def test_history_cut_cannot_change_earlier_predictions(tmp_path):
    m,_,item,_,_,_,_=fixture(tmp_path)
    base=forward(m,item,[1,2]);start=item['history_query_start']
    cut=forward(m,item,[1,2],cut_keys=item['groups']['history_seed'],cut_start=start)
    earlier=item['queries']<start;later=~earlier
    torch.testing.assert_close(cut['logits'][earlier],base['logits'][earlier],atol=0,rtol=0)
    assert (cut['logits'][later]-base['logits'][later]).abs().max()>0


def test_hooks_removed_when_intervention_fails(tmp_path):
    m,_,item,_,_,_,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='shape'):
        forward(m,item,[1,2],cut_keys=item['groups']['constraint'],cut_start=item['carrier'],
                patch=(1,torch.zeros(1,1,2)))
    for module in m.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks


def test_frozen_sequence_score_and_js():
    z=torch.tensor([[1.,2.,3.],[4.,2.,1.]])
    r=score(z,[1,0],[2,1],z)
    assert r['saved_vs_frozen_alternative'].tolist()==[-1.,2.]
    np.testing.assert_allclose(r['js_to_baseline_bits'],0,atol=1e-7)
    np.testing.assert_allclose(r['delta_saved_logp'],0,atol=0)


def test_restore_and_reverse_are_actual_downstream_computations(tmp_path):
    m,_,item,_,_,_,_=fixture(tmp_path)
    base=forward(m,item,[1,2]);g=item['groups']['constraint']
    cut=forward(m,item,[1,2],cut_keys=g,cut_start=item['carrier'])
    restore=forward(m,item,[1,2],cut_keys=g,cut_start=item['carrier'],patch=(1,base['states'][1]))
    reverse=forward(m,item,[1,2],patch=(1,cut['states'][1]))
    assert (restore['logits']-cut['logits']).abs().max()>0
    assert (reverse['logits']-base['logits']).abs().max()>0
    assert all(p.grad is None for p in m.parameters())


def test_native_case_end_to_end_cache_and_finite_arms(tmp_path,monkeypatch):
    m,tok,item,trace,c,s,st=fixture(tmp_path)
    fake=SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:tok),
                         AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a,**k:m))
    monkeypatch.setitem(sys.modules,'transformers',fake)
    cases=tmp_path/'cases.json';cases.write_text(json.dumps([c]))
    out=tmp_path/'out'
    args=parser().parse_args(['--samples',str(s),'--states',str(st),'--cases',str(cases),
                             '--output',str(out),'--layers','1','2','--device','cpu','--context','4'])
    before={k:v.detach().clone() for k,v in m.state_dict().items()}
    run(args)
    folder=out/'fixture'
    check=json.loads((folder/'checks.json').read_text())
    assert check['zero_cut_max_logit_error']==0
    assert (folder/'mechanism_complete.json').exists() and (out/'index.html').exists()
    assert (folder/'restore_carrier_l1.npz').exists()
    with np.load(folder/'baseline.npz') as f:
        np.testing.assert_array_equal(f['saved_token_ids'],np.asarray(item['original_ids'])[item['prompt_length']+f['steps']])
    for k,v in m.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    args.resume=True;run(args)
    args.context=3
    with pytest.raises(ValueError,match='same case'):run(args)


def test_optional_real_llama(tmp_path):
    tr=pytest.importorskip('transformers')
    m=tr.LlamaForCausalLM(tr.LlamaConfig(vocab_size=512,hidden_size=16,intermediate_size=32,
        num_attention_heads=4,num_key_value_heads=2,num_hidden_layers=4,attention_dropout=0.)).eval()
    _,_,item,_,_,_,_=fixture(tmp_path)
    base=forward(m,item,[1,2])
    zero=forward(m,item,[1,2],cut_keys=item['groups']['constraint'],cut_start=item['carrier'],strength=0.)
    torch.testing.assert_close(base['logits'],zero['logits'],atol=0,rtol=0)


def test_control_site_is_inside_cut_region_and_really_perturbed(tmp_path):
    m,_,item,_,_,_,_=fixture(tmp_path)
    assert item['source_query_start'] <= item['carrier'] < item['control_carrier']
    assert item['control_carrier'] < min(item['target_steps'])+item['prompt_length']-1
    base=forward(m,item,[1,2])
    cut=forward(m,item,[1,2],cut_keys=item['groups']['constraint'],cut_start=item['source_query_start'])
    assert (base['control_states'][1]-cut['control_states'][1]).norm()>0


def test_nonfinite_cached_entropy_fails(tmp_path):
    _,_,_,trace,_,s,st=fixture(tmp_path)
    trace['logit_entropy'][0]=np.nan
    np.savez(st/'00006.npz',**{k:trace[k] for k in ('hidden','source_mask','token_ids','logit_entropy')})
    with pytest.raises(ValueError,match='nonfinite'):load_trace(s,st,'14315',2)


def test_unknown_patch_layer_does_not_silently_skip(tmp_path):
    m,_,item,_,_,_,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='patch layer'):
        forward(m,item,[1,2],patch=(0,torch.zeros(1,1,16)))
