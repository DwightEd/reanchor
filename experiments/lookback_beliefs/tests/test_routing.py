"""Math and real-PyTorch execution tests; not natural mechanism success claims."""
import json
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from scipy.special import expit

from experiments.lookback_beliefs.routing_math import *
from experiments.lookback_beliefs.routing_engine import replay, snapshot, Intervention, changed_head, rope
from experiments.lookback_beliefs.routing_data import controlled, natural, altered_prompt
from experiments.lookback_beliefs.routing_audit import check_original, main

torch.set_num_threads(1)

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 4
        self.scaling = .5
        self.q_proj = nn.Linear(16, 16)
        self.k_proj = nn.Linear(16, 8)
        self.v_proj = nn.Linear(16, 8)
        self.o_proj = nn.Linear(16, 16)

    def forward(self, hidden_states, position_embeddings, cache=None, index=None):
        x = hidden_states
        q = self.q_proj(x).reshape(1, x.shape[1], 4, 4).transpose(1, 2)
        k = self.k_proj(x).reshape(1, x.shape[1], 2, 4).transpose(1, 2)
        v = self.v_proj(x).reshape(1, x.shape[1], 2, 4).transpose(1, 2)
        cos, sin = (a[:, None] for a in position_embeddings)
        def rotate(a):
            l, r = a.chunk(2, -1)
            return torch.cat((-r, l), -1)
        q, k = q*cos+rotate(q)*sin, k*cos+rotate(k)*sin
        past = 0
        if cache is not None:
            if index in cache:
                oldk, oldv = cache[index]
                past = oldk.shape[-2]
                k, v = torch.cat((oldk, k), -2), torch.cat((oldv, v), -2)
            cache[index] = (k, v)
        k, v = k.repeat_interleave(2, 1), v.repeat_interleave(2, 1)
        z = (q @ k.transpose(-1, -2)) * self.scaling
        pos = torch.arange(past, past+x.shape[1])
        mask = torch.arange(k.shape[-2])[None, :] > pos[:, None]
        a = z.masked_fill(mask, -torch.inf).float().softmax(-1).to(q.dtype)
        context = (a @ v).transpose(1, 2).reshape(1, x.shape[1], 16)
        return self.o_proj(context), a


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()
        self.norm = nn.LayerNorm(16)
        self.mlp = nn.Sequential(nn.Linear(16, 16), nn.Tanh())

    def forward(self, x, position_embeddings, cache=None, index=None):
        a, weights = self.self_attn(self.norm(x), position_embeddings=position_embeddings, cache=cache, index=index)
        x = x + a
        self.last_attention = weights
        return (x + self.mlp(x),)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1024, 16)
        self.layers = nn.ModuleList([Block() for _ in range(4)])
        self.norm = nn.LayerNorm(16)

    def forward(self, input_ids, past_key_values=None, use_cache=False, **kw):
        cache = ({} if past_key_values is None else past_key_values) if use_cache else None
        start = 0 if not cache else cache[0][0].shape[-2]
        x = self.embed(input_ids)
        phase = torch.arange(start, start+x.shape[1]).float()[:, None]*torch.tensor([.07,.03,.07,.03])[None]
        pos = (phase.cos()[None].to(x.dtype), phase.sin()[None].to(x.dtype))
        for i, layer in enumerate(self.layers):
            x = layer(x, pos, cache=cache, index=i)[0]
        return SimpleNamespace(last_hidden_state=self.norm(x), past_key_values=cache,
                               attentions=tuple(b.last_attention for b in self.layers))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Decoder()
        self.lm_head = nn.Linear(16, 1024, bias=False)
        self.config = SimpleNamespace(model_type='llama', num_attention_heads=4, hidden_size=16)
        self.calls = []

    def get_input_embeddings(self):
        return self.model.embed

    def forward(self, input_ids, past_key_values=None, **kwargs):
        self.calls.append((input_ids.shape[1], past_key_values is None))
        x = self.model(input_ids, past_key_values=past_key_values, **kwargs)
        return SimpleNamespace(logits=self.lm_head(x.last_hidden_state), past_key_values=x.past_key_values,
                               attentions=x.attentions)



def toy():
    torch.manual_seed(7)
    m=Model().eval().requires_grad_(False)
    c=dict(id='toy',ids=list(range(1,14)),prompt_length=4,queries=[3,7,12],anchors=[1,2],
           expected_tokens=[9,10,11],site_names=['a','b','c'],group='toy')
    return m,c


class Tokenizer:
    def __init__(self):self.tokens={};self.inverse={}
    def __call__(self,text,**kwargs):
        pieces=list(re.finditer(r' ?[A-Za-z]+| ?[0-9]+|[^\w\s]|\s+|_',text))
        assert ''.join(x.group() for x in pieces)==text
        ids=[]
        for p in pieces:
            if p.group() not in self.tokens:
                i=len(self.tokens)+1;self.tokens[p.group()]=i;self.inverse[i]=p.group()
            ids.append(self.tokens[p.group()])
        return dict(input_ids=ids,offset_mapping=[p.span() for p in pieces])
    def encode(self,text,**kwargs):return self(text)['input_ids']
    def decode(self,ids,**kwargs):return ''.join(self.inverse[int(i)] for i in ids)
    def apply_chat_template(self,messages,**kwargs):return '<user>'+messages[0]['content']+'</user><assistant>'


def test_information_not_truth_or_hidden_entropy():
    assert channel_information([[.5,.5,0],[.5,.5,0]])==pytest.approx(0)
    assert channel_information([[1,0,0],[0,1,0]])==pytest.approx(1)
    assert channel_information([[0,1,0],[1,0,0]])==pytest.approx(1) # perfectly WRONG is still informative
    assert channel_information([[.01,0,.99],[0,.01,.99]])==pytest.approx(.01)
    with pytest.raises(ValueError):channel_information([[.01,0],[0,.01]])


def test_softmax_exact_qk_decomposition():
    rng=np.random.default_rng(3)
    q0,q1=rng.normal(size=(2,8));k0,k1=rng.normal(size=(2,2,8));scale=8**-.5
    terms=decompose_qk(q0,q1,k0,k1,scale)
    assert terms['q_effect']+terms['k_effect']+terms['interaction']==pytest.approx(terms['total'])
    z=k0@q0*scale;prob=np.exp(z-logsumexp(z))
    assert np.log(prob[1]/prob[0])==pytest.approx(contrast(q0,k0,scale))


def test_query_information_can_be_orthogonal_to_address_decision():
    keys=np.array([[1.,0.,0.],[-1.,0.,0.]])
    delta=np.array([0.,8.,0.]) # large encoded difference, zero address contrast
    visible,null=address_components(delta,keys)
    np.testing.assert_allclose(visible,0)
    np.testing.assert_allclose(null,delta)
    assert contrast(delta,keys,1)==0


def test_subspace_not_edge_reconstruction():
    signal=np.array([[4,1,0.],[-4,1,0.]])
    nuisance=np.array([[0,1,0.],[0,2,0.]])
    fit=fit_constraint_space(signal,nuisance,1,1)
    np.testing.assert_allclose(abs(fit['basis']),[[1,0,0]],atol=1e-7)
    empty=fit_constraint_space(nuisance,nuisance,1,1)
    assert len(empty['basis'])==0


def test_variational_readout_is_not_clipped_into_fake_information():
    y=np.array([0,1]*10);bad=np.where(y==1,-5.,5.)
    assert accessible_bits(bad,y)<0
    x=np.column_stack((2*y-1,np.zeros(20)))
    probe=fit_probe(x,y,rank=2)
    assert np.mean((probe_logits(probe,x)>0)==y)==1


def test_balanced_factorial_and_heldout_sources():
    rows=controlled(Tokenizer(),4,0)
    mainrows=[r for r in rows if not r['content'] and not r.get('style')]
    assert len(mainrows)==32
    for r in mainrows:
        assert r['correct_index']==r['c']^r['b']
        assert [r['ids'][k] for k in r['anchors']]==r['candidate_ids']
        assert r['queries']==[len(r['ids'])-1]
    assert {r['group'] for r in rows if r['split']=='reference'}.isdisjoint(
           {r['group'] for r in rows if r['split']=='heldout'})
    base=next(r for r in rows if r['id']=='controlled_000_c0b0o0v0')
    same=next(r for r in rows if r['id']==base['id']+'_same')
    assert same['anchors']==base['anchors'] and same['expected_tokens']==base['expected_tokens']


def test_exact_original_incremental_schedule_and_no_parameter_change():
    m,c=toy();weights={k:v.clone() for k,v in m.state_dict().items()}
    r=replay(m,c)
    assert m.calls==[(4,True)]+[(1,False)]*9
    assert r['x'].shape==(3,4,16) and r['k'].shape==(3,4,4,2,4)
    assert all(torch.equal(m.state_dict()[k],v) for k,v in weights.items())
    assert all(not x._forward_hooks and not x._forward_pre_hooks for x in m.modules())


@pytest.mark.parametrize('kind',['q','k','qk','v'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_same_world_operator_is_noop(kind,dtype):
    m,c=toy();m=m.to(dtype);base=replay(m,c)
    p=Intervention(1,1,7,kind,snapshot(base,1,1))
    new=replay(m,c,p)
    np.testing.assert_array_equal(new['logits'],base['logits'])


@pytest.mark.parametrize('kind',['q','k','qk','v','q_subspace','k_subspace','q_visible','q_null','q_random','k_random'])
def test_single_physical_head_other_gqa_heads_and_earlier_queries_fixed(kind):
    m,c=toy();base=replay(m,c)
    d={**c,'ids':c['ids'].copy()};d['ids'][2]=37;d['ids'][6]=31
    donor=replay(m,d);basis=np.eye(16)[:3]
    p=Intervention(1,1,7,kind,snapshot(donor,1,1),basis=basis)
    new=replay(m,c,p)
    np.testing.assert_array_equal(new['logits'][0],base['logits'][0])
    # A native KV-level K edit would inadvertently change sibling head 0 too.
    np.testing.assert_allclose(new['attention'][1,1,[0,2,3]],base['attention'][1,1,[0,2,3]],atol=0,rtol=0)
    assert np.isfinite(new['logits']).all()
    assert all(not x._forward_hooks and not x._forward_pre_hooks for x in m.modules())


def test_query_null_preserves_candidate_odds_not_whole_softmax():
    m,c=toy();base=replay(m,c);d={**c,'ids':c['ids'].copy()};d['ids'][6]=45
    donor=replay(m,d)
    p=Intervention(1,1,7,'q_null',snapshot(donor,1,1),basis=np.eye(16))
    new=replay(m,c,p)
    a=base['attention'][1,1,1,c['anchors']];b=new['attention'][1,1,1,c['anchors']]
    assert np.log(a[1]/a[0])==pytest.approx(np.log(b[1]/b[0]),abs=2e-6)


def test_v_swap_keeps_routing_but_moves_postwo():
    m,c=toy();base=replay(m,c);d={**c,'ids':c['ids'].copy()};d['ids'][2]=38
    donor=replay(m,d);new=replay(m,c,Intervention(1,1,7,'v',snapshot(donor,1,1)))
    np.testing.assert_allclose(new['attention'][1,1],base['attention'][1,1],atol=2e-7)
    assert np.linalg.norm(new['post_WO_delta'][1,1])>0
    assert np.max(abs(new['logits'][1]-base['logits'][1]))>0


@pytest.mark.parametrize('kind',['attn_write','mlp_write'])
def test_upstream_writer_reaches_downstream_q(kind):
    m,c=toy();base=replay(m,c);d={**c,'ids':c['ids'].copy()};d['ids'][6]=38
    donor=replay(m,d);new=replay(m,c,Intervention(0,1,7,kind,snapshot(donor,1,0)))
    assert np.linalg.norm(new['q'][1,1]-base['q'][1,1])>0
    np.testing.assert_array_equal(new['logits'][0],base['logits'][0])


def test_failed_intervention_cleans_hooks():
    m,c=toy();b=replay(m,c)
    with pytest.raises(ValueError,match='subspace'):
        replay(m,c,Intervention(1,1,7,'q_subspace',snapshot(b,1,1)))
    assert all(not x._forward_hooks and not x._forward_pre_hooks for x in m.modules())


def test_native_cases_do_not_invent_correct_onion_number(tmp_path):
    tok=Tokenizer();prefix=('Reduce heat to medium and cook another 10 to 12 minutes. '
       'Remove the bratwurst from the beer mixture; reduce heat to low, and continue cooking the onions. '
       'Cook bratwurst on preheated grill for 10 to 14 minutes. Answer:')
    response='Cook the onions in the beer mixture for 10 to 12 minutes.'
    ids=tok.encode(prefix+response);p=len(tok.encode(prefix));n=len(ids)
    samples=tmp_path/'samples';samples.mkdir();row=dict(source_id='14375',seed=0,trace='x.npz',response=response)
    (samples/'samples.jsonl').write_text(json.dumps(row)+'\n')
    np.savez(samples/'x.npz',token_ids=ids,prompt_length=p,top_ids=np.ones((n-p,5),int),
             top_logits=np.zeros((n-p,5)),log_normalizer=np.zeros(n-p),attention=np.zeros((4,4,n-p,n)))
    rows,checks=natural(tok,samples,select=['cooking_onion'])
    base,flip,sham=rows
    assert base['correct_index'] is None and flip['correct_index'] is None
    assert base['expected_tokens']==flip['expected_tokens']==sham['expected_tokens']
    assert base['ids'][base['prompt_length']:]==flip['ids'][flip['prompt_length']:]
    assert 'cook the onions for 10 to 12 minutes' in flip['input_text']
    assert 'another 10 to 12 minutes' in base['input_text']


def test_replay_mismatch_does_not_become_mechanism():
    m,c=toy();r=replay(m,c);ids=np.argsort(-r['logits'],axis=1)[:,:5]
    check=dict(top_ids=ids,top_logits=np.take_along_axis(r['logits'],ids,1),
               log_normalizer=logsumexp(r['logits'],axis=1),attention=r['attention'].astype(np.float16))
    assert check_original(r,check)['valid']
    check['top_logits']+=.25
    assert not check_original(r,check)['valid']


def test_complete_cpu_pipeline_freezes_before_interventions(tmp_path,monkeypatch):
    from experiments.lookback_beliefs import routing_audit as audit
    tok=Tokenizer();m,_=toy()
    fake=SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:tok))
    monkeypatch.setitem(sys.modules,'transformers',fake)
    monkeypatch.setattr(audit,'load_model',lambda *a:m)
    root=tmp_path/'run'
    argv=['--model',str(tmp_path/'model'),'--suite','controlled','--groups','4','--filler-records','0',
          '--rank','2','--nuisance-rank','1','--probe-rank','2','--pairs','1','--channels','1',
          '--device','cpu','--dtype','float32','--output',str(root)]
    audit.main(argv)
    assert (root/'measurement.json').exists() and (root/'subspaces.json').exists()
    assert json.loads((root/'interventions_complete.json').read_text())['complete']
    effects=list(__import__('csv').DictReader((root/'intervention_effects.csv').open()))
    assert {'q','k','qk','v_content','q_same_binding'}<=set(x['arm'] for x in effects)
    audit.main(argv+['--resume','--phase','analyze'])


def test_optional_tiny_huggingface_llama():
    tr=pytest.importorskip('transformers')
    config=tr.LlamaConfig(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=4,
                         num_attention_heads=4,num_key_value_heads=2,attention_dropout=0.)
    config._attn_implementation='eager'
    m=tr.LlamaForCausalLM(config).eval().requires_grad_(False)
    _,case=toy();base=replay(m,case)
    sham=replay(m,case,Intervention(1,0,7,'q',snapshot(base,1,1)))
    np.testing.assert_array_equal(base['logits'],sham['logits'])


def test_receiver_frame_not_donor_position_for_qk():
    from experiments.lookback_beliefs.routing_math import receiver_coordinates,decompose_qk
    m,c=toy();base=replay(m,c)
    donor={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in base.items()}
    donor['q']*=17;donor['k']*=-11   # must be ignored: these are donor rotary coordinates
    q,k=receiver_coordinates(base,donor,1,1,1)
    np.testing.assert_allclose(q,base['q'][1,1,1],atol=2e-7)
    np.testing.assert_allclose(k,base['k'][1,1,1],atol=2e-7)


def test_compact_models_round_trip(tmp_path):
    from experiments.lookback_beliefs.routing_audit import save_models,load_models
    model={1:{'query_space':np.eye(3)[:1], 'q_probe_h2':{'center':np.zeros(3),'weight':np.ones(2)}}}
    save_models(tmp_path,model);back=load_models(tmp_path)
    np.testing.assert_array_equal(back[1]['query_space'],model[1]['query_space'])
    np.testing.assert_array_equal(back[1]['q_probe_h2']['weight'],model[1]['q_probe_h2']['weight'])


def test_nonfinite_information_is_rejected():
    with pytest.raises(ValueError):channel_information([[np.nan,0],[0,1]])


def test_actual_qk_joint_in_receiver_frame_matches_decomposition():
    from experiments.lookback_beliefs.routing_math import decompose_qk
    m,c=toy();base=replay(m,c);d={**c,'ids':c['ids'].copy()};d['ids'][2]=41;d['ids'][6]=32
    donor=replay(m,d);snap=snapshot(donor,1,1)
    q=replay(m,c,Intervention(1,1,7,'q',snap));k=replay(m,c,Intervention(1,1,7,'k',snap))
    both=replay(m,c,Intervention(1,1,7,'qk',snap))
    index=(1,1,1);q0=base['q'][index].astype(float);k0=base['k'][index].astype(float)
    result=decompose_qk(q0,q['q'][index].astype(float),k0,k['k'][index].astype(float),.5)
    q1=both['q'][index].astype(float);k1=both['k'][index].astype(float)
    assert result['total']==pytest.approx((q1@(k1[1]-k1[0])-q0@(k0[1]-k0[0]))*.5,abs=1e-8)
