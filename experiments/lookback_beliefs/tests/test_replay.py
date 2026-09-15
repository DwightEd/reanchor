"""Execution-path tests, not evidence for a natural hallucination mechanism."""
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch
from torch import nn

from experiments.lookback_beliefs.natural_engine import forward
from experiments.lookback_beliefs.natural import mechanism_case, parser
from experiments.lookback_beliefs.replay_checks import compare_replay


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
        self.ff = nn.Linear(16, 16)

    def forward(self, x, position_embeddings, cache=None, index=None):
        a, weights = self.self_attn(self.norm(x), position_embeddings=position_embeddings, cache=cache, index=index)
        x = x + a
        self.last_attention = weights
        return (x + torch.tanh(self.ff(x)),)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 16)
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
        self.lm_head = nn.Linear(16, 64, bias=False)
        self.config = SimpleNamespace(model_type='llama', num_attention_heads=4, hidden_size=16)
        self.calls = []

    def get_input_embeddings(self):
        return self.model.embed

    def forward(self, input_ids, past_key_values=None, **kwargs):
        self.calls.append((input_ids.shape[1], past_key_values is None))
        x = self.model(input_ids, past_key_values=past_key_values, **kwargs)
        return SimpleNamespace(logits=self.lm_head(x.last_hidden_state), past_key_values=x.past_key_values,
                               attentions=x.attentions)


def fixture(dtype=torch.float32):
    torch.manual_seed(17)
    model = Model().to(dtype).eval().requires_grad_(False)
    p, n = 4, 12
    item = dict(ids=list(range(1,n+1)), original_ids=list(range(1,n+2)), prompt_length=p,
                queries=np.arange(p-1,n), steps=np.arange(n-p+1), target_steps=np.array([5,6]),
                carrier=4, control_carrier=6, source_query_start=4, history_query_start=9,
                execution='incremental_kv', case=dict(id='fixture', status='software_only'),
                groups=dict(constraint=np.array([1]),payload=np.array([2]),
                            control_constraint=np.array([0]),control_payload=np.array([3]),
                            history_seed=np.array([9]),history_control=np.array([8])))
    return model, item


@torch.inference_mode()
def independent_saved_replay(model, item):
    p, n = item['prompt_length'], len(item['ids'])
    hs = np.zeros((5, n, 16), np.float16)
    att = np.zeros((4,4,n-p+1,n+1), np.float16)
    ids = torch.tensor([item['ids']]); cache = None; logits = []
    point = {}; hooks = []
    for i,b in enumerate(model.model.layers):
        def read(m,args,index=i):
            point[index] = args[0][0].detach().float().cpu().numpy()
        hooks.append(b.register_forward_pre_hook(read))
    try:
        for t in range(n-p+1):
            stop = p+t; start = 0 if t==0 else stop-1
            out = model(input_ids=ids[:,start:stop], past_key_values=cache, use_cache=True,
                        attention_mask=torch.ones_like(ids[:,:stop]), output_attentions=True, return_dict=True)
            logits.append(out.logits[0,-1].float().clone())
            for i in range(4):
                hs[i,start:stop] = point[i]
                att[i,:,t,:stop] = out.attentions[i][0,:,-1].float().numpy()
            cache = out.past_key_values
    finally:
        for h in hooks: h.remove()
    z = torch.stack(logits); top = z.topk(5,-1)
    return dict(top_ids=top.indices.numpy(),top_logits=top.values.numpy(),
                log_normalizer=z.logsumexp(-1).numpy(),attention=att,hidden=hs), z


def test_matches_original_kv_schedule_and_logits():
    m,i = fixture();trace,expected=independent_saved_replay(m,i)
    m.calls.clear()
    r=forward(m,{**i,'cached_attention':trace['attention']},[1,2])
    torch.testing.assert_close(r['logits'],expected,atol=0,rtol=0)
    assert m.calls == [(4,True)]+[(1,False)]*8
    assert not r['cached_attention_errors'].any()


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_zero_cut_and_same_world_patch(dtype):
    m,i=fixture(dtype)
    b=forward(m,i,[1,2])
    z=forward(m,i,[1,2],cut_keys=[1,2],cut_start=4,strength=0.)
    s=forward(m,i,[1,2],patch=(1,b['states'][1]))
    torch.testing.assert_close(z['logits'],b['logits'],atol=0,rtol=0)
    torch.testing.assert_close(s['logits'],b['logits'],atol=0,rtol=0)
    assert z['replay_error'] < 1e-6


def test_full_and_incremental_agree_in_fp32_but_use_distinct_paths():
    m,i=fixture()
    for kwargs in ({},dict(cut_keys=[1,2],cut_start=4)):
        a=forward(m,i,[1,2],**kwargs)
        b=forward(m,{**i,'execution':'full'},[1,2],**kwargs)
        torch.testing.assert_close(a['logits'],b['logits'],atol=2e-6,rtol=1e-5)
        torch.testing.assert_close(a['states'][1],b['states'][1],atol=2e-6,rtol=1e-5)


def test_history_cut_cannot_change_earlier_prediction_including_own_generation():
    m,i=fixture();b=forward(m,i,[1,2]);c=forward(m,i,[1,2],cut_keys=[9],cut_start=9)
    old=i['queries']<9
    torch.testing.assert_close(b['logits'][old],c['logits'][old],atol=0,rtol=0)
    assert (b['logits'][~old]-c['logits'][~old]).abs().max()>0


def test_source_cut_reciprocal_carrier_recovery_and_cache_isolation():
    m,i=fixture();b=forward(m,i,[1,2]);c=forward(m,i,[1,2],cut_keys=[1],cut_start=4)
    r=forward(m,i,[1,2],cut_keys=[1],cut_start=4,patch=(1,b['states'][1]))
    v=forward(m,i,[1,2],patch=(1,c['states'][1]))
    assert (r['logits']-c['logits']).abs().max()>0
    assert (v['logits']-b['logits']).abs().max()>0
    again=forward(m,i,[1,2]);torch.testing.assert_close(again['logits'],b['logits'],atol=0,rtol=0)
    assert sum(fresh for _,fresh in m.calls)==5
    assert all(p.grad is None for p in m.parameters())


def test_removed_codes_keep_physical_heads_and_absolute_sites():
    m,i=fixture();r=forward(m,i,[1,2],cut_keys=[1],cut_start=4)
    for entries in r['removed_codes'].values():
        assert [p for p,_ in entries]==[4,8]
        assert all(code.shape==(4,4) for _,code in entries)


def test_hooks_removed_on_failure():
    m,i=fixture()
    with pytest.raises(ValueError,match='shape'):
        forward(m,i,[1,2],cut_keys=[1],cut_start=4,patch=(1,torch.zeros(1,1,2)))
    assert all(not n._forward_hooks and not n._forward_pre_hooks for n in m.modules())


def test_parameters_unchanged_and_prefix_extension_invariant():
    m,i=fixture();before={k:v.clone() for k,v in m.state_dict().items()}
    full=forward(m,i,[1,2],cut_keys=[1],cut_start=4)
    short={**i,'ids':i['ids'][:10],'queries':i['queries'][:-2]}
    z=forward(m,short,[1,2],cut_keys=[1],cut_start=4)
    torch.testing.assert_close(z['logits'],full['logits'][:-2],atol=0,rtol=0)
    for k,v in m.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)


def test_no_hidden_execution_fallback():
    m,i=fixture()
    with pytest.raises(ValueError,match='unknown replay'):
        forward(m,{**i,'execution':'auto_ignore_error'},[1,2])
    m.train()
    with pytest.raises(ValueError,match='eval'):
        forward(m,i,[1,2])


def test_diagnostics_separate_logits_lse_and_probability_shift():
    m,i=fixture();trace,_=independent_saved_replay(m,i);r=forward(m,i,[1,2]);r['logits']=r['logits']+.25
    c,rows=compare_replay(r,trace,i)
    assert c['saved_top_logit_error']==pytest.approx(.25,abs=1e-6)
    assert c['saved_log_normalizer_error']==pytest.approx(.25,abs=1e-6)
    assert c['saved_top_logp_error']<1e-6
    assert len(rows)==len(i['steps']) and not c['intervention_complete']


def test_cli_defaults_separate_new_output_and_keep_old_tolerance():
    a=parser().parse_args([])
    assert a.execution=='incremental_kv' and a.replay_atol==.1
    assert a.output=='outputs/lookback_beliefs_natural_v2'


def test_failed_historical_check_stops_before_interventions(tmp_path):
    m,i=fixture();trace,_=independent_saved_replay(m,i);trace['top_logits']+=.25
    a=parser().parse_args(['--layers','1','2'])
    with pytest.raises(ValueError,match='No intervention executed'):
        mechanism_case(m,None,i,trace,tmp_path,a)
    assert not (tmp_path/'baseline.npz').exists()
    check=json.loads((tmp_path/'checks.json').read_text())
    assert check['saved_top_logit_error']>.1 and not check['intervention_complete']
    assert (tmp_path/'replay_tokens.csv').exists()


def test_mechanism_case_end_to_end_new_backend(tmp_path):
    m,i=fixture();trace,_=independent_saved_replay(m,i)
    a=parser().parse_args(['--layers','1','2'])
    effects=mechanism_case(m,None,i,trace,tmp_path,a)
    check=json.loads((tmp_path/'checks.json').read_text())
    assert check['intervention_complete'] and check['execution']=='incremental_kv'
    assert check['saved_attention_error']==0
    assert (tmp_path/'cut_history_seed.npz').exists()
    assert (tmp_path/'restore_carrier_l1.npz').exists()
    assert (tmp_path/'mediation.csv').exists() and effects


def test_optional_huggingface_llama_native_kv():
    tr=pytest.importorskip('transformers')
    config=tr.LlamaConfig(vocab_size=64,hidden_size=16,intermediate_size=32,
                         num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,
                         attention_dropout=0.)
    config._attn_implementation='eager'
    m=tr.LlamaForCausalLM(config).eval().requires_grad_(False)
    _,i=fixture();b=forward(m,i,[1,2])
    z=forward(m,i,[1,2],cut_keys=[1],cut_start=4,strength=0.)
    torch.testing.assert_close(b['logits'],z['logits'],atol=0,rtol=0)
    s=forward(m,i,[1,2],patch=(1,b['states'][1]))
    torch.testing.assert_close(b['logits'],s['logits'],atol=0,rtol=0)
