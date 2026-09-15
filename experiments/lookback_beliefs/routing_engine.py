"""Head-local Q/K/V interchange with native downstream computation.

A fresh prompt-prefill + one-token KV replay for EVERY world. Q/K/V replacement
is evaluated only for one physical query head at one selected query. K/V of GQA
siblings are not patched. Post-RoPE candidate-key matching and the *whole visible
softmax row* are recomputed; the native head context is replaced through W_O.
No source removal, no renormalization over selected sources, no JVP.
"""
from dataclasses import dataclass
import inspect
import numpy as np
import torch
import torch.nn.functional as F

from .routing_math import address_components, projected_delta


def rotate_half(x):
    a,b = x.chunk(2,-1)
    return torch.cat((-b,a),-1)


def rope(x, cos, sin):
    return x*cos + rotate_half(x)*sin


def first(output):
    return output if isinstance(output, torch.Tensor) else output[0]


def replace_first(output, value):
    return value if isinstance(output, torch.Tensor) else (value,*output[1:])


@dataclass
class Intervention:
    layer: int
    head: int
    site: int                    # absolute receiver query position
    kind: str
    donor: dict
    key_donor: dict | None = None
    basis: np.ndarray | None = None    # row-orthonormal in normalized residual space
    seed: int = 20260915
    strength: float = 1.


def snapshot(trace, site_index, layer):
    return {**{k:trace[k][site_index,layer] for k in
            ('x','q_raw','k_raw','v','attn_write','mlp_write')}, 'key_x':trace['key_x'][layer]}


def changed_head(q, keys, values, anchor_ids, cos_q, sin_q, cos_k, sin_k,
                 q_weight, native, patch):
    """All q/keys already RoPE-rotated. Donor raw keys rotate at RECEIVER positions.

    Returns full-row attention and one head context. This never edits native KV.
    """
    keys, values, query = keys.clone(), values.clone(), q.clone()
    to = lambda a:torch.as_tensor(a,device=q.device,dtype=q.dtype)
    donor, head = patch.donor, patch.head
    kind = patch.kind
    if kind in ('q','qk','q_subspace','q_visible','q_null','q_random'):
        delta = to(donor['q_raw'][head])-native['q_raw']
        if kind in ('q_subspace','q_visible','q_null','q_random'):
            if patch.basis is None or not len(patch.basis):
                raise ValueError('a nonempty reference-frozen selector subspace is required')
            dx = to(donor['x'])-native['x']
            b = to(patch.basis)
            delta = F.linear((dx@b.T)@b,q_weight,bias=None)
        dr = rope(delta,cos_q,sin_q)
        if kind in ('q_visible','q_null'):
            parallel, null = address_components(dr.float().cpu().numpy(),
                                                 keys[anchor_ids].float().cpu().numpy())
            dr = to(parallel if kind=='q_visible' else null)
        elif kind=='q_random':
            rng=np.random.default_rng(patch.seed)
            random=to(rng.standard_normal(dr.shape))
            dr=random/random.float().norm().clamp_min(1e-20)*dr.float().norm()
        query=query+patch.strength*dr
    if kind in ('k','qk','k_subspace','k_random'):
        donor_k=(patch.key_donor or donor)['k_raw'][head]
        if np.shape(donor_k)!=tuple(keys[anchor_ids].shape):
            raise ValueError('candidate key correspondence must be explicitly aligned')
        new=rope(to(donor_k),cos_k,sin_k)
        if kind in ('k_subspace','k_random'):
            if patch.basis is None or not len(patch.basis):
                raise ValueError('nonempty reference address subspace required')
            dx=to(donor['key_x'])-native['key_x']
            b=to(patch.basis)
            dr=rope(F.linear((dx@b.T)@b,native['key_weight'],None),cos_k,sin_k)
            if kind=='k_random':
                rng=np.random.default_rng(patch.seed)
                random=to(rng.standard_normal(dr.shape))
                dr=random/random.float().norm().clamp_min(1e-20)*dr.float().norm()
            new=keys[anchor_ids]+dr
        keys[anchor_ids] += patch.strength*(new-keys[anchor_ids])
    if kind=='v':
        new=to(donor['v'][head])
        if new.shape!=values[anchor_ids].shape:
            raise ValueError('candidate value correspondence differs')
        values[anchor_ids] += patch.strength*(new-values[anchor_ids])
    scale=native['scale']
    scores=(query@keys.T)*scale
    weights=scores.float().softmax(-1).to(values.dtype)
    context=weights@values
    return context,weights,query,keys[anchor_ids],values[anchor_ids]


@torch.inference_mode()
def replay(model, case, patch=None):
    """case contains ORIGINAL IDs, prompt_length, query positions, source anchors.

    Source anchors are disjoint representative token positions, NOT asserted
    complete facts. Arrays are per query/layer/head. Early and late queries are
    captured together, so prefix effects cannot be confused with future input.
    """
    cfg=model.config; blocks=model.model.layers
    if (cfg.model_type!='llama' or getattr(cfg,'pretraining_tp',1)!=1
            or getattr(cfg,'sliding_window',None)):
        raise ValueError('only ordinary dense Llama/GQA observers are implemented')
    ids=list(map(int,case['ids']));p=int(case['prompt_length'])
    queries=np.asarray(case['queries'],int);anchors=np.asarray(case['anchors'],int)
    if (not len(queries) or len(set(queries))!=len(queries) or np.any(np.diff(queries)<=0)
            or min(queries)<p-1 or max(queries)>=len(ids)
            or len(set(anchors))!=len(anchors) or not np.all((anchors>=0)&(anchors<p))):
        raise ValueError('invalid original query/anchor coordinates')
    L=len(blocks);H=cfg.num_attention_heads;D=cfg.hidden_size;d=D//H;S=len(queries);M=len(anchors)
    if patch and (patch.site not in queries or not 0<=patch.layer<L or not 0<=patch.head<H
                  or not 0<=patch.strength<=1 or patch.kind not in
                  ('q','k','qk','v','q_subspace','q_visible','q_null','q_random','k_subspace','k_random','attn_write','mlp_write')):
        raise ValueError('invalid intervention')
    dev=model.get_input_embeddings().weight.device
    result={name:np.empty((S,L,D),np.float32) for name in ('h','x','attn_write','mlp_write')}
    result.update({name:np.empty((S,L,H,d),np.float32) for name in ('q_raw','q')})
    result.update({name:np.empty((S,L,H,M,d),np.float32) for name in ('k_raw','k','v')})
    result['attention']=np.zeros((S,L,H,max(queries)+1),np.float32)
    result['key_x']=np.empty((L,M,D),np.float32)
    result['query_cos']=np.empty((S,d),np.float32);result['query_sin']=np.empty((S,d),np.float32)
    result['anchor_cos']=np.empty((M,d),np.float32);result['anchor_sin']=np.empty((M,d),np.float32)
    result['read_error']=np.zeros((S,L),np.float32)
    result['post_WO_delta']=np.zeros((S,L,D),np.float32)
    handles=[];point={};byq={int(q):i for i,q in enumerate(queries)}
    asnp=lambda x:x.detach().float().cpu().numpy()
    frame={'start':0,'stop':p}
    def install(li,block):
        attn=block.self_attn;store={};source={};bank={}
        if getattr(attn,'q_norm',None) is not None or getattr(attn,'k_norm',None) is not None:
            raise ValueError('Q/K normalized variants require their own verified observer')
        def block_in(module,args): store['h']=args[0].detach()
        def projection(name):
            def observe(module,args,out):
                store[name]=out.detach()
                if name=='q':store['x']=args[0].detach()
            return observe
        def context(module,args):store['context']=args[0].detach()
        def attention(module,args,kwargs,out):
            bound=inspect.signature(module.forward).bind_partial(*args,**kwargs).arguments
            position=bound.get('position_embeddings',kwargs.get('position_embeddings'))
            qr,kr,vr=[store[n][0].reshape(-1,H if n=='q' else store[n].shape[-1]//d,d)
                      .transpose(0,1) for n in ('q','k','v')]
            if H%kr.shape[0]:raise ValueError('invalid grouped-query mapping')
            if position is None:
                posids=bound.get('position_ids',kwargs.get('position_ids'))
                if not hasattr(module,'rotary_emb') or posids is None:
                    raise ValueError('native RoPE tensors unavailable')
                position=module.rotary_emb(vr[None],posids)
            cos,sin=[a[0] for a in position]
            rotq=rope(qr,cos[None],sin[None]);rotk=rope(kr,cos[None],sin[None])
            rep=H//kr.shape[0]
            kr=kr.repeat_interleave(rep,0);rotk=rotk.repeat_interleave(rep,0);vr=vr.repeat_interleave(rep,0)
            if frame['start']==0:
                source.update(k_raw=kr[:,anchors].clone(),k=rotk[:,anchors].clone(),
                              v=vr[:,anchors].clone(),cos=cos[anchors].clone(),sin=sin[anchors].clone())
                result['key_x'][li]=asnp(store['x'][0,anchors])
                result['anchor_cos']=asnp(cos[anchors]);result['anchor_sin']=asnp(sin[anchors])
            if patch and patch.layer==li and patch.kind not in ('attn_write','mlp_write'):
                bank['k']=rotk if frame['start']==0 else torch.cat((bank['k'],rotk),1)
                bank['v']=vr if frame['start']==0 else torch.cat((bank['v'],vr),1)
            qabs=frame['stop']-1
            altered=out
            if qabs in byq:
                si=byq[qabs]
                result['query_cos'][si]=asnp(cos[-1]);result['query_sin'][si]=asnp(sin[-1])
                if not isinstance(out,tuple) or len(out)<2 or out[1] is None:
                    raise ValueError('eager native attention is required; no attention substitute')
                weights=out[1][0,:,-1].detach()
                result['h'][si,li]=asnp(store['h'][0,-1]);result['x'][si,li]=asnp(store['x'][0,-1])
                result['q_raw'][si,li]=asnp(qr[:,-1]);result['q'][si,li]=asnp(rotq[:,-1])
                for name in ('k_raw','k','v'):result[name][si,li]=asnp(source[name])
                result['attention'][si,li,:,:len(weights[0])]=asnp(weights)
                if patch and patch.layer==li and patch.site==qabs and patch.kind not in ('attn_write','mlp_write'):
                    head=patch.head;scale=getattr(module,'scaling',d**-.5)
                    old=store['context'][0,-1].reshape(H,d)[head]
                    recompute=(rotq[head,-1]@bank['k'][head].T*scale).float().softmax(-1).to(bank['v'].dtype)
                    error=(recompute@bank['v'][head]-old).float().norm()/old.float().norm().clamp_min(1e-9)
                    result['read_error'][si,li]=float(error)
                    if not torch.isfinite(error) or error>.02:
                        raise ValueError(f'head QK replay mismatch L{li}H{head}: {float(error)}')
                    native=dict(q_raw=qr[head,-1],x=store['x'][0,-1],scale=scale,
                                key_x=torch.as_tensor(result['key_x'][li],device=dev,dtype=qr.dtype),
                                key_weight=module.k_proj.weight[(head//rep)*d:(head//rep+1)*d])
                    new,w,newq,newk,newv=changed_head(rotq[head,-1],bank['k'][head],bank['v'][head],
                        anchors,cos[-1],sin[-1],source['cos'],source['sin'],
                        module.q_proj.weight[head*d:(head+1)*d],native,patch)
                    # Difference against reconstructed same-world context avoids injecting rounding drift.
                    diff=new-recompute@bank['v'][head]
                    delta=F.linear(diff,module.o_proj.weight[:,head*d:(head+1)*d],None)
                    edited=first(out).clone();edited[0,-1]+=delta.to(edited)
                    altered=replace_first(out,edited)
                    result['attention'][si,li,head,:len(w)]=asnp(w)
                    result['q'][si,li,head]=asnp(newq);result['k'][si,li,head]=asnp(newk)
                    result['v'][si,li,head]=asnp(newv);result['post_WO_delta'][si,li]=asnp(delta)
            if patch and patch.layer==li and patch.kind=='attn_write' and patch.site==qabs:
                value=first(altered).clone();donor=torch.as_tensor(patch.donor['attn_write'],device=dev,dtype=value.dtype)
                value[0,-1]+=patch.strength*(donor-value[0,-1]);altered=replace_first(altered,value)
            if qabs in byq:result['attn_write'][byq[qabs],li]=asnp(first(altered)[0,-1])
            return altered
        def mlp(module,args,out):
            qabs=frame['stop']-1
            if patch and patch.layer==li and patch.kind=='mlp_write' and patch.site==qabs:
                out=out.clone();donor=torch.as_tensor(patch.donor['mlp_write'],device=dev,dtype=out.dtype)
                out[0,-1]+=patch.strength*(donor-out[0,-1])
            if qabs in byq:result['mlp_write'][byq[qabs],li]=asnp(out[0,-1])
            return out
        def clear(module,args,out):store.clear()
        handles.append(block.register_forward_pre_hook(block_in))
        for n in ('q','k','v'):handles.append(getattr(attn,n+'_proj').register_forward_hook(projection(n)))
        handles.append(attn.o_proj.register_forward_pre_hook(context))
        handles.append(attn.register_forward_hook(attention,with_kwargs=True))
        mlp_module=getattr(block,'mlp',None)
        if mlp_module is None:raise ValueError('decoder block must expose its MLP output')
        handles.append(mlp_module.register_forward_hook(mlp))
        handles.append(block.register_forward_hook(clear))
    logits={};cache=None
    try:
        for li,b in enumerate(blocks):install(li,b)
        seq=torch.tensor([ids],device=dev)
        for t in range(max(queries)-p+2):
            stop=p+t;start=0 if t==0 else stop-1
            frame.update(start=start,stop=stop)
            out=model(input_ids=seq[:,start:stop],past_key_values=cache,use_cache=True,
                      attention_mask=torch.ones_like(seq[:,:stop]),output_attentions=True,
                      output_hidden_states=False,return_dict=True)
            if stop-1 in byq:logits[stop-1]=out.logits[0,-1].float().cpu().numpy()
            cache=out.past_key_values
            del out
        result.update(logits=np.stack([logits[int(q)] for q in queries]),queries=queries,anchors=anchors)
        return result
    finally:
        for handle in handles:handle.remove()
        del cache
