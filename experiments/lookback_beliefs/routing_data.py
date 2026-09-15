"""Paired selectors/addresses and the original cooking/clothing decision windows.

Constructed counterfactuals are labelled as such. No new response is generated.
The onion step has NO known unique correct number; we never relabel 1-2/10-14
minutes as its ground truth. Natural cases are diagnosis, not a benchmark.
"""
import json
from pathlib import Path
import numpy as np

DEFAULT_SAMPLES='outputs/samples_20260911_145421_235'


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def token_span(tokenizer,text,quote,*,last=False):
    starts=[];i=text.find(quote)
    while i>=0:starts.append(i);i=text.find(quote,i+1)
    if not starts or (len(starts)!=1 and not last):
        raise ValueError(f'missing/ambiguous quote {quote!r} ({len(starts)} matches)')
    start=starts[-1] if last else starts[0];end=start+len(quote)
    enc=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    span=[i for i,(a,b) in enumerate(enc['offset_mapping']) if a<end and b>start and b>a]
    if not span:raise ValueError('empty token alignment')
    return enc['input_ids'],span


def controlled(tokenizer,groups=16,filler_records=8):
    """Eight worlds/group: selector C x condition-to-value binding B x order O.

    Value identities fixed under B, record order is independently varied. Thus
    query-only (C flip), address-only (B flip) and both-flip have distinct expected
    choices. Content donor uses two NEW values, giving a third-answer control.
    No filtering on whether the model happened to answer correctly.
    """
    if groups<4 or groups%2 or filler_records<0:raise ValueError('even groups>=4 and nonnegative filler required')
    words=['amber','violet','copper','silver','green','orange','black','white','blue','red','yellow','brown']
    values=[]
    for word in words:
        token=tokenizer.encode(' '+word,add_special_tokens=False)
        if len(token)==1:values.append((word,token[0]))
    if len(values)<8:raise ValueError('need eight distinct single-token value strings under this tokenizer')
    if len({t for _,t in values})!=len(values):raise ValueError('value token collision')
    records=[]
    for g in range(groups):
        split='reference' if g<groups//2 else 'heldout'
        bank=values[:4] if split=='reference' else values[4:8]
        roll=g%4;bank=bank[roll:]+bank[:roll]
        subject=f'item-{g:03d}';group=f'controlled_{g:03d}'
        for content in (0,1):
            states=bank[2:] if content else bank[:2]
            worlds=[(c,b,o) for c in (0,1) for b in (0,1) for o in (0,1)] if not content else [(0,0,0)]
            for c,b,o in worlds:
                phase=['early','late']
                lines=[f'Record {chr(65+j)}: {subject}; phase: {phase[j^b]}; value: {states[j][0]}.' for j in (0,1)]
                if o:lines.reverse()
                filler='\n'.join(f'Record: distractor-{i}; phase: {phase[i%2]}; value: {states[i%2][0]}.'
                                 for i in range(filler_records))
                user=('Use only the supplied records. Match BOTH the item and its phase; do not use the last '
                      'mentioned value merely because it is recent. Complete the answer with the recorded value.\n'+
                      '\n'.join(lines)+'\n'+filler+
                      f'\nQuestion: What is the value of {subject} in the {phase[c]} phase?')
                prefix=tokenizer.apply_chat_template([dict(role='user',content=user)],tokenize=False,add_generation_prompt=True)
                # A fixed answer prefix makes the next candidate the same leading-space
                # token as the source value, rather than conflating spelling/format gating.
                prefix += 'The requested value is'
                # Each anchor is the single value token, not its punctuation.
                ids=tokenizer.encode(prefix,add_special_tokens=False);anchors=[]
                for j in (0,1):
                    line=f'Record {chr(65+j)}: {subject}; phase: {phase[j^b]}; value: {states[j][0]}'
                    _,span=token_span(tokenizer,prefix,line)
                    if ids[span[-1]]!=states[j][1]:
                        raise ValueError('source and answer value tokens differ; no guessed token alignment')
                    anchors.append(span[-1])
                records.append(dict(id=f'{group}_c{c}b{b}o{o}v{content}',group=group,split=split,
                    kind='constructed',c=c,b=b,o=o,content=content,ids=ids,prompt_length=len(ids),
                    queries=[len(ids)-1],anchors=anchors,candidate_ids=[x[1] for x in states],
                    correct_index=c^b,expected_tokens=[states[c^b][1]],selector=c,
                    input_text=prefix,site_names=['fact_choice'],semantic_supervision='constructed selector and relation only'))
    # Same binding and record order, different question wording. Never select a
    # donor because it answered correctly; correctness is reported afterwards.
    extra=[]
    for base in records:
        if base['c']==base['b']==base['o']==base['content']==0:
            text=base['input_text']; subject=f"item-{int(base['group'].rsplit('_',1)[1]):03d}"
            old=f'What is the value of {subject} in the early phase?'
            new=f'For the early phase of {subject}, which value is recorded?'
            if text.count(old)!=1:raise ValueError('controlled paraphrase anchor missing')
            changed=text.replace(old,new); ids=tokenizer.encode(changed,add_special_tokens=False)
            extra.append({**base,'id':base['id']+'_same','style':1,'input_text':changed,
                          'ids':ids,'prompt_length':len(ids),'queries':[len(ids)-1]})
    return records+extra


SPECS=[
 dict(id='cooking_onion',source_id='14375',seed=0,target='for 10 to 12 minutes',
      sites=['for','10','12'],anchors=['10 to 12 minutes','10 to 14 minutes'],anchor_words=['12','14'],
      status='unsupported_stage_duration',correct_index=None,family='cooking'),
 dict(id='cooking_grill_control',source_id='14375',seed=0,target='for 10 to 14 minutes',
      sites=['for','10','14'],anchors=['10 to 12 minutes','10 to 14 minutes'],anchor_words=['12','14'],
      status='supported_local_control_not_whole_answer',correct_index=1,family='cooking'),
 dict(id='headdress',source_id='14315',seed=2,
      target='and wore a headdress with a special fringe of gold and feathers.',
      sites=['and','headdress','gold'],anchors=['headdress with his special fringe of gold and feathers','large metal pins'],anchor_words=['headdress','pins'],
      status='exclusive_scope_missing',correct_index=None,family='headdress'),
 dict(id='clothes_lengths_scope_control',source_id='14315',seed=3,
      target='with tunic lengths reaching up to the knee, and skirts that covered up to the ankle.',
      sites=['with','knee','ankle'],anchors=['their knee','up to ankle'],anchor_words=['knee','ankle'],
      status='scope_ambiguous_not_gold_error',correct_index=None,family='clothing')]


def altered_prompt(text,family,mode):
    """Explicitly declared meaning-changing and meaning-preserving comparisons.

    These interventions do not establish that an original hidden pointer is
    correct. They only make the source constraint differ while payload words
    and the original response prefix stay fixed.
    """
    if family=='cooking':
        if mode=='constraint_flip':
            edits=[('Reduce heat to medium and cook another 10 to 12 minutes.',
                    'Reduce heat to medium and continue cooking.'),
                   ('reduce heat to low, and continue cooking the onions.',
                    'reduce heat to low, and cook the onions for 10 to 12 minutes.')]
        else:
            edits=[('reduce heat to low, and continue cooking the onions.',
                    'reduce the heat to low and continue to cook the onions.')]
    elif family=='headdress':
        edits=[('Only the Inca could wear',
                'All the Incas could wear' if mode=='constraint_flip' else 'The Inca alone could wear')]
    else:
        edits=([('Inca men wore simple tunics.','Inca women wore simple tunics.'),
                ('The women wore skirt.','The men wore skirt.')] if mode=='constraint_flip'
               else [('The women wore skirt.','The women wore skirts.')])
    for old,new in edits:
        if text.count(old)!=1:raise ValueError(f'original source edit missing/ambiguous: {old}')
        text=text.replace(old,new,1)
    return text,edits


def natural(tokenizer,samples,states=None,select=None):
    """Use exact original tokens. Changed prompts are separate donor worlds."""
    samples=Path(samples);rows=read_jsonl(samples/'samples.jsonl');cases=[];checks={}
    specs=[s for s in SPECS if not select or s['id'] in select]
    if select and set(select)!={s['id'] for s in specs}:raise ValueError('unknown selected natural case')
    for spec in specs:
        found=[r for r in rows if str(r['source_id'])==spec['source_id'] and r['seed']==spec['seed']]
        if len(found)!=1:raise ValueError('natural source/seed identity is not unique')
        row=found[0];name=row['trace']
        if Path(name).name!=name:raise ValueError('trace must be a local filename')
        with np.load(samples/name,allow_pickle=False) as f:
            original=f['token_ids'].astype(int);p=int(f['prompt_length'])
            prefix=tokenizer.decode(original[:p],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            response=tokenizer.decode(original[p:],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            if tokenizer.encode(prefix,add_special_tokens=False)!=original[:p].tolist():
                raise ValueError('original prompt does not round-trip to exact saved IDs')
            full=prefix+response
            encoded=tokenizer(full,add_special_tokens=False,return_offsets_mapping=True)
            if encoded['input_ids']!=original.tolist():raise ValueError('original full tokens differ')
            begin=response.find(spec['target'])
            if begin<0 or response.count(spec['target'])!=1:raise ValueError('target missing/ambiguous in ORIGINAL response')
            offsets=np.asarray(encoded['offset_mapping']);queries=[]
            for word in spec['sites']:
                start=full.index(spec['target'])+spec['target'].index(word);end=start+len(word)
                hit=np.flatnonzero((offsets[:,0]<end)&(offsets[:,1]>start)&(offsets[:,1]>offsets[:,0]))
                queries.append(int(hit[0]-1))
            steps=np.asarray(queries)-p+1
            attention=f['attention'][:,:,steps,:max(queries)+1].transpose(2,0,1,3).copy()
            checks[spec['id']]=dict(top_ids=f['top_ids'][steps].copy(),top_logits=f['top_logits'][steps].copy(),
                log_normalizer=f['log_normalizer'][steps].copy(),attention=attention)
        if states:
            with np.load(Path(states)/name,allow_pickle=False) as f:
                if not np.array_equal(f['token_ids'],original):raise ValueError('saved state token identity mismatch')
        for mode in ('original','constraint_flip','paraphrase_control'):
            changed,edits=(prefix,[]) if mode=='original' else altered_prompt(prefix,spec['family'],mode)
            newids=tokenizer.encode(changed,add_special_tokens=False);newp=len(newids)
            anchors=[]
            for quote,word in zip(spec['anchors'],spec['anchor_words']):
                if changed.count(quote)!=1 or quote.count(word)!=1:raise ValueError('source anchor must be unique')
                left=changed.index(quote)+quote.index(word);right=left+len(word)
                mapping=tokenizer(changed,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
                span=[i for i,(a,b) in enumerate(mapping) if a<right and b>left and b>a]
                if not span:raise ValueError('empty role-aligned anchor')
                anchors.append(span[-1])
            # Preserve every originally sampled response token; only prompt donors differ.
            ids=newids+original[p:].tolist();newq=(np.asarray(queries)-p+newp).tolist()
            cases.append(dict(id=spec['id']+'__'+mode,group=spec['id'],split='natural',kind='natural',
                world=mode,source_id=spec['source_id'],seed=spec['seed'],trace=name,
                ids=ids[:max(newq)+1],prompt_length=newp,queries=newq,anchors=anchors,
                expected_tokens=original[np.asarray(queries)+1].tolist(),candidate_ids=[ids[a] for a in anchors],
                correct_index=None,case_status=spec['status'],input_text=changed,
                original_response=row['response'],site_names=['commitment','value_entry','value_completion'],
                edits=edits,anchor_quotes=spec['anchors'],anchor_words=spec['anchor_words'],
                local_claim_supported=(None if spec['family']=='clothing' else True if spec['id']=='cooking_grill_control' or mode=='constraint_flip' else False),
                support_basis='case-specific source/claim interpretation; counterfactual permission is constructed',
                semantic_supervision='manual case/quote diagnosis, not official hallucination annotations'))
    return cases,checks
