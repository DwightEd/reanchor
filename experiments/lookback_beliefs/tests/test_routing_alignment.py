"""Regression: a copied phrase must be scored in the response, not the prompt."""
import json
import numpy as np
from experiments.lookback_beliefs.routing_data import natural
from experiments.lookback_beliefs.tests.test_routing import Tokenizer


def test_natural_target_present_in_source_is_located_in_response(tmp_path):
    tok=Tokenizer()
    prefix=('Reduce heat to medium and cook another 10 to 12 minutes. '
            'Remove the bratwurst from the beer mixture; reduce heat to low, and continue cooking the onions. '
            'Cook bratwurst on preheated grill for 10 to 14 minutes. Answer:')
    response='Cook the bratwurst on the preheated grill for 10 to 14 minutes.'
    ids=tok.encode(prefix+response);p=len(tok.encode(prefix));n=len(ids)
    folder=tmp_path/'samples';folder.mkdir()
    row=dict(source_id='14375',seed=0,trace='x.npz',response=response)
    (folder/'samples.jsonl').write_text(json.dumps(row)+'\n')
    np.savez(folder/'x.npz',token_ids=ids,prompt_length=p,top_ids=np.ones((n-p,5),int),
             top_logits=np.zeros((n-p,5)),log_normalizer=np.zeros(n-p),attention=np.zeros((4,4,n-p,n)))
    rows,_=natural(tok,folder,select=['cooking_grill_control'])
    for case in rows:
        assert min(case['queries'])>=case['prompt_length']
        assert [tok.decode([case['ids'][q+1]]).strip() for q in case['queries'][:-1]]==['for','10']
        assert case['expected_tokens']==rows[0]['expected_tokens']
