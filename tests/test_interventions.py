import numpy as np
import torch

from decoding.interventions import attention_mask, local_to_remote


def test_edge_cut_only_changes_answer_queries_and_never_opens_future_edges():
    mask = attention_mask(8, 4, blocked=[1, 2])[0, 0].numpy()
    assert mask[2, 1] == 0
    assert mask[3, 1] < -1e20
    assert mask[7, 2] < -1e20
    assert mask[3, 3] == 0
    assert np.all(mask[np.triu_indices(8, 1)] < -1e20)
    local = attention_mask(8, 4, window=2, sinks=1)[0, 0].numpy()
    assert local[7, 0] == 0
    assert local[7, 1] < -1e20
    assert local[7, 5] < -1e20
    assert local[7, 6] == 0


def test_directional_revisit_distinguishes_local_to_source_from_the_reverse():
    weights = np.zeros((1, 1, 3, 10))
    weights[0, 0, 0, [1, 5]] = [0.2, 0.8]
    weights[0, 0, 1, [1, 5]] = [0.8, 0.2]
    weights[0, 0, 2, [1, 5]] = [0.2, 0.8]
    source = np.array([False, True, False, False])
    score = local_to_remote(weights, np.array([7, 8, 9]), source, window=4)
    np.testing.assert_allclose(score, [0, 0.6, 0])


def test_forward_readout_and_sham_mlp_patch_preserve_native_logits():
    from transformers import LlamaConfig, LlamaForCausalLM

    from decoding.interventions import CausalReadout

    torch.manual_seed(4)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    ).eval()
    model.set_attn_implementation("eager")
    reader = CausalReadout(model)
    baseline = reader.run([1, 2, 3, 4, 5], [3, 4], [[6, 7], [8, 9]])
    sham = reader.run([1, 2, 3, 4, 5], [3, 4], [[6, 7], [8, 9]], mlp_patch=baseline["mlp_updates"])
    np.testing.assert_array_equal(baseline["logits"], sham["logits"])
    blocked = reader.run(
        [1, 2, 3, 4, 5], [3, 4], [[6, 7], [8, 9]], mask=attention_mask(5, 4, blocked=[1])
    )
    assert np.all(blocked["attention"][..., 1] == 0)
    assert baseline["layer_contrast"].shape == (3, 3, 2)
