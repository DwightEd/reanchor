"""Small regression tests for semantic targets, patch algebra and the runnable CLI."""
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.lookback_beliefs.data import (Pair, encode_pair, load_pairs, make_pairs,
                                              select_positions, symbolic_example)
from experiments.lookback_beliefs.patching import (capture_states, last_logits, patch_logits,
                                                   subspace_mix)
from experiments.lookback_beliefs.run import parser, run, summarize
from experiments.lookback_beliefs.subspace import dcm_loss, fit_mask, source_split, svd_basis


class Tokenizer:
    """Whitespace tokenizer for software tests; not a claim about Llama tokens."""
    def __init__(self):
        self.vocab = {"<bos>": 0}
    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
        matches = list(re.finditer(r"\w+|[^\w\s]", text))
        tokens, offsets = ([0], [(0, 0)]) if add_special_tokens else ([], [])
        for m in matches:
            self.vocab.setdefault(m.group(), len(self.vocab))
            tokens.append(self.vocab[m.group()]); offsets.append(m.span())
        out = dict(input_ids=tokens, attention_mask=[1] * len(tokens))
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out
    def decode(self, ids):
        reverse = {v: k for k, v in self.vocab.items()}
        return " ".join(reverse.get(int(i), f"v{i}") for i in ids)
    def apply_chat_template(self, messages, **kwargs):
        return "User: " + messages[0]["content"] + "\nAssistant:"


class Block(nn.Module):
    def __init__(self, width, return_tuple=False):
        super().__init__()
        self.q = nn.Linear(width, width); self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width); self.ff = nn.Linear(width, width)
        self.return_tuple = return_tuple
    def forward(self, x):
        n = x.shape[1]
        z = self.q(x) @ self.k(x).transpose(-1, -2) / x.shape[-1] ** .5
        z = z.masked_fill(torch.ones(n, n, dtype=torch.bool, device=x.device).triu(1), -torch.inf)
        h = x + z.softmax(-1) @ self.v(x)
        h = h + torch.tanh(self.ff(h))
        return (h, None) if self.return_tuple else h


class Decoder(nn.Module):
    def __init__(self, width=12):
        super().__init__()
        self.embed = nn.Embedding(256, width)
        self.layers = nn.ModuleList([Block(width), Block(width, True), Block(width)])
        self.norm = nn.LayerNorm(width)
    def forward(self, input_ids, **kwargs):
        h = self.embed(input_ids)
        for block in self.layers:
            out = block(h); h = out if isinstance(out, torch.Tensor) else out[0]
        return SimpleNamespace(last_hidden_state=self.norm(h))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Decoder(); self.lm_head = nn.Linear(12, 256, bias=False)


def fixture():
    torch.manual_seed(7)
    model = Model().eval().requires_grad_(False)
    tokenizer = Tokenizer()
    e = encode_pair(make_pairs(1)[0], tokenizer)
    return model, tokenizer, e


def assert_no_hooks(model):
    assert all(not layer._forward_hooks for layer in model.model.layers)


def test_third_answer_is_distinct_and_generated_from_binding():
    sample = symbolic_example()
    assert len({sample[k] for k in ("base_answer", "donor_answer", "pointer_intervention")}) == 3
    for p in make_pairs(20):
        assert p.hypotheses["payload"] == p.donor_answer
        assert len({p.base_answer, p.donor_answer, p.hypotheses["pointer"]}) == 3
        assert p.hypotheses["pointer"] in p.base_prompt
        assert p.hypotheses["pointer"] not in p.donor_prompt
    assert make_pairs(5, seed=1) == make_pairs(5, seed=1)


def test_binding_swap_preserves_word_values():
    for p in make_pairs(5, experiment="binding"):
        assert p.base_answer == p.donor_answer
        assert p.hypotheses["binding_redirect"] != p.base_answer
        e = encode_pair(p, Tokenizer())
        assert len(e["base_sites"]) == len(e["donor_sites"]) == 2
        assert e["base_sites"] != e["donor_sites"]


def test_char_positions_and_no_hardcoded_prompt_length():
    assert select_positions({"text": "tea", "occurrence": 1}, "tea tea", [(0, 0), (0, 3), (4, 7)]) == [2]
    assert select_positions({"span": [0, 2]}, "tea", [(0, 0), (0, 3)]) == [1]
    with pytest.raises(ValueError):
        select_positions({"text": "beer"}, "tea", [(0, 3)])
    a = encode_pair(make_pairs(1)[0], Tokenizer(), chat_template=True)
    assert a["base_sites"][0] == a["base"]["input_ids"].shape[1] - 1


def test_multi_token_target_rejected():
    p = make_pairs(1)[0]; p.hypotheses["pointer"] = "two words"
    with pytest.raises(ValueError, match="single aligned"):
        encode_pair(p, Tokenizer())


def test_capture_tuple_and_tensor_outputs_and_noop():
    model, _, e = fixture()
    requests = {i: e["base_sites"] for i in range(3)}
    cached, baseline = capture_states(model, e["base"], requests)
    for i in range(3):
        with torch.no_grad():
            z = patch_logits(model, e["base"], i, e["base_sites"], cached[i])
        torch.testing.assert_close(z, baseline, rtol=0, atol=0)
    assert all(v.device.type == "cpu" for v in cached.values())
    assert_no_hooks(model)


def test_final_layer_patch_equals_donor_readout():
    model, _, e = fixture()
    cached, donor_logits = capture_states(model, e["donor"], {2: e["donor_sites"]})
    with torch.no_grad():
        z = patch_logits(model, e["base"], 2, e["base_sites"], cached[2])
    torch.testing.assert_close(z, donor_logits)


def test_hook_cleanup_after_exception():
    model, _, e = fixture()
    with pytest.raises(ValueError):
        patch_logits(model, e["base"], 0, e["base_sites"], torch.zeros(1, 9, 12))
    assert_no_hooks(model)
    with pytest.raises(IndexError):
        capture_states(model, e["base"], {0: [99999]})
    assert_no_hooks(model)


@pytest.mark.parametrize("mask_value", [0., .5, 1.])
def test_factorized_mask_equals_official_dense_operator(mask_value):
    base, donor = torch.randn(1, 2, 8), torch.randn(1, 2, 8)
    basis = torch.linalg.qr(torch.randn(8, 4))[0].T
    mask = torch.full((4,), mask_value, requires_grad=True)
    masked = basis * mask[:, None]
    expected = base + (donor - base) @ (masked.T @ masked)
    actual = subspace_mix(base, donor, basis, mask)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward(); assert mask.grad is not None


def test_real_suffix_gradient_only_updates_mask():
    model, _, e = fixture()
    donor, _ = capture_states(model, e["donor"], {0: e["donor_sites"]})
    basis = torch.eye(12); mask = torch.nn.Parameter(torch.ones(12))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    z = patch_logits(model, e["base"], 0, e["base_sites"], donor[0], basis, mask)
    loss = dcm_loss(z, e["candidate_ids"][0], mask, .1); loss.backward()
    assert mask.grad is not None and torch.isfinite(mask.grad).all() and mask.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    assert_no_hooks(model)


def test_restoring_baseline_final_state_cancels_earlier_patch():
    model, _, e = fixture()
    base, baseline = capture_states(model, e["base"], {2: e["base_sites"]})
    donor, _ = capture_states(model, e["donor"], {0: e["donor_sites"]})
    with torch.no_grad():
        restored = patch_logits(model, e["base"], 0, e["base_sites"], donor[0],
                                restore={2: (e["base_sites"], base[2])})
    torch.testing.assert_close(restored, baseline)
    with pytest.raises(ValueError, match="overlaps"):
        patch_logits(model, e["base"], 2, e["base_sites"], base[2], restore={2: (e["base_sites"], base[2])})


def test_source_disjoint_and_local_basis():
    pairs = make_pairs(10)
    pairs[1].source_id = pairs[0].source_id
    split = source_split(pairs)
    train = {p.source_id for p in pairs if split[p.id] == "train"}
    validation = {p.source_id for p in pairs if split[p.id] == "validation"}
    assert not train & validation and train and validation
    basis = svd_basis([torch.randn(1, 3, 12)], 32)
    assert basis.shape == (3, 12)
    torch.testing.assert_close(basis @ basis.T, torch.eye(3), atol=1e-5, rtol=1e-5)


def test_target_logit_objective_not_cross_entropy():
    z = torch.tensor([[2., 3., 7.]])
    assert dcm_loss(z, 1, torch.tensor([1., .5]), .2).item() == pytest.approx(-2.7)


def args_for(tmp_path, mode="scan"):
    return parser().parse_args(["--mode", mode, "--samples", "4", "--output", str(tmp_path / mode),
                               "--layers", "0", "2", "--device", "cpu", "--cpu-threads", "1"])


def test_full_scan_outputs_and_resume(tmp_path):
    model, tokenizer, _ = fixture()
    args = args_for(tmp_path)
    result = run(args, model, tokenizer)
    assert result["examples"] == 4 and len(result["curves"]) == 4
    assert (tmp_path / "scan/complete.json").exists()
    with np.load(tmp_path / "scan/samples/answer_0000.npz", allow_pickle=False) as saved:
        assert saved["patch_logits"].shape[0] == 2
    args.resume = True
    assert run(args, model, tokenizer) == result
    args.samples = 5
    with pytest.raises(ValueError):
        run(args, model, tokenizer)


def test_dcm_pipeline_source_holdout_and_serialized_mask(tmp_path):
    model, tokenizer, _ = fixture()
    args = args_for(tmp_path, "dcm"); args.layer = 0; args.rank = 3; args.include_errors = True
    result = run(args, model, tokenizer)
    assert result["train_examples"] == 3 and result["validation_examples"] == 1
    assert result["effective_basis_rank"] <= 3
    arms = {r["arm"] for r in result["curves"]}
    assert arms == {"full_state", "dcm", "zero", "rank_matched_random"}
    with np.load(tmp_path / "dcm/subspace.npz", allow_pickle=False) as data:
        assert np.isin(data["binary_mask"], [0, 1]).all()
        assert data["binary_mask"].sum() == data["random_rank_matched_mask"].sum()
    assert all(p.grad is None for p in model.parameters())


def test_iia_uses_intervention_answer_and_reports_denominator():
    rows = [dict(hypotheses={"pointer": "beer", "payload": "tea"}, baseline={"both_correct": False},
                 measurements=[dict(layer=3, arm="full_state", matches={"beer": True, "tea": False})])]
    metrics = summarize(rows)
    pointer = next(c for c in metrics["curves"] if c["hypothesis"] == "pointer")
    assert pointer["iia_all"] == 1 and pointer["n_both_correct"] == 0 and pointer["iia_both_correct"] is None


def test_custom_json_roundtrip_and_sites(tmp_path):
    p = make_pairs(1, experiment="binding")[0]
    path = tmp_path / "pairs.jsonl"; path.write_text(json.dumps(p.to_dict()))
    assert load_pairs(path) == [p]
    p.sites *= 2
    with pytest.raises(ValueError, match="overlapping"):
        encode_pair(p, Tokenizer())


def test_optional_transformers_tiny_llama():
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(vocab_size=256, hidden_size=32, intermediate_size=64,
              num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    config._attn_implementation = "sdpa"
    model = transformers.LlamaForCausalLM(config).eval().requires_grad_(False)
    e = encode_pair(make_pairs(1)[0], Tokenizer())
    donor, expected = capture_states(model, e["donor"], {1: e["donor_sites"]})
    with torch.no_grad():
        actual = patch_logits(model, e["base"], 1, e["base_sites"], donor[1])
    torch.testing.assert_close(actual, expected)
