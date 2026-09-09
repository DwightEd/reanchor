from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from experiments.constraint_control.generation import SamplingConfig
from experiments.constraint_control.model import HuggingFaceBackend
from experiments.constraint_control.records import ChatMessage


class TinyTokenizer:
    eos_token_id = 4
    all_special_ids = [4]

    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{"role": "user", "content": "Question"}]
        return torch.tensor([[1, 2]], dtype=torch.long)

    def convert_ids_to_tokens(self, token_ids):
        return [f"token-{token_id}" for token_id in token_ids]

    def decode(self, token_ids, **kwargs):
        return " ".join(str(token_id) for token_id in token_ids)


class TinyCausalModel:
    config = SimpleNamespace(_commit_hash="resolved-hash")

    def __init__(self, floor=-4.0, state_dtype=torch.float32):
        self.floor = floor
        self.state_dtype = state_dtype

    def eval(self):
        return self

    def __call__(
        self,
        *,
        input_ids,
        output_hidden_states=False,
        output_attentions=False,
        **kwargs,
    ):
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 5), self.floor)
        next_by_token = {2: 3, 3: 4}
        for position, token_id in enumerate(input_ids[0].tolist()):
            logits[0, position, next_by_token.get(token_id, 0)] = 4.0
        hidden_states = None
        attentions = None
        if output_hidden_states:
            state = torch.arange(length * 3, dtype=self.state_dtype).reshape(1, length, 3)
            hidden_states = (state, state + 1)
        if output_attentions:
            attentions = (torch.ones((1, 1, length, length), dtype=self.state_dtype),)
        return SimpleNamespace(
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )


class ShapeSensitiveCausalModel(TinyCausalModel):
    """Emulate low-precision drift when the forward matrix shape changes."""

    def __call__(self, *, input_ids, **kwargs):
        output = super().__call__(input_ids=input_ids, **kwargs)
        output.logits += input_ids.shape[1] * 0.25
        return output


def test_huggingface_backend_explains_how_to_access_a_gated_model(monkeypatch):
    def gated_repository(*_args, **_kwargs):
        raise OSError(
            "401 Client Error. Cannot access gated repo for "
            "meta-llama/Llama-3.1-8B-Instruct"
        )

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", gated_repository)

    with pytest.raises(RuntimeError, match=r"hf auth login.*local model path"):
        HuggingFaceBackend(
            "meta-llama/Llama-3.1-8B-Instruct",
            device="cpu",
        )


def test_huggingface_backend_samples_then_replays_the_exact_tokens():
    backend = HuggingFaceBackend(
        "tiny-model",
        device="cpu",
        model=TinyCausalModel(),
        tokenizer=TinyTokenizer(),
        revision="fixture",
    )

    capture = backend.sample_and_replay(
        (ChatMessage(role="user", content="Question"),),
        SamplingConfig(seed=3, max_new_tokens=4, top_k=1, trace_top_k=2),
    )

    np.testing.assert_array_equal(capture.token_ids, [1, 2, 3, 4])
    assert capture.response_start == 2
    assert capture.response_text == "3 4"
    assert capture.stop_reason == "eos"
    assert capture.replay_max_abs_logit_error == 0.0
    np.testing.assert_array_equal(capture.special_mask, [False, False, False, True])
    assert capture.residual_states.shape == (2, 2, 3)
    assert capture.attention_weights.shape == (1, 1, 2, 4)
    np.testing.assert_array_equal(capture.top_token_ids[:, 0], [3, 4])


def test_huggingface_backend_replays_each_generation_prefix_at_the_same_shape():
    backend = HuggingFaceBackend(
        "shape-sensitive-model",
        device="cpu",
        model=ShapeSensitiveCausalModel(),
        tokenizer=TinyTokenizer(),
    )

    capture = backend.sample_and_replay(
        (ChatMessage(role="user", content="Question"),),
        SamplingConfig(seed=3, max_new_tokens=4, top_k=1, trace_top_k=2),
    )

    assert capture.replay_max_abs_logit_error == 0.0
    np.testing.assert_array_equal(
        capture.attention_weights,
        [[[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]]]],
    )


def test_huggingface_backend_reports_finite_entropy_for_underflowed_probabilities():
    backend = HuggingFaceBackend(
        "tiny-model",
        device="cpu",
        model=TinyCausalModel(floor=-1000.0),
        tokenizer=TinyTokenizer(),
    )

    capture = backend.sample_and_replay(
        (ChatMessage(role="user", content="Question"),),
        SamplingConfig(max_new_tokens=1, top_k=1, trace_top_k=2),
    )

    assert np.isfinite(capture.entropy).all()


def test_huggingface_backend_serializes_bfloat16_replay_tensors():
    backend = HuggingFaceBackend(
        "tiny-model",
        device="cpu",
        model=TinyCausalModel(state_dtype=torch.bfloat16),
        tokenizer=TinyTokenizer(),
    )

    capture = backend.sample_and_replay(
        (ChatMessage(role="user", content="Question"),),
        SamplingConfig(max_new_tokens=1, top_k=1, trace_top_k=2),
    )

    assert capture.residual_states.dtype == np.float16
    assert capture.attention_weights.dtype == np.float16


def test_huggingface_backend_records_the_resolved_model_revision():
    backend = HuggingFaceBackend(
        "tiny-model",
        device="cpu",
        model=TinyCausalModel(),
        tokenizer=TinyTokenizer(),
    )

    assert backend.metadata["revision"] == "resolved-hash"


def test_huggingface_backend_replay_matches_a_real_transformer_forward():
    config = LlamaConfig(
        vocab_size=5,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        bos_token_id=1,
        eos_token_id=4,
        attention_dropout=0.0,
        attn_implementation="eager",
    )
    backend = HuggingFaceBackend(
        "random-tiny-llama",
        device="cpu",
        model=LlamaForCausalLM(config).eval(),
        tokenizer=TinyTokenizer(),
        revision="test-weights",
    )

    capture = backend.sample_and_replay(
        (ChatMessage(role="user", content="Question"),),
        SamplingConfig(seed=5, max_new_tokens=2, top_k=1, trace_top_k=2),
    )

    assert capture.replay_max_abs_logit_error < 1e-6
    assert capture.residual_states.shape[0] == 2
    assert capture.attention_weights.shape[:2] == (1, 2)
