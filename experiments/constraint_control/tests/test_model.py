from types import SimpleNamespace

import numpy as np
import torch

from experiments.constraint_control.generation import SamplingConfig
from experiments.constraint_control.model import HuggingFaceBackend
from experiments.constraint_control.records import ChatMessage


class TinyTokenizer:
    eos_token_id = 4

    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{"role": "user", "content": "Question"}]
        return torch.tensor([[1, 2]], dtype=torch.long)

    def convert_ids_to_tokens(self, token_ids):
        return [f"token-{token_id}" for token_id in token_ids]

    def decode(self, token_ids, **kwargs):
        return " ".join(str(token_id) for token_id in token_ids)


class TinyCausalModel:
    def __init__(self, floor=-4.0):
        self.floor = floor

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
            state = torch.arange(length * 3, dtype=torch.float32).reshape(1, length, 3)
            hidden_states = (state, state + 1)
        if output_attentions:
            attentions = (torch.ones((1, 1, length, length), dtype=torch.float32),)
        return SimpleNamespace(
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
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
    assert capture.residual_states.shape == (2, 2, 3)
    assert capture.attention_weights.shape == (1, 1, 2, 4)
    np.testing.assert_array_equal(capture.top_token_ids[:, 0], [3, 4])


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
