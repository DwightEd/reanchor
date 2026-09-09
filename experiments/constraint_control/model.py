"""Hugging Face model execution for free sampling and exact replay."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from .generation import CapturedGeneration, SamplingConfig
from .records import ChatMessage


class HuggingFaceBackend:
    """Sample one trajectory and replay it with inspectable eager attention."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        dtype: str = "auto",
        revision: str | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ):
        if (model is None) != (tokenizer is None):
            raise ValueError("model and tokenizer must be supplied together")
        self.model_name = model_name
        self.revision = revision or "main"
        self.device = torch.device(device)
        self.dtype = dtype
        if model is None:
            model, tokenizer = self._load_model(model_name, dtype, revision)
        self.model = model.eval()
        self.tokenizer = tokenizer

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "model": self.model_name,
            "revision": self.revision,
            "dtype": self.dtype,
            "device": str(self.device),
            "attention_implementation": "eager",
        }

    @torch.inference_mode()
    def sample_and_replay(
        self,
        messages: tuple[ChatMessage, ...],
        sampling: SamplingConfig,
    ) -> CapturedGeneration:
        prompt_ids = self._tokenize(messages)
        sequence = prompt_ids
        raw_logits = []
        selected_logits = []
        model_logprobs = []
        sampling_logprobs = []
        entropies = []
        top_ids = []
        top_logits = []
        generator = torch.Generator(device=self.device)
        generator.manual_seed(sampling.seed)
        eos_ids = self._eos_ids()
        stop_reason = "max_new_tokens"

        for _ in range(sampling.max_new_tokens):
            output = self.model(input_ids=sequence, use_cache=False, return_dict=True)
            logits = output.logits[0, -1].float()
            log_probability = logits.log_softmax(dim=-1)
            probability = log_probability.exp()
            sampling_probability = self._sampling_distribution(logits, sampling)
            token_id = int(
                torch.multinomial(sampling_probability, 1, generator=generator).item()
            )
            trace_k = min(sampling.trace_top_k, logits.numel())
            values, indices = logits.topk(trace_k)

            raw_logits.append(logits.cpu())
            selected_logits.append(float(logits[token_id]))
            model_logprobs.append(float(log_probability[token_id]))
            sampling_logprobs.append(float(sampling_probability[token_id].log()))
            entropies.append(float(-(probability * log_probability).sum()))
            top_ids.append(indices.cpu())
            top_logits.append(values.cpu())
            next_token = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
            sequence = torch.cat((sequence, next_token), dim=1)
            if token_id in eos_ids:
                stop_reason = "eos"
                break

        response_start = prompt_ids.shape[1]
        response_ids = sequence[0, response_start:]
        replay = self.model(
            input_ids=sequence,
            use_cache=False,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        if replay.hidden_states is None or replay.attentions is None:
            raise ValueError("model replay did not return hidden states and eager attention")

        prediction_rows = torch.arange(
            response_start - 1,
            sequence.shape[1] - 1,
            device=self.device,
        )
        replay_logits = replay.logits[0, prediction_rows].float().cpu()
        generation_logits = torch.stack(raw_logits)
        replay_error = float((generation_logits - replay_logits).abs().max())
        replay_selected = replay_logits.gather(1, response_ids.cpu()[:, None]).squeeze(1)
        residual_states = torch.stack(
            [state[0, prediction_rows] for state in replay.hidden_states]
        )
        attention_weights = torch.stack(
            [attention[0, :, prediction_rows, :] for attention in replay.attentions]
        )
        token_ids = sequence[0].cpu().numpy().astype(np.int64, copy=False)
        pieces = self.tokenizer.convert_ids_to_tokens(token_ids.tolist())

        return CapturedGeneration(
            token_ids=token_ids,
            token_text=tuple(str(piece) for piece in pieces),
            response_start=response_start,
            response_text=self.tokenizer.decode(response_ids.tolist(), skip_special_tokens=True),
            generation_selected_logits=np.asarray(selected_logits, dtype=np.float32),
            replay_selected_logits=replay_selected.numpy().astype(np.float32, copy=False),
            model_logprobs=np.asarray(model_logprobs, dtype=np.float32),
            sampling_logprobs=np.asarray(sampling_logprobs, dtype=np.float32),
            entropy=np.asarray(entropies, dtype=np.float32),
            top_token_ids=torch.stack(top_ids).numpy().astype(np.int64, copy=False),
            top_logits=torch.stack(top_logits).numpy().astype(np.float32, copy=False),
            residual_states=residual_states.cpu().numpy().astype(np.float16, copy=False),
            attention_weights=attention_weights.cpu().numpy().astype(np.float16, copy=False),
            replay_max_abs_logit_error=replay_error,
            stop_reason=stop_reason,
        )

    def _load_model(self, model_name: str, dtype: str, revision: str | None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_value: str | torch.dtype
        if dtype == "auto":
            dtype_value = "auto"
        else:
            try:
                dtype_value = getattr(torch, dtype)
            except AttributeError as error:
                raise ValueError(f"unsupported torch dtype: {dtype}") from error
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=dtype_value,
            attn_implementation="eager",
        ).to(self.device)
        return model, tokenizer

    def _tokenize(self, messages: tuple[ChatMessage, ...]) -> torch.Tensor:
        values = [asdict(message) for message in messages]
        token_ids = self.tokenizer.apply_chat_template(
            values,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        if isinstance(token_ids, dict):
            token_ids = token_ids["input_ids"]
        return token_ids.to(self.device)

    def _eos_ids(self) -> set[int]:
        eos = self.tokenizer.eos_token_id
        if eos is None:
            return set()
        if isinstance(eos, int):
            return {eos}
        return {int(token_id) for token_id in eos}

    def _sampling_distribution(
        self, logits: torch.Tensor, sampling: SamplingConfig
    ) -> torch.Tensor:
        scores = logits / sampling.temperature
        if sampling.top_k:
            keep = min(sampling.top_k, scores.numel())
            threshold = scores.topk(keep).values[-1]
            scores = scores.masked_fill(scores < threshold, -torch.inf)
        if sampling.top_p < 1:
            sorted_scores, sorted_indices = scores.sort(descending=True)
            cumulative = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
            remove = cumulative > sampling.top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            scores = scores.clone()
            scores[sorted_indices[remove]] = -torch.inf
        return scores.softmax(dim=-1)
