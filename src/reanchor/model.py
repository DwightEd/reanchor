"""Hugging Face boundary for conditional continuation scoring."""

from __future__ import annotations

from pathlib import Path


def encode_scoring_pair(tokenizer, context: str, continuation: str):
    """Tokenize joined text and locate continuation tokens by character offsets."""

    encoded = tokenizer(
        context + continuation,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded["input_ids"])
    boundary = len(context)
    target_start = next(
        (index for index, (_start, end) in enumerate(encoded["offset_mapping"]) if end > boundary),
        None,
    )
    if target_start is None:
        raise ValueError("candidate continuation must contain at least one token")

    targets = input_ids[target_start:]
    bos_id = tokenizer.bos_token_id
    add_bos = bos_id is not None and (not input_ids or input_ids[0] != bos_id)
    if add_bos:
        input_ids = [bos_id, *input_ids]
        target_start += 1
    return input_ids, target_start, targets


class CausalLMScorer:
    """Score short candidate continuations with one batched forward pass."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str,
        dtype: str,
        system_prompt: str,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtypes = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in dtypes:
            raise ValueError(f"dtype must be one of {tuple(dtypes)}")

        self.torch = torch
        self.device = torch.device(device)
        self.model_path = model_path.resolve()
        self.dtype = dtype
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if not self.tokenizer.is_fast:
            raise ValueError("continuation scoring requires a fast tokenizer with offset mappings")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtypes[dtype],
            local_files_only=True,
        ).to(self.device)
        self.model.eval()

    @property
    def metadata(self) -> dict[str, str]:
        revision = getattr(self.model.config, "_commit_hash", None) or "local-unresolved"
        return {
            "model": str(self.model_path),
            "revision": revision,
            "dtype": self.dtype,
            "device": str(self.device),
        }

    def render_prompt(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"{self.system_prompt}\n\n{prompt}\n\nAssistant:"

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return mean conditional log probability for each context/continuation pair."""

        sequences = []
        boundaries = []
        for context, continuation in pairs:
            sequence, start, continuation_ids = encode_scoring_pair(
                self.tokenizer, context, continuation
            )
            sequences.append(sequence)
            boundaries.append((start, continuation_ids))

        batch = self.tokenizer.pad(
            {"input_ids": sequences},
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            logits = self.model(**batch, use_cache=False).logits

        scores = []
        for row, (start, continuation_ids) in enumerate(boundaries):
            positions = logits[row, start - 1 : start + len(continuation_ids) - 1].float()
            targets = self.torch.tensor(continuation_ids, device=self.device).unsqueeze(1)
            token_scores = positions.log_softmax(dim=-1).gather(1, targets)
            scores.append(float(token_scores.mean().item()))
        return scores
