"""Capture missing internal states on the exact saved token sequence."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from decoding.io import read_jsonl, write_json
from decoding.routes import source_positions


def trace_signature(trace: dict) -> str:
    digest = hashlib.sha256()
    for name in (
        "token_ids",
        "prompt_length",
        "top_ids",
        "top_logits",
        "log_normalizer",
        "attention",
    ):
        digest.update(trace[name].tobytes())
    return digest.hexdigest()


class FixedPrefixStates:
    def __init__(self, samples: Path, output: Path, device: str | None, atol: float):
        self.samples, self.output, self.device, self.atol = samples, output, device, atol

    @torch.inference_mode()
    def run(self) -> None:
        if self.atol < 0 or not np.isfinite(self.atol):
            raise ValueError("atol must be finite and nonnegative")
        settings = json.loads((self.samples / "settings.json").read_text(encoding="utf8"))
        identity = dict(model=settings["model"], dtype=settings["dtype"], atol=self.atol)
        self._directory(identity)
        prompts = {
            str(s["source_id"]): s["prompt"] for s in read_jsonl(self.samples / "prompts.jsonl")
        }
        model, tokenizer = None, None
        progress = tqdm(list(read_jsonl(self.samples / "samples.jsonl")), desc="fixed prefixes")
        for sample in progress:
            progress.set_postfix_str(sample["trace"])
            trace = self._trace(sample["trace"])
            target = self.output / sample["trace"]
            if target.exists():
                with np.load(target, allow_pickle=False) as saved:
                    if str(saved["signature"]) != trace_signature(trace):
                        raise ValueError(f"{target}: saved token/logit identity changed")
                continue
            if model is None:
                device = self.device or settings["device"]
                tokenizer = AutoTokenizer.from_pretrained(settings["model"], local_files_only=True)
                model = (
                    AutoModelForCausalLM.from_pretrained(
                        settings["model"],
                        local_files_only=True,
                        torch_dtype=getattr(torch, settings["dtype"]),
                        attn_implementation="eager",
                    )
                    .to(device)
                    .eval()
                )
            captured = self._capture(model, trace)
            positions, _, _ = source_positions(
                tokenizer,
                prompts[str(sample["source_id"])],
                trace["token_ids"][: int(trace["prompt_length"])],
            )
            mask = np.zeros(int(trace["prompt_length"]), dtype=bool)
            mask[positions] = True
            partial = target.with_suffix(".partial")
            with partial.open("wb") as stream:
                np.savez(
                    stream,
                    **captured,
                    source_mask=mask,
                    token_ids=trace["token_ids"],
                    signature=trace_signature(trace),
                )
            partial.replace(target)

    def _directory(self, identity):
        self.output.mkdir(parents=True, exist_ok=True)
        config = self.output / "settings.json"
        if config.exists():
            if json.loads(config.read_text(encoding="utf8")) != identity:
                raise ValueError("state directory was created with different model/dtype/tolerance")
        else:
            write_json(config, identity)

    def _trace(self, name):
        if Path(name).name != name or not name.endswith(".npz"):
            raise ValueError("trace must be an NPZ filename inside samples")
        with np.load(self.samples / name, allow_pickle=False) as saved:
            trace = {
                k: saved[k]
                for k in (
                    "token_ids",
                    "prompt_length",
                    "top_ids",
                    "top_logits",
                    "log_normalizer",
                    "attention",
                )
            }
        prompt = int(trace["prompt_length"])
        if not 0 < prompt < len(trace["token_ids"]):
            raise ValueError(f"{name}: invalid prompt length")
        if len(trace["top_logits"]) != len(trace["token_ids"]) - prompt:
            raise ValueError(f"{name}: logits and response lengths differ")
        return trace

    def _capture(self, model, trace):
        ids = torch.as_tensor(trace["token_ids"], device=model.device).reshape(1, -1)
        prompt, cache = int(trace["prompt_length"]), None
        steps = ids.shape[1] - prompt
        hidden = np.empty(
            (model.config.num_hidden_layers + 1, ids.shape[1] - 1, model.config.hidden_size),
            dtype=np.float16,
        )
        entropies, maximum, attention_error = [], 0.0, 0.0
        for t in tqdm(range(steps), desc="saved tokens", unit="token", leave=False):
            stop = prompt + t
            start = 0 if t == 0 else stop - 1
            output = model(
                input_ids=ids[:, start:stop],
                past_key_values=cache,
                use_cache=True,
                attention_mask=torch.ones_like(ids[:, :stop]),
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )
            logits = output.logits[0, -1].float()
            error = self._logit_error(logits, trace, t)
            maximum = max(maximum, error)
            attention_error = max(attention_error, self._attention_error(output, trace, t))
            logp = logits - logits.logsumexp(-1)
            entropies.append(float(-(logp.exp() * logp).sum() / np.log(2)))
            for layer, state in enumerate(output.hidden_states):
                hidden[layer, start:stop] = state[0].to("cpu", dtype=torch.float16).numpy()
            cache = output.past_key_values
            del output
        return dict(
            hidden=hidden,
            logit_entropy=np.array(entropies, dtype=np.float32),
            max_logit_error=np.array(maximum),
            max_attention_error=np.array(attention_error),
        )

    def _attention_error(self, output, trace, step):
        row = torch.stack([a[0, :, -1] for a in output.attentions])
        row = row.to("cpu", dtype=torch.float16).numpy()
        saved = trace["attention"][:, :, step, : row.shape[-1]]
        error = float(np.max(np.abs(row.astype(np.float32) - saved)))
        if not np.isfinite(error) or error > self.atol:
            raise ValueError(
                f"step {step}: fixed-prefix attention differs by {error:g} "
                f"(atol={self.atol:g}); do not combine inconsistent states"
            )
        return error

    def _logit_error(self, logits, trace, step):
        candidates = logits[torch.as_tensor(trace["top_ids"][step], device=logits.device)]
        error = max(
            float(np.max(np.abs(candidates.cpu().numpy() - trace["top_logits"][step]))),
            abs(float(logits.logsumexp(-1)) - float(trace["log_normalizer"][step])),
        )
        if not np.isfinite(error) or error > self.atol:
            raise ValueError(
                f"step {step}: fixed-prefix logits differ by {error:g} "
                f"(atol={self.atol:g}); check the model and numerical environment"
            )
        return error
