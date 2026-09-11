"""Sample from a local model and save the states used to choose each token."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from reanchor.io import empty_directory, read_jsonl, write_json, write_jsonl


@dataclass(frozen=True)
class SamplingConfig:
    dataset: Path
    source_ids: tuple[str, ...]
    model: Path
    output: Path
    seeds: tuple[int, ...]
    max_new_tokens: int
    temperature: float
    top_p: float
    device: str
    dtype: str


class SamplingExperiment:
    def __init__(self, config: SamplingConfig):
        self.config = config

    @torch.inference_mode()
    def run(self) -> list[dict]:
        cfg = self.config
        if cfg.max_new_tokens < 1 or cfg.temperature <= 0 or not 0 < cfg.top_p <= 1:
            raise ValueError("invalid sampling parameters")
        if len(set(cfg.seeds)) != len(cfg.seeds) or len(set(cfg.source_ids)) != len(cfg.source_ids):
            raise ValueError("source IDs and seeds must be unique")
        selected = [
            row
            for row in read_jsonl(cfg.dataset / "source_info.jsonl")
            if str(row["source_id"]) in cfg.source_ids
        ]
        if len(selected) != len(cfg.source_ids) or {str(r["source_id"]) for r in selected} != set(
            cfg.source_ids
        ):
            raise ValueError("requested sources missing or duplicated in dataset")
        empty_directory(cfg.output)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model, local_files_only=True)
        model = (
            AutoModelForCausalLM.from_pretrained(
                cfg.model,
                local_files_only=True,
                torch_dtype=getattr(torch, cfg.dtype),
                attn_implementation="eager",
            )
            .to(cfg.device)
            .eval()
        )
        settings = asdict(cfg)
        for name in ("dataset", "model", "output"):
            settings[name] = str(settings[name])
        write_json(cfg.output / "settings.json", settings)
        write_jsonl(cfg.output / "prompts.jsonl", selected)
        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
        records = []
        for source in tqdm(selected, desc="questions", unit="question"):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": source["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(cfg.device)
            for seed in cfg.seeds:
                trace = self._generate(model, prompt, seed, eos_ids)
                token_ids = trace["token_ids"]
                trace["special_mask"] = np.isin(token_ids, tokenizer.all_special_ids)
                filename = f"{len(records):05d}.npz"
                np.savez_compressed(cfg.output / filename, **trace)
                response_ids = token_ids[prompt.shape[1] :]
                records.append(
                    {
                        "source_id": str(source["source_id"]),
                        "seed": seed,
                        "response": tokenizer.decode(response_ids, skip_special_tokens=True),
                        "tokens": len(response_ids),
                        "stop_reason": "eos"
                        if int(response_ids[-1]) in eos_ids
                        else "max_new_tokens",
                        "trace": filename,
                    }
                )
                # Write each completed sample immediately; a later failure leaves it inspectable.
                with (cfg.output / "samples.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(records[-1], ensure_ascii=False) + "\n")
        return records

    def _generate(self, model, prompt, seed, eos_ids):
        cfg = self.config
        generator = torch.Generator(device=cfg.device).manual_seed(seed)
        ids = prompt
        next_input, cache = prompt, None
        attention, states, top_ids, top_logits, entropy, logprob, sampling_logprob = (
            [] for _ in range(7)
        )
        for _ in tqdm(range(cfg.max_new_tokens), desc=f"seed {seed}", unit="token", leave=False):
            if ids.shape[1] > model.config.max_position_embeddings:
                raise ValueError("generation exceeded model context length")
            output = model(
                input_ids=next_input,
                past_key_values=cache,
                use_cache=True,
                attention_mask=torch.ones_like(ids),
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            logits = output.logits[0, -1].float()
            log_p = logits.log_softmax(-1)
            probabilities = (logits / cfg.temperature).softmax(-1)
            values, indices = probabilities.sort(descending=True)
            # Keep the token that crosses the top-p boundary.
            remove = values.cumsum(-1) - values >= cfg.top_p
            probabilities[indices[remove]] = 0
            probabilities /= probabilities.sum()
            chosen = torch.multinomial(probabilities, 1, generator=generator)
            logprob.append(float(log_p[chosen]))
            sampling_logprob.append(float(probabilities[chosen].log()))
            entropy.append(float(torch.special.entr(log_p.exp()).sum()))
            top = logits.topk(min(32, logits.numel()))
            top_ids.append(top.indices.cpu().numpy())
            top_logits.append(top.values.cpu().numpy())
            # Row t is query P+t-1, the state BEFORE sampling response token t.
            attention.append(
                torch.stack([a[0, :, -1] for a in output.attentions])
                .to(device="cpu", dtype=torch.float16)
                .numpy()
            )
            states.append(
                torch.stack([h[0, -1] for h in output.hidden_states])
                .to(device="cpu", dtype=torch.float16)
                .numpy()
            )
            cache = output.past_key_values
            next_input = chosen.reshape(1, 1)
            ids = torch.cat((ids, next_input), dim=1)
            del output
            if int(chosen) in eos_ids:
                break
        layers, heads = attention[0].shape[:2]
        weights = np.zeros((layers, heads, len(attention), ids.shape[1]), dtype=np.float16)
        for t, row in enumerate(attention):
            weights[:, :, t, : row.shape[-1]] = row
        return {
            "token_ids": ids[0].cpu().numpy(),
            "prompt_length": np.array(prompt.shape[1]),
            "attention": weights,
            "hidden_states": np.stack(states, axis=1),
            "top_ids": np.stack(top_ids),
            "top_logits": np.stack(top_logits),
            "entropy": np.array(entropy),
            "logprob": np.array(logprob),
            "sampling_logprob": np.array(sampling_logprob),
        }
