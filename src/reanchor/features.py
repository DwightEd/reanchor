"""Frozen causal replay and disk caches with explicit prediction/token alignment."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import torch

from reanchor.binding_data import SCHEMA, load_examples
from reanchor.io import digest, empty_directory, file_digest, read_jsonl, write_json


def candidate_ids(logits, source_ids, observed: int, special_ids, top_k: int = 32):
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    top = logits.topk(min(top_k, logits.numel())).indices.tolist()
    source = set(source_ids) - set(special_ids)
    return sorted(set(top) | source | {observed})


def token_kind(offset, example):
    if example["schema"] != SCHEMA:
        return "unlabeled"
    start, end = offset
    if end <= start:
        return None
    for kind in ("binding", "neutral"):
        spans = example["supervision"][f"{kind}_spans"]
        if any(start < b and end > a for a, b in spans):
            return kind
    return None


class FrozenBackbone:
    """First supported observer: Llama-family causal LM, local checkpoints only."""

    def __init__(self, model_path: Path, device="cuda:0", dtype="bfloat16", max_tokens=4096):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device)
        self.dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if not self.tokenizer.is_fast:
            raise ValueError("a fast tokenizer is required for character-span alignment")
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=self.dtype,
                attn_implementation="sdpa",
            )
            .to(self.device)
            .eval()
        )
        self.model.requires_grad_(False)
        if self.model.config.model_type != "llama":
            raise ValueError("G0 replay currently supports Llama-family checkpoints only")
        self.max_tokens = min(max_tokens, self.model.config.max_position_embeddings)
        config = self.model.config.to_dict()
        for key in ("_name_or_path", "transformers_version"):
            config.pop(key, None)
        weight_files = sorted(model_path.glob("*.safetensors"))
        if not weight_files:
            weight_files = sorted(model_path.glob("pytorch_model*.bin"))
        if not weight_files:
            raise ValueError("no local backbone weight files found for content verification")
        print("STAGE fingerprint checkpoint weight contents", flush=True)
        weight_digests = {p.name: file_digest(p) for p in weight_files}
        self.metadata = {
            "path": str(model_path.resolve()),
            "model_type": "llama",
            "dtype": dtype,
            "input_dim": self.model.config.hidden_size,
            "max_tokens": self.max_tokens,
            "fingerprint": digest(
                {
                    "config": config,
                    "tokenizer": self.tokenizer.backend_tokenizer.to_str(),
                    "chat_template": self.tokenizer.chat_template,
                    "special_ids": self.tokenizer.all_special_ids,
                    "weights": weight_digests,
                }
            ),
            "weight_digests": weight_digests,
            "fingerprint_scope": "config, tokenizer/template, special IDs and weight contents",
            "interpretation": "observer replay; not original-generator causal evidence",
        }

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def prompt_ids(self, instruction, source=None):
        content = instruction if source is None else f"Evidence:\n{source}\n\nTask:\n{instruction}"
        if self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return self.encode(text)
        ids = self.encode(f"User: {content}\nAssistant:")
        if self.tokenizer.bos_token_id is not None:
            ids.insert(0, self.tokenizer.bos_token_id)
        return ids

    @torch.no_grad()
    def states(self, ids):
        if not ids or len(ids) > self.max_tokens:
            raise ValueError(
                f"context length {len(ids)} outside [1,{self.max_tokens}]; no truncation"
            )
        tensor = torch.tensor([ids], device=self.device)
        return self.model.base_model(
            input_ids=tensor,
            attention_mask=torch.ones_like(tensor),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[0]

    @torch.no_grad()
    def logits(self, state):
        return self.model.get_output_embeddings()(state).float()

    @torch.no_grad()
    def embeddings(self, ids):
        ids = torch.tensor(ids, device=self.device)
        return self.model.get_input_embeddings()(ids).detach().cpu()


def extract_features(input_path: Path, output: Path, backbone, top_k=32, progress=True):
    from tqdm.auto import tqdm

    examples = load_examples(input_path)
    empty_directory(output)
    (output / "sources").mkdir()
    (output / "contexts").mkdir()
    known_sources = {}
    vocabulary = set()
    count = 0
    kinds = {}
    with (output / "samples.jsonl").open("x", encoding="utf-8") as stream:
        for example in tqdm(examples, desc="frozen causal replay", disable=not progress):
            source_key = digest(example["source"])
            if source_key not in known_sources:
                source_ids = backbone.encode(example["source"])
                if source_ids:
                    evidence = backbone.states(source_ids).detach().cpu()
                else:
                    evidence = torch.empty(0, backbone.metadata["input_dim"])
                source_file = f"sources/{source_key}.pt"
                torch.save(evidence, output / source_file)
                known_sources[source_key] = (source_file, source_ids)
            source_file, source_ids = known_sources[source_key]
            encoded = backbone.tokenizer(
                example["response"],
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            response_ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]
            positions = [(t, token_kind(offset, example)) for t, offset in enumerate(offsets)]
            positions = [(t, kind) for t, kind in positions if kind is not None]
            if not positions:
                raise ValueError(f"no scorable positions: {example['id']}")
            native_prompt = backbone.prompt_ids(example["instruction"], example["source"])
            blind_prompt = backbone.prompt_ids(example["instruction"])
            last_position = positions[-1][0]
            # A decoder-only causal pass is shared, but each read sees only <t.
            # y_t and future are never in the address slice supplied to the head.
            history = response_ids[:last_position]
            native = backbone.states(native_prompt + history)
            blind = backbone.states(blind_prompt + history).detach().cpu()
            context_file = f"contexts/{digest(example['id'])}.pt"
            torch.save(blind, output / context_file)
            for token_index, kind in positions:
                observed = int(response_ids[token_index])
                logits = backbone.logits(native[len(native_prompt) + token_index - 1])
                candidates = candidate_ids(
                    logits,
                    source_ids,
                    observed,
                    backbone.tokenizer.all_special_ids,
                    top_k,
                )
                vocabulary.update(candidates)
                other = logits.clone()
                other[observed] = -torch.inf
                margin = float((logits[observed] - other.max()).item())
                row = {
                    "response_id": example["id"],
                    "source_id": example["source_id"],
                    "split": example["split"],
                    "task": example["task"],
                    "token_index": token_index,
                    "token_id": observed,
                    "char_start": offsets[token_index][0],
                    "char_end": offsets[token_index][1],
                    "kind": kind,
                    "candidate_ids": candidates,
                    "target_index": candidates.index(observed),
                    "source_file": source_file,
                    "context_file": context_file,
                    "address_end": len(blind_prompt) + token_index,
                    "source_length": len(source_ids),
                    "negative_margin": -margin,
                    "response_digest": digest(example["response"]),
                    "source_text_digest": digest(example["source"]),
                    "instruction_digest": digest(example["instruction"]),
                    "response_token_count": len(response_ids),
                    "program": example.get("program", {}),
                }
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                kinds[kind] = kinds.get(kind, 0) + 1
                count += 1
            del native, blind
    ids = sorted(vocabulary)
    torch.save(
        {"ids": torch.tensor(ids), "values": backbone.embeddings(ids)}, output / "embeddings.pt"
    )
    extraction_identity = digest(
        {
            "backbone": backbone.metadata["fingerprint"],
            "dtype": backbone.metadata["dtype"],
            "max_tokens": backbone.max_tokens,
            "top_k": top_k,
            "schema": "reanchor/frozen-support-cache@2",
            "extractor_code": file_digest(Path(__file__)),
            "data_code": file_digest(Path(__file__).with_name("binding_data.py")),
            "torch_version": str(torch.__version__),
            "transformers_version": version("transformers"),
        }
    )
    tensor_digests = {
        p.relative_to(output).as_posix(): file_digest(p) for p in sorted(output.rglob("*.pt"))
    }
    preparation = {digest(e.get("dataset_identity")): e.get("dataset_identity") for e in examples}
    if len(preparation) != 1:
        raise ValueError("cannot mix preparation provenance within a cache")
    preparation_ids = {e.get("preparation_identity") for e in examples}
    if len(preparation_ids) != 1:
        raise ValueError("cannot mix preparation splits/configurations within a cache")
    manifest = {
        "schema": "reanchor/frozen-support-cache@2",
        "samples": count,
        "kinds": kinds,
        "input_digest": file_digest(input_path),
        "backbone": backbone.metadata,
        "top_k": top_k,
        "candidate_rule": "native-top-k + all-source-ids + observed",
        "examples": [{k: e[k] for k in ("id", "source_id", "split", "task")} for e in examples],
        "scope": "program-selected-slots"
        if all(e["schema"] == SCHEMA for e in examples)
        else "all-response-tokens",
        "dataset_identity": next(iter(preparation.values())),
        "preparation_identity": next(iter(preparation_ids)),
        "extraction_identity": extraction_identity,
        "index_digest": file_digest(output / "samples.jsonl"),
        "tensor_digests": tensor_digests,
    }
    manifest["cache_digest"] = digest(manifest)
    # Written last: incomplete extraction never has a usable manifest.
    write_json(output / "manifest.json", manifest)
    return manifest


class FeatureStore:
    def __init__(self, path: Path):
        self.path = path
        self.manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["schema"] != "reanchor/frozen-support-cache@2":
            raise ValueError("unsupported cache schema")
        body = {k: v for k, v in self.manifest.items() if k != "cache_digest"}
        if digest(body) != self.manifest["cache_digest"]:
            raise ValueError("cache manifest changed after extraction")
        if file_digest(path / "samples.jsonl") != self.manifest["index_digest"]:
            raise ValueError("cache index changed after extraction")
        for name, expected in self.manifest["tensor_digests"].items():
            target = (path / name).resolve()
            if not target.is_relative_to(path.resolve()) or file_digest(target) != expected:
                raise ValueError("cache tensor changed after extraction")
        self.samples = list(read_jsonl(path / "samples.jsonl"))
        if len(self.samples) != self.manifest["samples"]:
            raise ValueError("incomplete cache")
        embeddings = torch.load(path / "embeddings.pt", map_location="cpu", weights_only=True)
        self.ids, self.values = embeddings["ids"], embeddings["values"]

    @lru_cache(maxsize=8)
    def _tensor(self, name):
        if name not in self.manifest["tensor_digests"]:
            raise ValueError("cache tensor not registered in manifest")
        target = (self.path / name).resolve()
        if not target.is_relative_to(self.path.resolve()):
            raise ValueError("cache tensor path escapes cache directory")
        return torch.load(target, map_location="cpu", weights_only=True)

    def tensors(self, row, device):
        requested = torch.tensor(row["candidate_ids"])
        indices = torch.searchsorted(self.ids, requested)
        if (indices >= len(self.ids)).any() or not torch.equal(self.ids[indices], requested):
            raise ValueError("candidate embedding missing")
        source = self._tensor(row["source_file"]).to(device=device, dtype=torch.float32)
        full_address = self._tensor(row["context_file"])
        if not 1 <= row["address_end"] <= full_address.shape[0]:
            raise ValueError("invalid causal address slice")
        address = full_address[: row["address_end"]].to(device=device, dtype=torch.float32)
        candidates = self.values[indices].to(device=device, dtype=torch.float32)
        return source, address, candidates
