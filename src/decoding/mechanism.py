"""Controlled relation swaps, route cuts, and natural decision windows."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from decoding.interventions import CausalReadout, attention_mask, history_mask, local_to_remote
from decoding.io import read_jsonl


def binding_input(tokenizer, case, world, subject):
    facts = [
        f"{name} {case['predicate']} {case['values'][(i + world) % 2]} {case['unit']}."
        for i, name in enumerate(case["subjects"])
    ]
    prompt = (
        "Complete the answer using only the stated facts. Keep the object and its "
        "constraint together.\nFacts:\n" + "\n".join(facts)
    )
    text = tokenizer.apply_chat_template(
        [dict(role="user", content=prompt)], tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    spans = []
    for fact in facts:
        left = text.index(fact)
        spans.append(
            [
                i
                for i, (a, b) in enumerate(encoded["offset_mapping"])
                if a < left + len(fact) and b > left
            ]
        )
    prefix = (
        "I will use the stated facts to identify the required value.\n"
        f"{case['subjects'][subject]} {case['predicate']} "
    )
    response = tokenizer.encode(prefix, add_special_tokens=False)
    candidates = [tokenizer.encode(v, add_special_tokens=False) for v in case["values"]]
    if any(len(c) != 1 for c in candidates) or candidates[0] == candidates[1]:
        raise ValueError("binding values must be distinct single tokens")
    prompt_length = len(encoded["input_ids"])
    source = np.zeros(prompt_length, bool)
    source[spans[0] + spans[1]] = True
    return dict(
        ids=encoded["input_ids"] + response,
        prompt_length=prompt_length,
        source=source,
        spans=spans,
        candidates=[c[0] for c in candidates],
        expected=(subject + world) % 2,
        prefix=prefix,
        prompt=prompt,
    )


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class MechanismExperiment:
    def __init__(self, model, cases, output, samples=None, states=None, device="cuda:0"):
        self.model_path, self.cases, self.output = model, cases, output
        self.samples, self.states, self.device = samples, states, device

    def run(self):
        self._settings()
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_path,
                local_files_only=True,
                dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            .to(self.device)
            .eval()
        )
        reader = CausalReadout(model)
        for case in tqdm(list(read_jsonl(self.cases)), desc="binding cases"):
            target = self.output / case["id"]
            if (target / "complete.json").exists():
                continue
            target.mkdir(exist_ok=True)
            self._binding(reader, tokenizer, case, target)
        if self.samples is not None:
            self._natural(reader, tokenizer)

    def _settings(self):
        if (self.samples is None) != (self.states is None):
            raise ValueError("natural windows require both --samples and --states")
        self.output.mkdir(parents=True, exist_ok=True)
        code = (
            Path(__file__).read_bytes() + Path(__file__).with_name("interventions.py").read_bytes()
        ).replace(b"\r\n", b"\n")
        settings = dict(
            model=str(self.model_path),
            device=self.device,
            dtype="bfloat16",
            samples=str(self.samples),
            states=str(self.states),
            cases_sha256=hashlib.sha256(self.cases.read_bytes()).hexdigest(),
            code_sha256=hashlib.sha256(code).hexdigest(),
            entropy_unit="nats",
        )
        path = self.output / "settings.json"
        if path.exists() and json.loads(path.read_text()) != settings:
            raise ValueError("output settings differ; choose a new output directory")
        path.write_text(json.dumps(settings, indent=2) + "\n")

    def _binding(self, reader, tokenizer, case, target):
        token_rows, layer_rows, checks = [], [], []
        for subject in range(2):
            worlds = [binding_input(tokenizer, case, w, subject) for w in range(2)]
            if Counter(worlds[0]["ids"]) != Counter(worlds[1]["ids"]):
                raise ValueError("relation swap changed the token multiset")
            if worlds[0]["prompt_length"] != worlds[1]["prompt_length"]:
                raise ValueError("relation swap changed prompt length")
            baseline = []
            for world, item in enumerate(worlds):
                item["condition"] = f"object{subject}_world{world}"
                item["queries"] = np.arange(item["prompt_length"] - 1, len(item["ids"]))
                item["candidates"] = np.tile(item["candidates"], (len(item["queries"]), 1))
                baseline.append(
                    reader.run(
                        item["ids"],
                        item["queries"],
                        item["candidates"],
                        mask=attention_mask(len(item["ids"]), item["prompt_length"]),
                    )
                )
            for world, item in enumerate(worlds):
                variants = self._binding_variants(
                    reader, item, baseline[world], baseline[1 - world], subject
                )
                for name, result in variants:
                    tokens, layers = self._rows(tokenizer, item, result, name)
                    token_rows.extend(tokens)
                    layer_rows.extend(layers)
                    if name == "sham_mlp":
                        checks.append(
                            dict(
                                condition=item["condition"],
                                sham_max_logit_error=result["repeat_error"],
                            )
                        )
        save_csv(target / "tokens.csv", token_rows)
        save_csv(target / "layers.csv", layer_rows)
        (target / "complete.json").write_text(json.dumps(dict(case=case, checks=checks), indent=2))

    def _binding_variants(self, reader, item, full, donor, subject):
        p, n = item["prompt_length"], len(item["ids"])
        masks = dict(
            cut_required=attention_mask(n, p, blocked=item["spans"][subject]),
            cut_competitor=attention_mask(n, p, blocked=item["spans"][1 - subject]),
            local16=attention_mask(n, p, window=16),
            local64=attention_mask(n, p, window=64),
        )
        yield "full", full
        for name, mask in tqdm(masks.items(), desc=item["condition"], leave=False):
            yield name, reader.run(item["ids"], item["queries"], item["candidates"], mask=mask)
        for name, states in [("sham_mlp", full), ("swap_mlp", donor)]:
            result = reader.run(
                item["ids"],
                item["queries"],
                item["candidates"],
                mask=attention_mask(n, p),
                mlp_patch=states["mlp_updates"],
            )
            if name == "sham_mlp":
                error = float(np.max(np.abs(result["logits"] - full["logits"])))
                if error > 1e-5:
                    raise ValueError(f"sham intervention changed logits by {error}")
                result["repeat_error"] = error
            yield name, result

    def _rows(self, tokenizer, item, result, variant):
        queries, candidates = item["queries"], item["candidates"]
        key, expected = item["condition"], item.get("expected")
        token_rows, layer_rows = [], []
        shifts = {
            w: local_to_remote(result["attention"], queries, item["source"], w) for w in [8, 16, 32]
        }
        for t, query in enumerate(queries):
            logits = result["logits"][t]
            top = np.argpartition(logits, -2)[-2:]
            top = top[np.argsort(logits[top])[::-1]]
            a, b = candidates[t]
            token_rows.append(
                dict(
                    condition=key,
                    variant=variant,
                    step=int(query - item["prompt_length"] + 1),
                    query=tokenizer.decode([item["ids"][query]]),
                    entropy=float(result["entropy"][t]),
                    top_margin=float(logits[top[0]] - logits[top[1]]),
                    candidate_a=tokenizer.decode([int(a)]),
                    candidate_b=tokenizer.decode([int(b)]),
                    candidate_margin=float(logits[a] - logits[b]),
                    top_token=tokenizer.decode([int(top[0])]),
                    expected_candidate="" if expected is None else expected,
                    revisit8=float(shifts[8][t]),
                    revisit16=float(shifts[16][t]),
                    revisit32=float(shifts[32][t]),
                    is_value_slot=int(expected is not None and query == len(item["ids"]) - 1),
                )
            )
            for layer, values in enumerate(result["layer_contrast"]):
                layer_rows.append(
                    dict(
                        condition=key,
                        variant=variant,
                        step=int(query - item["prompt_length"] + 1),
                        layer=layer,
                        before_attention=float(values[0, t]),
                        after_attention=float(values[1, t]),
                        after_mlp=float(values[2, t]),
                    )
                )
        return token_rows, layer_rows

    def _natural(self, reader, tokenizer):
        cases = [
            ("cooking_correct", "00012.npz", "14375", 0, 79, 109),
            ("cooking_wrong", "00012.npz", "14375", 0, 102, 141),
            ("headdress", "00006.npz", "14315", 2, 17, 46),
            ("clothes", "00007.npz", "14315", 3, 2, 46),
            ("normal_transition", "00000.npz", "14304", 0, 53, 81),
        ]
        samples = {r["trace"]: r for r in read_jsonl(self.samples / "samples.jsonl")}
        for name, trace, source_id, seed, start, stop in tqdm(cases, desc="natural windows"):
            if str(samples[trace]["source_id"]) != source_id or samples[trace]["seed"] != seed:
                raise ValueError(f"{trace}: expected source {source_id}, seed {seed}")
            target = self.output / name
            if (target / "complete.json").exists():
                continue
            target.mkdir(exist_ok=True)
            with np.load(self.samples / trace, allow_pickle=False) as saved:
                ids, p = saved["token_ids"], int(saved["prompt_length"])
                top = saved["top_ids"][:stop]
            with np.load(self.states / trace, allow_pickle=False) as saved:
                if not np.array_equal(ids, saved["token_ids"]):
                    raise ValueError("saved states belong to different generated tokens")
                source = saved["source_mask"]
            if len(ids) - p < stop:
                raise ValueError(f"{trace}: the declared natural window exceeds the sample")
            item = dict(ids=ids[: p + stop - 1].tolist(), prompt_length=p, source=source)
            queries = np.arange(p + start - 1, p + stop - 1)
            chosen = ids[p + start : p + stop]
            alternative = [
                row[0] if row[0] != token else row[1] for row, token in zip(top[start:stop], chosen)
            ]
            candidates = np.column_stack([chosen, alternative])
            item.update(queries=queries, candidates=candidates, condition=name)
            n = len(item["ids"])
            masks = dict(
                full=attention_mask(n, p),
                cut_source=attention_mask(n, p, blocked=np.flatnonzero(source)),
                cut_remote_history=history_mask(n, p, 16, recent=False),
                cut_recent_history=history_mask(n, p, 16, recent=True),
                local16=attention_mask(n, p, window=16),
                local64=attention_mask(n, p, window=64),
            )
            token_rows, layer_rows, repeat_error = [], [], None
            for variant, mask in tqdm(masks.items(), desc=name, leave=False):
                result = reader.run(item["ids"], queries, candidates, mask=mask)
                if variant == "full":
                    repeated = reader.run(item["ids"], queries, candidates, mask=mask)
                    repeat_error = float(np.max(np.abs(result["logits"] - repeated["logits"])))
                    if repeat_error > 1e-5:
                        raise ValueError(
                            f"{name}: repeated prefix changed logits by {repeat_error}"
                        )
                    del repeated
                tokens, layers = self._rows(tokenizer, item, result, variant)
                token_rows.extend(tokens)
                layer_rows.extend(layers)
            save_csv(target / "tokens.csv", token_rows)
            save_csv(target / "layers.csv", layer_rows)
            (target / "complete.json").write_text(
                json.dumps(
                    dict(
                        trace=trace,
                        start=start,
                        stop=stop,
                        repeat_max_logit_error=repeat_error,
                        labels="inspection windows, not onset ground truth",
                    )
                )
            )
