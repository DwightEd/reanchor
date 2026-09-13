"""Validate graph context readouts on paired, fixed-prefix binding worlds."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from route_graph.context import source_context_readout
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import __version__ as transformers_version

from decoding.io import read_jsonl
from decoding.mechanism import binding_input


def validation_input(tokenizer, case, world, subject, layout):
    if layout == "reverse":
        reversed_case = {**case, "subjects": case["subjects"][::-1], "values": case["values"][::-1]}
        item = binding_input(tokenizer, reversed_case, world, 1 - subject)
    else:
        item = binding_input(tokenizer, case, world, subject)
    if layout in {"distractor", "misbound"}:
        # Same potentially misleading history in BOTH worlds; never use the label.
        if layout == "distractor":
            prefix = f"One earlier value was {case['values'][1 - subject]}. " * 12
        else:
            prefix = (
                f"{case['subjects'][subject]} {case['predicate']} "
                f"{case['values'][subject]} {case['unit']}. "
            ) * 32
        prefix += f"Now, {case['subjects'][subject]} {case['predicate']} "
        item["ids"] = item["ids"][: item["prompt_length"]] + tokenizer.encode(
            prefix, add_special_tokens=False
        )
        item["prefix"] = prefix
    item["expected_token"] = item["candidates"][item["expected"]]
    return item


class BindingValidation:
    def __init__(
        self,
        model,
        cases,
        output,
        device="cuda:0",
        layouts=("clean", "reverse", "distractor"),
        limit=None,
    ):
        self.model, self.cases, self.output, self.device = model, cases, output, device
        self.layouts, self.limit = tuple(layouts), limit
        if (
            (limit is not None and limit < 1)
            or not self.layouts
            or len(set(self.layouts)) != len(self.layouts)
            or not set(self.layouts) <= {"clean", "reverse", "distractor", "misbound"}
        ):
            raise ValueError("invalid template limit or layouts")

    @torch.inference_mode()
    def run(self):
        self._settings()
        tokenizer = AutoTokenizer.from_pretrained(self.model, local_files_only=True)
        model = None
        records = list(read_jsonl(self.cases))[: self.limit]
        if not records or len({r["id"] for r in records}) != len(records):
            raise ValueError("expected nonempty distinct binding templates")
        if any(not r["id"].replace("_", "").isalnum() for r in records):
            raise ValueError("unsafe template ID")
        results = []
        for case in tqdm(records, desc="binding templates"):
            for layout in self.layouts:
                for subject in range(2):
                    worlds = [
                        validation_input(tokenizer, case, world, subject, layout)
                        for world in range(2)
                    ]
                    if worlds[0]["prefix"] != worlds[1]["prefix"] or Counter(
                        worlds[0]["ids"]
                    ) != Counter(worlds[1]["ids"]):
                        raise ValueError("paired worlds changed the fixed prefix or token multiset")
                    if worlds[0]["prompt_length"] != worlds[1]["prompt_length"]:
                        raise ValueError("paired prompt lengths differ")
                    for world, item in enumerate(worlds):
                        target = self.output / f"{case['id']}_{layout}_{subject}_{world}.json"
                        if target.exists():
                            results.append(json.loads(target.read_text()))
                            continue
                        if model is None:
                            model = (
                                AutoModelForCausalLM.from_pretrained(
                                    self.model,
                                    local_files_only=True,
                                    dtype=torch.bfloat16,
                                    attn_implementation="eager",
                                )
                                .to(self.device)
                                .eval()
                            )
                            if model.config.num_hidden_layers != 32:
                                raise ValueError(
                                    "predeclared layer pairs require the 32-layer model"
                                )
                        result = self._capture(model, tokenizer, case, item, layout, subject, world)
                        temporary = target.with_suffix(".partial")
                        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                        temporary.replace(target)
                        results.append(result)
        summary = summarize(results)
        (self.output / "numbers.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        (self.output / "complete.json").write_text(json.dumps(dict(conditions=len(results))) + "\n")
        print(json.dumps(summary, indent=2))
        return summary

    def _capture(self, model, tokenizer, case, item, layout, subject, world):
        started = time.monotonic()
        ids = torch.tensor([item["ids"]], device=self.device)
        output = model(ids, use_cache=False, output_attentions=True)
        logits = output.logits[0, -1].float()
        top = logits.topk(4)
        candidates = top.indices.cpu().tolist()
        logp = logits.log_softmax(-1)
        roles = np.ones(len(item["ids"]), dtype=int)
        roles[: item["prompt_length"]][item["source"]] = 0
        roles[item["prompt_length"] :] = 2
        roles[np.isin(item["ids"], tokenizer.all_special_ids)] = 3
        readouts = {}
        for pair in ([15, 16], [30, 31]):
            attention = np.stack(
                [output.attentions[layer][0].float().cpu().numpy() for layer in pair]
            )
            reading_seed = permutation_seed(case["id"], layout, subject, world, pair[-1])
            readouts[str(pair[-1])] = source_context_readout(
                attention,
                item["ids"],
                roles,
                len(item["ids"]) - 1,
                candidates,
                seed=reading_seed,
            )
        expected = item["expected_token"]
        alternate = next(c for c in item["candidates"] if c != expected)
        result = dict(
            template=case["id"],
            layout=layout,
            subject=subject,
            world=world,
            expected_token=expected,
            expected_text=tokenizer.decode([expected]),
            top_token=candidates[0],
            top_text=tokenizer.decode([candidates[0]]),
            model_correct=candidates[0] == expected,
            expected_in_top4=expected in candidates,
            candidates=candidates,
            candidate_text=tokenizer.batch_decode([[c] for c in candidates]),
            candidate_logits=top.values.cpu().tolist(),
            entropy_nats=float(-(logp.exp() * logp).sum()),
            expected_margin=float(logits[expected] - logits[alternate]),
            prompt=item["prompt"],
            prefix=item["prefix"],
            token_ids=item["ids"],
            prompt_length=item["prompt_length"],
            readouts=readouts,
            elapsed_seconds=time.monotonic() - started,
        )
        return result

    def _settings(self):
        from route_graph import context

        code_paths = [
            Path(__file__),
            Path(context.__file__),
            Path(__file__).with_name("mechanism.py"),
        ]
        settings = dict(
            model=str(self.model.resolve()),
            device=self.device,
            dtype="bfloat16",
            attention="eager",
            layer_pairs=[[15, 16], [30, 31]],
            candidates=4,
            window=16,
            permutation_seed=20260912,
            permutation_scheme="sha256(base,template,layout,subject,world,end_layer)[:8]",
            layouts=self.layouts,
            limit=self.limit,
            code_sha256=hashlib.sha256(
                b"".join(p.read_bytes().replace(b"\r\n", b"\n") for p in code_paths)
            ).hexdigest(),
            cases_sha256=hashlib.sha256(self.cases.read_bytes()).hexdigest(),
            torch=torch.__version__,
            transformers=transformers_version,
            model_files={
                p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                for p in sorted(self.model.iterdir())
                if p.suffix in {".json", ".safetensors"}
            },
        )
        settings = json.loads(json.dumps(settings))
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / "settings.json"
        if path.exists() and json.loads(path.read_text()) != settings:
            raise ValueError("binding validation settings changed; use a new output directory")
        temporary = path.with_suffix(".partial")
        temporary.write_text(json.dumps(settings, indent=2) + "\n")
        temporary.replace(path)
        source_dir = self.output / "source"
        source_dir.mkdir(exist_ok=True)
        for source in code_paths:
            (source_dir / source.name).write_bytes(source.read_bytes())
        (source_dir / "binding_cases.jsonl").write_bytes(self.cases.read_bytes())
        return settings


def permutation_seed(template, layout, subject, world, layer):
    identity = json.dumps([20260912, template, layout, subject, world, layer])
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")


def method_correct(row, layer, method):
    reading = row["readouts"][layer]
    if not reading[f"{method}_valid"]:
        return False
    support = np.asarray(reading[f"{method}_support"])
    maxima = np.flatnonzero(support == support.max())
    return len(maxima) == 1 and row["candidates"][int(maxima[0])] == row["expected_token"]


def summarize(rows):
    groups = {}
    methods = ("context", "direct", "shuffled", "count")
    for layout in sorted({r["layout"] for r in rows}):
        selected = [r for r in rows if r["layout"] == layout]
        group = dict(
            conditions=len(selected),
            templates=len({r["template"] for r in selected}),
            model_correct=sum(r["model_correct"] for r in selected),
            layers={},
        )
        for layer in ("16", "31"):
            common = [r for r in selected if r["readouts"][layer]["valid"]]
            errors = [r for r in common if not r["model_correct"]]
            values = dict(
                common_valid=len(common),
                actual_errors=sum(not r["model_correct"] for r in selected),
                common_actual_errors=len(errors),
                methods={},
                template_pairs=[],
            )
            for method in methods:
                valid = [r for r in selected if r["readouts"][layer][f"{method}_valid"]]
                values["methods"][method] = dict(
                    valid=len(valid),
                    correct=sum(method_correct(r, layer, method) for r in valid),
                    common_correct=sum(method_correct(r, layer, method) for r in common),
                    valid_errors=sum(not r["model_correct"] for r in valid),
                    error_correct=sum(
                        method_correct(r, layer, method) for r in valid if not r["model_correct"]
                    ),
                )
            values["pairwise"] = {}
            for baseline in ("direct", "shuffled", "count"):
                paired = [
                    r
                    for r in selected
                    if r["readouts"][layer]["context_valid"]
                    and r["readouts"][layer][f"{baseline}_valid"]
                ]
                paired_errors = [r for r in paired if not r["model_correct"]]
                values["pairwise"][f"context_vs_{baseline}"] = dict(
                    conditions=len(paired),
                    actual_errors=len(paired_errors),
                    context_correct=sum(method_correct(r, layer, "context") for r in paired),
                    baseline_correct=sum(method_correct(r, layer, baseline) for r in paired),
                    context_error_correct=sum(
                        method_correct(r, layer, "context") for r in paired_errors
                    ),
                    baseline_error_correct=sum(
                        method_correct(r, layer, baseline) for r in paired_errors
                    ),
                )
            for template in sorted({r["template"] for r in selected}):
                block = [r for r in selected if r["template"] == template]
                pairs = {(r["subject"], r["world"]) for r in block}
                if len(block) != 4 or pairs != {(s, w) for s in range(2) for w in range(2)}:
                    raise ValueError("template must contain both subjects and both worlds")
                pair = dict(template=template, conditions=len(block))
                for method in methods:
                    pair[method] = sum(method_correct(r, layer, method) for r in block) / len(block)
                    pair[f"{method}_all_pairs_correct"] = all(
                        method_correct(r, layer, method) for r in block
                    )
                pair["context_minus_direct"] = pair["context"] - pair["direct"]
                pair["context_minus_shuffled"] = pair["context"] - pair["shuffled"]
                values["template_pairs"].append(pair)
            group["layers"][layer] = values
        groups[layout] = group
    return dict(
        conditions=len(rows),
        layouts=groups,
        independent_unit="template",
        error_detection_tested=any(not r["model_correct"] for r in rows),
        scope=(
            "controlled next-token decisions; independent coverage and paired templates; "
            "no natural detection claim"
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=["clean", "reverse", "distractor", "misbound"],
        default=["clean", "reverse", "distractor"],
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    BindingValidation(
        args.model, args.cases, args.output, args.device, args.layouts, args.limit
    ).run()


if __name__ == "__main__":
    main()
