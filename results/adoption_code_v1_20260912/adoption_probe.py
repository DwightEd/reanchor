"""Diagnostic real-message and history-access interventions on a saved response."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from route_graph import adoption
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import __version__ as transformers_version


def write_json(path, value):
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def causal_groups(tokenizer, ids, prompt_length, source_mask):
    """Partition only observed tokens; punctuation closes a unit after its token."""
    groups, descriptions = [], []
    previous_role, ended = None, True
    for position, token in enumerate(ids):
        role = (
            "history"
            if position >= prompt_length
            else ("source" if source_mask[position] else "prompt_other")
        )
        text = tokenizer.decode([int(token)])
        if ended or role != previous_role:
            descriptions.append(dict(role=role, start=position, stop=position, text=""))
        groups.append(len(descriptions) - 1)
        descriptions[-1]["stop"] = position + 1
        descriptions[-1]["text"] += text
        ended = any(mark in text for mark in (".", "!", "?", "\n"))
        previous_role = role
    return np.asarray(groups), descriptions


@torch.no_grad()
def capture_terminal(model, ids, queries):
    layer, captured = model.model.layers[-1], {}

    def value_hook(module, args, output):
        captured["values"] = output[0].detach().clone()

    def residual_hook(module, args):
        captured["residual"] = args[0][0, queries].detach().clone()

    def attention_hook(module, args, output):
        captured["attention_output"] = output[0][0, queries].detach().clone()
        captured["attention"] = output[1][0, :, queries].detach().clone()

    handles = [
        layer.self_attn.v_proj.register_forward_hook(value_hook),
        layer.post_attention_layernorm.register_forward_pre_hook(residual_hook),
        layer.self_attn.register_forward_hook(attention_hook),
    ]
    try:
        output = model(ids, use_cache=False, output_attentions=True)
    finally:
        for handle in handles:
            handle.remove()
    captured["logits"] = output.logits[0].float().cpu()
    return captured


def run(args):
    started = time.time()
    samples, states, output_dir = Path(args.samples), Path(args.states), Path(args.output)
    if Path(args.trace).name != args.trace:
        raise ValueError("trace must be a filename")
    with np.load(samples / args.trace, allow_pickle=False) as saved:
        ids, prompt_length = saved["token_ids"].copy(), int(saved["prompt_length"])
    with np.load(states / args.trace, allow_pickle=False) as saved:
        source_mask = saved["source_mask"].copy()
        if not np.array_equal(saved["token_ids"], ids):
            raise ValueError("saved state token identity differs")
    response_length = len(ids) - prompt_length
    steps = args.queries
    if (
        len(source_mask) != prompt_length
        or not steps
        or len(set(steps)) != len(steps)
        or any(t < 0 or t >= response_length for t in steps)
        or response_length <= 134
    ):
        raise ValueError("this predeclared pilot requires a response longer than 134 tokens")
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(20260912)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = json.loads((samples / "settings.json").read_text())
    settings = dict(
        model=config["model"],
        trace=args.trace,
        queries=steps,
        history_cut=[119, 133],
        earlier_cut=[85, 99],
        first_affected_prediction=134,
        selection="previously inspected diagnostic case; no benchmark claim",
        dtype="bfloat16",
        suffix_dtype="float32",
        attention="eager",
        candidate_rule="native top8 union every unique alphanumeric non-special source token",
        torch=torch.__version__,
        transformers=transformers_version,
        cuda=torch.version.cuda,
        device=args.device,
        gpu=torch.cuda.get_device_name(torch.device(args.device)),
        seed=20260912,
        code_sha256={
            str(path.resolve()): sha256(path) for path in (Path(__file__), Path(adoption.__file__))
        },
        input_sha256={
            str(path.resolve()): sha256(path)
            for path in (samples / args.trace, samples / "settings.json")
        },
        source_mask_sha256=hashlib.sha256(source_mask.tobytes()).hexdigest(),
        model_files=[
            dict(name=path.name, size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
            for path in sorted(Path(config["model"]).iterdir())
            if path.is_file()
        ],
    )
    write_json(output_dir / "settings.json", settings)
    tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            config["model"],
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .to(args.device)
        .eval()
        .requires_grad_(False)
    )
    layer = model.model.layers[-1]
    if layer.self_attn.o_proj.bias is not None:
        raise ValueError("message reconstruction requires bias-free output projection")
    input_ids = torch.as_tensor(ids[:-1], device=args.device)[None]
    queries = np.asarray(steps) + prompt_length - 1
    captured = capture_terminal(model, input_ids, queries.tolist())
    values = captured["values"].reshape(
        len(ids) - 1, model.config.num_key_value_heads, layer.self_attn.head_dim
    )
    source_tokens = sorted(
        int(token)
        for token in np.unique(ids[:prompt_length][source_mask])
        if int(token) not in tokenizer.all_special_ids
        and any(char.isalnum() for char in tokenizer.decode([int(token)]))
    )
    records = []
    for index, (step, query) in enumerate(zip(steps, queries, strict=True)):
        groups, descriptions = causal_groups(
            tokenizer, ids[: query + 1], prompt_length, source_mask
        )
        native = captured["logits"][query]
        candidates = native.topk(8).indices.tolist()
        candidates += [token for token in source_tokens if token not in candidates]
        with torch.no_grad():
            head_messages = adoption.grouped_head_messages(
                captured["attention"][:, index, : query + 1],
                values[: query + 1],
                layer.self_attn.o_proj.weight,
                groups,
            )
            reconstructed = head_messages.sum((0, 1))
            actual = captured["attention_output"][index].float()
            closure = dict(
                max_absolute_error=float((reconstructed - actual).abs().max()),
                relative_l2_error=float(
                    (reconstructed - actual).norm() / actual.norm().clamp_min(1e-12)
                ),
            )
        suffix = adoption.TerminalSuffix(model, candidates)
        result, arrays = adoption.probe_messages(
            suffix, captured["residual"][index], head_messages, native[candidates].to(args.device)
        )
        result.update(
            step=int(step),
            next_saved_token=tokenizer.decode([int(ids[prompt_length + step])]),
            candidates=candidates,
            candidate_text=[tokenizer.decode([token]) for token in candidates],
            groups=descriptions,
            reconstruction=closure,
        )
        np.savez_compressed(output_dir / f"messages_{step:05d}.npz", **arrays)
        write_json(output_dir / f"messages_{step:05d}.json", result)
        records.append(result)
        print(
            f"MESSAGE step={step} groups={len(descriptions)} "
            f"candidates={len(candidates)} closure={closure}",
            flush=True,
        )
        del suffix, head_messages
    baseline = captured["logits"][prompt_length - 1 :]
    baseline_logp = baseline.log_softmax(-1)
    influence = dict(
        response_tokens=[tokenizer.decode([int(token)]) for token in ids[prompt_length:]],
        entropy_nats=(-(baseline_logp.exp() * baseline_logp).sum(-1)).tolist(),
        native_top1=baseline.argmax(-1).tolist(),
        conditions={},
    )
    for name, span in (("sham", None), ("history_cut", (119, 133)), ("earlier_cut", (85, 99))):
        blocked = (
            [] if span is None else list(range(prompt_length + span[0], prompt_length + span[1]))
        )
        mask = adoption.outgoing_key_mask(len(ids) - 1, blocked, prompt_length + 133)
        with torch.no_grad():
            changed = model(
                input_ids, attention_mask=mask.to(args.device, model.dtype), use_cache=False
            )
            logits = changed.logits[0, prompt_length - 1 :].float().cpu()
            del changed
        effect = adoption.distribution_effect(baseline, logits, ids[prompt_length:])
        effect["max_logit_error_before_activation"] = float(
            (baseline[:134] - logits[:134]).abs().max()
        )
        if name == "sham":
            effect["max_logit_error"] = float((baseline - logits).abs().max())
        influence["conditions"][name] = effect
        print(
            f"INFLUENCE condition={name} post134_mean_JS={np.mean(effect['js_nats'][134:]):.6g}",
            flush=True,
        )
    write_json(output_dir / "influence.json", influence)
    summary = dict(
        n_cases=1,
        n_queries=len(records),
        elapsed_seconds=time.time() - started,
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(torch.device(args.device)),
        sham_max_logit_error=influence["conditions"]["sham"]["max_logit_error"],
        numerical_checks_only=True,
        scientific_review="REVIEW_UNAVAILABLE",
    )
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "manifest.json",
        {path.name: sha256(path) for path in sorted(output_dir.iterdir()) if path.is_file()},
    )
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default="outputs/samples_20260911_145421_235")
    parser.add_argument("--states", default="outputs/states_samples_20260911_145421_235")
    parser.add_argument("--trace", default="00012.npz")
    parser.add_argument("--queries", nargs="+", type=int, default=[96, 99, 128, 131, 145, 166, 180])
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
