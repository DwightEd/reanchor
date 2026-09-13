"""Build complete attributed graphs for the real same-source decision pair."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from route_graph import sample_graph
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import __version__ as transformers_version

from decoding.adoption_probe import sha256, write_json


def run(output):
    root = Path(__file__).resolve().parents[2]
    samples = root / "outputs/samples_20260911_145421_235"
    states = root / "outputs/states_samples_20260911_145421_235"
    config = json.loads((samples / "settings.json").read_text())
    rows = {
        row["trace"]: row
        for row in map(json.loads, (samples / "samples.jsonl").read_text().splitlines())
    }
    traces = ("00012.npz", "00013.npz")
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    sources = [
        Path(__file__),
        Path(sample_graph.__file__),
        Path(__file__).with_name("adoption_probe.py"),
    ]
    settings = dict(
        traces=traces,
        selection="same-source real sampled responses; no labels in graph arrays",
        model=config["model"],
        model_files=[
            dict(name=path.name, size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
            for path in sorted(Path(config["model"]).iterdir())
            if path.is_file()
        ],
        code_sha256={str(path.resolve()): sha256(path) for path in sources},
        input_sha256={name: sha256(samples / name) for name in traces},
        torch=torch.__version__,
        transformers=transformers_version,
        gpu=torch.cuda.get_device_name(0),
        compute_dtype="bfloat16",
        storage_dtype="float32 (exactly preserves captured bf16, including small attention)",
        attention="eager",
        edge_orientation="attention[layer,head,receiver,sender]; messages sender->receiver",
        node_alignment="input token i, target t uses node prompt_length+t-1",
        output_projection="model self_attn.o_proj weights referenced by model identity",
        no_pruning=True,
        no_head_or_layer_average=True,
        no_candidate_projection=True,
    )
    for path in sources:
        (output / f"executed_{path.name}").write_bytes(path.read_bytes())
    write_json(output / "settings.json", settings)
    tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            config["model"],
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .to("cuda:0")
        .eval()
        .requires_grad_(False)
    )
    checks = {}
    for name in traces:
        with np.load(samples / name, allow_pickle=False) as saved:
            ids, prompt = saved["token_ids"].copy(), int(saved["prompt_length"])
        with np.load(states / name, allow_pickle=False) as saved:
            mask = saved["source_mask"].copy()
            if not np.array_equal(saved["token_ids"], ids):
                raise ValueError("source metadata has different token identity")
        graph = sample_graph.capture_sample_graph(
            model, ids, prompt, mask, tokenizer.all_special_ids
        )
        checks[name] = sample_graph.save_graph(
            output / Path(name).stem,
            graph,
            dict(
                trace=name,
                source_id=rows[name]["source_id"],
                seed=rows[name]["seed"],
                response=rows[name]["response"],
                input_token_text=[tokenizer.decode([int(token)]) for token in ids[:-1]],
                source_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                numerical_protocol="fresh complete forward; not mixed with cached states",
            ),
        )
        del graph
        print(f"GRAPH {name}: {json.dumps(checks[name])}", flush=True)
    summary = dict(
        graphs=checks,
        elapsed_seconds=time.time() - started,
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(0),
        constraint_ownership_identification="not yet evaluated",
    )
    write_json(output / "summary.json", summary)
    manifest = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
