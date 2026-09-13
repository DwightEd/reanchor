"""Persistent single-GPU RAGTruth mechanism sweep with response-level resume."""

import argparse
import fcntl
import gc
import json
import os
import time
import traceback
from pathlib import Path

import torch
from route_graph import population_mechanism as mechanism
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import __version__ as transformers_version

from decoding import adoption_probe, ragtruth_population_io
from decoding.adoption_probe import sha256, write_json
from decoding.ragtruth_population_io import (
    append_json,
    prepare_records,
    publish_response,
    roster_summary,
    verify_response,
)

ROOT = Path(__file__).resolve().parents[2]
LYS = ROOT.parent.parent


def contract(args):
    files = [
        Path(__file__),
        Path(mechanism.__file__),
        Path(ragtruth_population_io.__file__),
        Path(adoption_probe.__file__),
        ROOT / "scripts/run_ragtruth_population.sh",
        ROOT / "src/decoding/ragtruth_population_evaluate.py",
        ROOT.parent / "graph/route_graph/metrics.py",
        ROOT.parent / "graph/docs/RAGTRUTH_POPULATION_MECHANISM_PLAN_20260912.md",
    ]
    settings = dict(
        schema="ragtruth-population-mechanism@1",
        model=str(args.model.resolve()),
        model_files=[
            dict(name=p.name, size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
            for p in sorted(args.model.iterdir())
            if p.is_file()
        ],
        dataset=str(args.dataset.resolve()),
        input_sha256={
            name: sha256(args.dataset / name) for name in ("source_info.jsonl", "response.jsonl")
        },
        code_sha256={str(p.resolve()): sha256(p) for p in files},
        torch=torch.__version__,
        transformers=transformers_version,
        gpu=torch.cuda.get_device_name(0),
        dtype="bfloat16",
        attention="eager",
        max_input_tokens=args.max_input_tokens,
        query_chunk=args.chunk,
        history_window=args.history_window,
        seed=args.seed,
        smoke=args.smoke,
        conditions=mechanism.CONDITIONS,
        tokenization="BOS + original prompt separately tokenized + original response",
        observer_replay=True,
        labels_used=False,
        scientific_review="REVIEW_UNAVAILABLE",
    )
    return settings, files


def run(args):
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"use a new output or explicit --resume: {output}")
    if args.resume and not (output / "settings.json").exists():
        raise ValueError("resume requires initialized settings")
    output.mkdir(parents=True, exist_ok=args.resume)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(args, output)


def run_locked(args, output):
    started = time.monotonic()
    settings, code = contract(args)
    # JSON conversion normalizes tuples before equality checks on resume.
    settings = json.loads(json.dumps(settings))
    if args.resume:
        if settings != json.loads((output / "settings.json").read_text()):
            raise ValueError("code/model/input/scientific settings changed; use a new output")
    else:
        write_json(output / "settings.json", settings)
        execution = output / "execution"
        execution.mkdir()
        for p in code:
            (execution / p.name).write_bytes(p.read_bytes())
    write_json(
        output / "progress.json",
        dict(status="preparing", pid=os.getpid(), started_unix=time.time()),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    roster_path = output / "inputs.jsonl"
    if roster_path.exists():
        manifest_path = output / "input_manifest.json"
        if not manifest_path.exists():
            rebuilt = prepare_records(args.dataset, tokenizer, args.smoke)
            existing = [json.loads(line) for line in roster_path.open()]
            if existing != json.loads(json.dumps(rebuilt)):
                raise ValueError("interrupted input roster differs from frozen input")
            write_json(manifest_path, dict(sha256=sha256(roster_path), **roster_summary(rebuilt)))
        expected = json.loads(manifest_path.read_text())
        if sha256(roster_path) != expected["sha256"]:
            raise ValueError("stored input roster changed")
        records = [json.loads(line) for line in roster_path.open()]
    else:
        records = prepare_records(args.dataset, tokenizer, args.smoke)
        partial = output / "inputs.partial.jsonl"
        with partial.open("w") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        partial.replace(roster_path)
        write_json(
            output / "input_manifest.json",
            dict(sha256=sha256(roster_path), **roster_summary(records)),
        )
    overview = roster_summary(records)
    write_json(
        output / "length_preflight.json",
        dict(
            configured_limit=args.max_input_tokens,
            max_input_tokens=overview["max_input_tokens"],
            oversized=[
                dict(id=r["id"], input_tokens=len(r["token_ids"]) - 1)
                for r in records
                if len(r["token_ids"]) - 1 > args.max_input_tokens
            ],
        ),
    )
    print(json.dumps(dict(roster=overview)), flush=True)
    responses, partials = output / "responses", output / "partial"
    responses.mkdir(exist_ok=True)
    partials.mkdir(exist_ok=True)
    done, failures = set(), {}
    error_path = output / "failures.jsonl"
    if error_path.exists():
        for line in error_path.open():
            row = json.loads(line)
            failures[row["id"]] = row
    for record in records:
        destination = responses / record["id"]
        if destination.exists():
            verify_response(destination, record)
            done.add(record["id"])
    total, last = len(records), None

    def progress(status="running", current=None, condition=None):
        failed = len(set(failures) - done)
        write_json(
            output / "progress.json",
            dict(
                status=status,
                pid=os.getpid(),
                total=total,
                completed=len(done),
                failed=failed,
                remaining=total - len(done) - failed,
                current_response=current,
                condition=condition,
                last_response_seconds=last,
                elapsed_seconds_this_process=time.monotonic() - started,
                updated_unix=time.time(),
                **{"by_task": overview["by_task"]},
            ),
        )

    pending = [
        r for r in records if r["id"] not in done and (args.retry_failed or r["id"] not in failures)
    ]
    progress("loading_model" if pending else "finishing")
    if pending:
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
            )
            .to("cuda:0")
            .eval()
            .requires_grad_(False)
        )
        for record in pending:
            torch.cuda.reset_peak_memory_stats(0)
            rid = record["id"]
            item_started = time.monotonic()
            partial = partials / f"{rid}_{time.time_ns()}"
            partial.mkdir()
            try:
                ids, prompt = record["token_ids"], record["prompt_length"]
                if len(ids) - 1 > min(args.max_input_tokens, model.config.max_position_embeddings):
                    raise ValueError(f"input length {len(ids) - 1} exceeds explicit context limit")
                saved = ids[prompt:]
                permutation = mechanism.endpoint_permutation(
                    sum(record["source_mask"]), rid, args.seed
                )
                metrics, profiles, diagnostics, baseline = {}, {}, {}, None
                for condition in mechanism.CONDITIONS:
                    progress(current=rid, condition=condition)
                    states, captured, diag = mechanism.response_forward(
                        model,
                        ids[:-1],
                        prompt,
                        record["source_mask"],
                        condition,
                        permutation,
                        history_window=args.history_window,
                        chunk=args.chunk,
                    )
                    if baseline is None:
                        baseline, profiles = states, captured
                    prefix_error = (
                        float((states[: prompt - 1] - baseline[: prompt - 1]).abs().max())
                        if prompt > 1
                        else 0.0
                    )
                    if prefix_error:
                        raise ValueError(
                            f"intervention changed earlier prompt states: {prefix_error}"
                        )
                    diag["prompt_state_max_error"] = prefix_error
                    diag["prompt_state_count"] = prompt - 1
                    if condition == "sham":
                        diag["all_state_max_error"] = float((states - baseline).abs().max())
                        if diag["all_state_max_error"]:
                            raise ValueError("nonzero same-input sham")
                    values = mechanism.compare_states(
                        model, baseline[prompt - 1 :], states[prompt - 1 :], saved, args.chunk
                    )
                    metrics.update({f"{condition}__{key}": value for key, value in values.items()})
                    diagnostics[condition] = diag
                diagnostics["source_permutation"] = permutation.tolist()
                diagnostics["elapsed_seconds"] = time.monotonic() - item_started
                diagnostics["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
                publish_response(partial, responses / rid, record, metrics, profiles, diagnostics)
                done.add(rid)
                last = time.monotonic() - item_started
                progress()
                print(
                    f"DONE {len(done)}/{total} id={rid} task={record['task']} "
                    f"generator={record['generator']} tokens={len(saved)} seconds={last:.2f}",
                    flush=True,
                )
            except Exception as error:
                failure = dict(
                    id=rid,
                    task=record["task"],
                    generator=record["generator"],
                    error_type=type(error).__name__,
                    error=str(error),
                    traceback=traceback.format_exc(),
                    partial=str(partial),
                    time_unix=time.time(),
                )
                failures[rid] = failure
                write_json(partial / "failure.json", failure)
                append_json(error_path, failure)
                progress()
                print(f"FAILED id={rid}: {type(error).__name__}: {error}", flush=True)
                if not isinstance(error, torch.cuda.OutOfMemoryError):
                    progress("failed", current=rid)
                    raise
                gc.collect()
                torch.cuda.empty_cache()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    failed = set(failures) - done
    if len(done) + len(failed) != total:
        progress("incomplete")
        raise ValueError("queue ended before all records were accounted for")
    progress("evaluating")
    for name, expected in settings["code_sha256"].items():
        if sha256(Path(name)) != expected:
            raise ValueError("execution code changed before evaluation; artifacts preserved")
    # This is the only stage that reads hallucination annotations.
    from decoding.ragtruth_population_evaluate import evaluate

    evaluation = output / f"evaluation_{time.time_ns()}.json"
    evaluate(output, args.dataset / "response.jsonl", evaluation)
    progress("finished_with_failures" if failed else "complete")
    marker = "COMPLETED_WITH_FAILURES" if failed else "COMPLETE"
    (output / marker).write_text(f"completed={len(done)} failed={len(failed)} total={total}\n")
    print(f"EVALUATION {evaluation}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=LYS / "data/RAGTruth/dataset")
    parser.add_argument("--model", type=Path, default=LYS / "models/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--history-window", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
