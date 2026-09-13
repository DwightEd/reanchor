"""Run one bounded relation experiment, then always resume the frozen population job."""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/ragtruth_population_20260912"


def save(path, value):
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(value, indent=2) + "\n")
    partial.replace(path)


def alive(pid):
    status = Path(f"/proc/{pid}/status")
    if not status.exists():
        return False
    return not any(
        line.startswith("State:") and "Z" in line for line in status.read_text().splitlines()
    )


def run(args):
    with (ROOT / "runs/relation_interleave_20260913.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(args)


def run_locked(args):
    journal = ROOT / "runs/relation_interleave_20260913.json"
    mechanism_log = ROOT / "runs/relation_only_20260913.log"
    resume_log = ROOT / "runs/ragtruth_population_resume_20260913.log"
    if any(p.exists() for p in (journal, args.output, mechanism_log, resume_log)):
        raise FileExistsError("interleave journal/output already exists")
    command_line = Path(f"/proc/{args.population_pid}/cmdline").read_bytes().split(b"\0")
    if b"decoding.ragtruth_population" not in command_line:
        raise ValueError("target PID is not the expected population module")
    if b"outputs/ragtruth_population_20260912" not in command_line:
        raise ValueError("target PID output differs")
    before = json.loads((RUN / "progress.json").read_text())
    if before["pid"] != args.population_pid or before["status"] != "running":
        raise ValueError("population progress and PID disagree")
    settings = json.loads((RUN / "settings.json").read_text())
    for name, expected in settings["code_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen population code changed: {name}")
    record = dict(
        status="pausing",
        previous_progress=before,
        started_unix=time.time(),
        mechanism_output=str(args.output),
        population_output=str(RUN),
    )
    save(journal, record)
    paused, signaled, child, population_lock = False, False, None, None
    try:
        os.kill(args.population_pid, signal.SIGINT)
        signaled = True
        for _ in range(60):
            if not alive(args.population_pid):
                paused = True
                break
            time.sleep(1)
        if not paused:
            raise RuntimeError("population did not exit; no second GPU job launched")
        population_lock = (RUN / ".lock").open("a")
        fcntl.flock(population_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record.update(status="running_relation", stopped_unix=time.time())
        save(journal, record)
        with mechanism_log.open("x") as log:
            child = subprocess.Popen(
                ["bash", "scripts/run_relation_only.sh", "--output", str(args.output)],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record["mechanism_pid"] = child.pid
            save(journal, record)
            result = child.wait()
        record["mechanism_exit_code"] = result
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
            child.wait()
        paused = paused or (signaled and not alive(args.population_pid))
        if population_lock is not None:
            population_lock.close()
        if paused:
            with resume_log.open("x") as log:
                resumed = subprocess.Popen(
                    [
                        "bash",
                        "scripts/run_ragtruth_population.sh",
                        "--output",
                        "outputs/ragtruth_population_20260912",
                        "--resume",
                    ],
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            record.update(
                status="population_resume_started",
                resumed_pid=resumed.pid,
                resume_log=str(resume_log),
                resumed_unix=time.time(),
            )
            save(journal, record)
            print(json.dumps(record), flush=True)
    if record.get("mechanism_exit_code") != 0:
        raise RuntimeError("relation experiment failed; population resume was launched")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
