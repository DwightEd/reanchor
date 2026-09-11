import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def result_repository(tmp_path):
    bash, git = shutil.which("bash"), shutil.which("git")
    if not bash or not git:
        pytest.skip("result-push integration requires Bash and Git")
    remote, repo = tmp_path / "remote.git", tmp_path / "checkout"
    repo.mkdir()

    def run_git(*args):
        return subprocess.run(
            [git, *args], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()

    run_git("init", "--bare", str(remote))
    run_git("init", "-b", "main")
    run_git("config", "user.name", "Result export test")
    run_git("config", "user.email", "test@example.invalid")
    run_git("config", "commit.gpgsign", "false")
    run_git("config", "core.autocrlf", "false")
    (repo / "scripts").mkdir()
    source = Path(__file__).resolve().parents[1] / "scripts/analyze_and_push.sh"
    if source.exists():
        (repo / "scripts/analyze_and_push.sh").write_bytes(source.read_bytes())
    # The producer is a separate executable; test the publishing command's Git behavior.
    (repo / "scripts/analyze_cases.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'mkdir -p "$OUTPUT_DIR/00012"\n'
        'printf "step,shift\\n0,\\n1,0.5\\n" > "$OUTPUT_DIR/00012/trajectory.csv"\n'
        'printf "%s\\n" "${1:-latest}" > "$OUTPUT_DIR/input.txt"\n'
        'exit "${PRODUCER_EXIT:-0}"\n',
        encoding="utf-8",
        newline="\n",
    )
    (repo / ".gitignore").write_text("outputs/\n", encoding="utf-8")
    (repo / "source.py").write_text("# tracked code\n", encoding="utf-8")
    (repo / "outputs").mkdir()
    (repo / "outputs/raw.npz").write_bytes(b"raw model states stay local")
    run_git("add", ".")
    run_git("commit", "-m", "Initial code")
    run_git("remote", "add", "origin", str(remote))
    run_git("push", "-u", "origin", "main")
    return repo, bash, run_git


def test_analyze_and_push_commits_only_results_and_pushes_to_the_current_branch(result_repository):
    repo, bash, git = result_repository
    old_head = git("rev-parse", "HEAD")
    (repo / "unrelated.txt").write_text("leave untracked", encoding="utf-8")
    environment = dict(os.environ)
    environment.pop("OUTPUT_DIR", None)
    run = subprocess.run(
        [bash, "scripts/analyze_and_push.sh", "outputs/existing samples"],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    assert git("rev-parse", "HEAD") != old_head
    assert git("rev-parse", "HEAD") == git("rev-parse", "origin/main")
    added = git("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines()
    assert added and all(path.startswith("results/") for path in added)
    assert any(path.endswith("trajectory.csv") for path in added)
    assert "unrelated.txt" not in git("ls-files").splitlines()
    assert not any(path.endswith(".npz") for path in git("ls-files").splitlines())
    assert (
        next((repo / "results").glob("*/input.txt")).read_text().strip()
        == "outputs/existing samples"
    )


@pytest.mark.parametrize("failure", ["analysis", "dirty_code"])
def test_result_publishing_stops_before_commit_or_push_on_failure(result_repository, failure):
    repo, bash, git = result_repository
    old_head = git("rev-parse", "HEAD")
    environment = dict(os.environ, PRODUCER_EXIT="7" if failure == "analysis" else "0")
    if failure == "dirty_code":
        (repo / "source.py").write_text("# uncommitted edit\n", encoding="utf-8")
    run = subprocess.run(
        [bash, "scripts/analyze_and_push.sh"],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert run.returncode == (7 if failure == "analysis" else 2)
    assert git("rev-parse", "HEAD") == git("rev-parse", "origin/main") == old_head
    assert not any(p.startswith("results/") for p in git("ls-files").splitlines())
