import os
import shutil
import subprocess
from pathlib import Path

import pytest


def shell_path(path):
    path = Path(path).resolve().as_posix()
    if os.name == "nt" and len(path) > 1 and path[1] == ":":
        return "/" + path[0].lower() + path[2:]
    return path


def test_shell_propagates_failure_and_preserves_existing_log(tmp_path):
    bash = shutil.which("bash")
    if bash is None and Path("D:/Apps/Research/Tools/Git/bin/bash.exe").is_file():
        bash = "D:/Apps/Research/Tools/Git/bin/bash.exe"
    if bash is None:
        pytest.skip("bash unavailable")
    fake = tmp_path / "failed-python.sh"
    fake.write_text("#!/usr/bin/env bash\nprintf 'fixture failure\\n'\nexit 7\n", encoding="utf-8")
    fake.chmod(0o755)
    output = tmp_path / "run with spaces"
    env = {
        **os.environ,
        "PYTHON_BIN": shell_path(fake),
        "OUTPUT_DIR": shell_path(output),
        "PROGRAM_ONLY": "1",
    }
    repo = Path(__file__).resolve().parents[1]
    command = [bash, "scripts/run_g0.sh"]
    failed = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, timeout=30)
    assert failed.returncode == 7, failed.stdout + failed.stderr
    log = Path(str(output) + ".log")
    assert "fixture failure" in log.read_text(encoding="utf-8")
    before = log.read_bytes()
    refused = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, timeout=30)
    assert refused.returncode == 2
    assert log.read_bytes() == before
