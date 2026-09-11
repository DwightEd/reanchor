#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
python_bin=${PYTHON_BIN:-python}

if (( $# > 1 )); then
  printf 'Usage: bash scripts/analyze_cases.sh [existing_samples_directory]\n' >&2
  exit 2
fi
samples=${1:-${SAMPLES_DIR:-}}
if [[ -z "$samples" ]]; then
  samples=$("$python_bin" - <<'PY'
from pathlib import Path

captures = list(Path("outputs").glob("samples_*/samples.jsonl"))
if not captures:
    raise SystemExit("No existing samples found. Pass the captured samples directory as argument.")
print(max(captures, key=lambda p: p.stat().st_mtime_ns).parent)
PY
  )
fi

if ! "$python_bin" -c 'import importlib.util; raise SystemExit(importlib.util.find_spec("matplotlib") is None)'; then
  "$python_bin" -m pip install 'matplotlib>=3.6'
fi
output=${OUTPUT_DIR:-"$samples/routes_$(date +%Y%m%d_%H%M%S)_$$"}
args=(--samples "$samples" --cases "$repo_root/examples/route_cases.csv" --output "$output")
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
  args+=(--tokenizer "$TOKENIZER_PATH")
fi
printf 'Existing samples: %s\n' "$samples"
"$python_bin" -u main.py routes "${args[@]}"
printf 'Results: %s\n' "$output"
