#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
python_bin=${PYTHON_BIN:-python}

if (( $# > 1 )); then
  printf 'Usage: bash scripts/revisits_and_push.sh [existing_samples_directory]\n' >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  printf 'Commit tracked code changes before publishing results.\n' >&2
  exit 2
fi
branch=$(git symbolic-ref --quiet --short HEAD)
git remote get-url origin >/dev/null
samples=${1:-${SAMPLES_DIR:-}}
if [[ -z "$samples" ]]; then
  samples=$("$python_bin" - <<'PY'
from pathlib import Path

captures = list(Path("outputs").glob("samples_*/samples.jsonl"))
if not captures:
    raise SystemExit("Pass the existing samples directory; no saved generation was found.")
print(max(captures, key=lambda p: p.stat().st_mtime_ns).parent)
PY
  )
fi

result_path="results/revisits_$(date -u +%Y%m%d_%H%M%S)_$$"
printf 'Existing samples: %s\n' "$samples"
"$python_bin" -u main.py revisits --samples "$samples" --output "$result_path" \
  --window 16 --quantile 0.95 --context 4
cp "$samples/prompts.jsonl" "$result_path/prompts.jsonl"
cp "$samples/settings.json" "$result_path/sampling.json"
git add -- "$result_path"
git commit -m "Add automatic revisit analysis $(basename "$result_path")" -- "$result_path"
git push origin "HEAD:refs/heads/$branch"
printf 'Pushed results: %s (%s)\n' "$result_path" "$branch"
