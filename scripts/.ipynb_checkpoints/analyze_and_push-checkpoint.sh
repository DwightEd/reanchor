#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
if (( $# > 1 )); then
  printf 'Usage: bash scripts/analyze_and_push.sh [existing_samples_directory]\n' >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  printf 'Commit tracked code changes before publishing results, so results match committed code.\n' >&2
  exit 2
fi
branch=$(git symbolic-ref --quiet --short HEAD)
git remote get-url origin >/dev/null
result_path="results/routes_$(date -u +%Y%m%d_%H%M%S)_$$"
export OUTPUT_DIR="$repo_root/$result_path"

bash scripts/analyze_cases.sh "$@"
git add -- "$result_path"
git commit -m "Add route analysis results $(basename "$result_path")" -- "$result_path"
git push origin "HEAD:refs/heads/$branch"
printf 'Pushed results: %s (%s)\n' "$result_path" "$branch"
