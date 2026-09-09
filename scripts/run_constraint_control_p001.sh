#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 INPUT_JSONL OUTPUT_ROOT MODEL [DEVICE]" >&2
  exit 2
fi

INPUT_JSONL=$1
OUTPUT_ROOT=$2
MODEL=$3
DEVICE=${4:-cuda:0}
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

cd "$REPO_ROOT"
python -m pip install -e . --no-deps
python -u -m experiments.constraint_control.main \
  --input "$INPUT_JSONL" \
  --output "$OUTPUT_ROOT" \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype bfloat16 \
  --seeds 0 1 2 \
  --max-new-tokens 64 \
  --temperature 0.7 \
  --top-p 0.9 \
  --top-k 20 \
  --trace-top-k 20 \
  --max-samples 20
