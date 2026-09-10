#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

input=${1:-data/pilot_events.jsonl}
output=${2:-outputs/constraint_control_pilot}
model=${3:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}
device=${4:-cuda:0}

python -u main.py \
  --input "$input" \
  --output "$output" \
  --model "$model" \
  --device "$device" \
  --dtype bfloat16
