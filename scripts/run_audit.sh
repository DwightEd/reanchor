#!/usr/bin/env bash
# Independent legacy mechanism audit, NOT G0 training.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
input=${1:-data/pilot_events.jsonl}
output=${2:-outputs/constraint_control_$(date +%Y%m%d_%H%M%S)_$$}
model=${3:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}
device=${4:-cuda:0}
exec "${PYTHON_BIN:-python}" -u main.py audit \
  --input "$input" --output "$output" --model "$model" \
  --device "$device" --dtype "${DTYPE:-bfloat16}"
