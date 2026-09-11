#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

python_bin=${PYTHON_BIN:-python}
model=${MODEL_PATH:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}
ragtruth=${RAGTRUTH_DIR:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset}
output=${OUTPUT_DIR:-"$repo_root/outputs/samples_$(date +%Y%m%d_%H%M%S)_$$"}

"$python_bin" -u main.py sample \
  --dataset "$ragtruth" \
  --model "$model" \
  --source-ids 14304 14315 14325 14375 \
  --seeds 0 1 2 3 \
  --max-new-tokens 512 \
  --temperature 0.7 --top-p 0.9 \
  --device "${DEVICE:-cuda:0}" --dtype "${DTYPE:-bfloat16}" \
  --output "$output"

"$python_bin" -u main.py inspect \
  --samples "$output" \
  --output "$output/reading"

printf 'Results: %s\n' "$output"
