#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
base=/share/home/tm902089733300000/a903202310/lys
python_bin=${PYTHON_BIN:-$base/conda_envs/research/bin/python}
output=${OUTPUT_DIR:-$repo_root/outputs/mechanism_controls_20260912_v3}
samples=${SAMPLES_DIR:-$repo_root/outputs/samples_20260911_145421_235}
states=${STATES_DIR:-$repo_root/outputs/states_samples_20260911_145421_235}
export PYTHONPATH="$repo_root/src"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
"$python_bin" -u main.py mechanism \
  --model "${MODEL_PATH:-$base/models/Meta-Llama-3.1-8B-Instruct}" \
  --cases examples/binding_cases.jsonl --output "$output" \
  --samples "$samples" --states "$states" --device "${DEVICE:-cuda:0}"
