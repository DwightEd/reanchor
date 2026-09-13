#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
base=/share/home/tm902089733300000/a903202310/lys
python_bin=${PYTHON_BIN:-$base/conda_envs/research/bin/python}
graph_root=${GRAPH_ROOT:-$repo_root/../graph}
export PYTHONPATH="$repo_root/src:$graph_root${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
output=${OUTPUT_DIR:-$repo_root/outputs/binding_validation_v3_20260912}
args=()
if [[ -n ${LIMIT_TEMPLATES:-} ]]; then args+=(--limit "$LIMIT_TEMPLATES"); fi
"$python_bin" -u -m decoding.binding_validation \
  --model "${MODEL_PATH:-$base/models/Meta-Llama-3.1-8B-Instruct}" \
  --cases examples/binding_cases.jsonl --output "$output" \
  --layouts clean reverse distractor misbound \
  --device "${DEVICE:-cuda:0}" "${args[@]}"
