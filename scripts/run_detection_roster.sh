#!/usr/bin/env bash
set -euo pipefail

shared=/share/home/tm902089733300000/a903202310/lys
project="$shared/research/reanchor"
python="$shared/conda_envs/research/bin/python"
output="${1:?Usage: bash scripts/run_detection_roster.sh /absolute/fresh/output}"
export PYTHONPATH="$project/src"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=""
cd "$project"

"$python" -u -m decoding.detection_roster \
  --dataset "$shared/data/RAGTruth/dataset" \
  --model "$shared/models/Meta-Llama-3.1-8B-Instruct" \
  --output "$output" \
  --protocol "$shared/codex/research/refine-logs/R04_PROTOCOL.md" \
  --seed 20260913 \
  --exclude-roster "$shared/research/graph/outputs/native_audit_design_20260913/inputs.jsonl" \
  --exclude-roster "$shared/research/reanchor/outputs/samples_20260911_145421_235/samples.jsonl" \
  --exclude-roster "$shared/research/graph/outputs/typed_native_v2_20260913/settings.json" \
  --exclude-roster "$shared/research/graph/outputs/grounded_graph_natural_features_v1_20260913/entries.json" \
  2>&1 | tee "${output}.log"
