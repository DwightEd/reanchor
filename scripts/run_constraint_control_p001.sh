#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 4 ]]; then
  echo "Usage: $0 [RAGTRUTH_ROOT] [OUTPUT_ROOT] [MODEL] [DEVICE]" >&2
  exit 2
fi

DEFAULT_DATA=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset
DEFAULT_MODEL=/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct
RAGTRUTH_ROOT=${1:-${RAGTRUTH_ROOT:-$DEFAULT_DATA}}
OUTPUT_ROOT=${2:-${OUTPUT_ROOT:-runs/p001_ragtruth_qa_llama31_8b}}
MODEL=${3:-${MODEL:-$DEFAULT_MODEL}}
DEVICE=${4:-${DEVICE:-cuda:0}}
TASK=${TASK:-QA}
SPLIT=${SPLIT:-train}
MAX_SAMPLES=${MAX_SAMPLES:-20}
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

if [[ ! -f "$RAGTRUTH_ROOT/source_info.jsonl" ]]; then
  echo "RAGTruth source file does not exist: $RAGTRUTH_ROOT/source_info.jsonl" >&2
  exit 2
fi
if [[ ! -f "$RAGTRUTH_ROOT/response.jsonl" ]]; then
  echo "RAGTruth response file does not exist: $RAGTRUTH_ROOT/response.jsonl" >&2
  exit 2
fi
if [[ "$MODEL" = /* && ! -d "$MODEL" ]]; then
  echo "Local model directory does not exist: $MODEL" >&2
  exit 2
fi
if [[ -f "$OUTPUT_ROOT/index.json" ]]; then
  echo "Refusing to overwrite completed run: $OUTPUT_ROOT/index.json" >&2
  exit 2
fi

python -m pip install -e . --no-deps
python -c 'import tqdm' >/dev/null
echo "RAGTruth: $RAGTRUTH_ROOT"
echo "Model: $MODEL"
echo "Task/split/sources: $TASK/$SPLIT/$MAX_SAMPLES"
echo "Output: $OUTPUT_ROOT"
python -u -m experiments.constraint_control.main \
  --input "$RAGTRUTH_ROOT" \
  --input-format ragtruth \
  --task "$TASK" \
  --split "$SPLIT" \
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
  --max-samples "$MAX_SAMPLES"
