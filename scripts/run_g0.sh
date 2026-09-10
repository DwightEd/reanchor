#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

python_bin=${PYTHON_BIN:-python}
if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  exec "$python_bin" main.py pipeline --help
fi
for option in "$@"; do
  if [[ "$option" == "--output" || "$option" == --output=* ]]; then
    printf 'Use OUTPUT_DIR=... so output and log paths stay aligned.\n' >&2
    exit 2
  fi
done

model=${MODEL_PATH:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}
ragtruth=${RAGTRUTH_DIR:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset}
output=${OUTPUT_DIR:-"$repo_root/outputs/g0_$(date +%Y%m%d_%H%M%S)_$$"}
case "$output" in
  /*) ;;
  *) output="$repo_root/$output" ;;
esac
log="${output}.log"
if [[ -e "$output" || -e "$log" ]]; then
  printf 'Refusing an existing output/log: %s\n' "$output" >&2
  exit 2
fi
mkdir -p "$(dirname "$output")"

args=(
  pipeline --model "$model" --output "$output"
  --device "${DEVICE:-cuda:0}" --dtype "${DTYPE:-bfloat16}"
  --sources "${PROGRAM_SOURCES:-64}" --records "${RECORDS:-6}"
  --epochs "${EPOCHS:-10}" --batch-size "${BATCH_SIZE:-4}"
  --width "${WIDTH:-128}" --blocks "${BLOCKS:-2}"
  --learning-rate "${LEARNING_RATE:-0.0003}"
  --candidate-chunk "${CANDIDATE_CHUNK:-256}" --max-tokens "${MAX_TOKENS:-4096}"
  --seed "${SEED:-42}" --bootstrap "${BOOTSTRAP:-200}"
)
if [[ ${PROGRAM_ONLY:-0} != "1" ]]; then
  args+=(--ragtruth "$ragtruth" --ragtruth-sources "${RAGTRUTH_SOURCES:-32}")
fi
if [[ -n ${GENERATOR_FILTER:-} ]]; then
  args+=(--generator "$GENERATOR_FILTER")
fi
if [[ -n ${EXCLUDE_SOURCES:-} ]]; then
  args+=(--exclude-sources "$EXCLUDE_SOURCES")
fi
if [[ ${ALLOW_BUSY_GPU:-0} == "1" ]]; then
  args+=(--allow-busy-gpu)
fi

printf 'Method: G0 conditional binding (not the legacy factorial audit)\n'
printf 'Output: %s\nLog: %s\n' "$output" "$log"
printf 'Program sources: %s; natural source limit: %s\n' "${PROGRAM_SOURCES:-64}" "${RAGTRUTH_SOURCES:-32}"
trap 'status=$?; printf "FAILED (exit %s). Keep output and inspect: %s\n" "$status" "$log" >&2; exit "$status"' ERR
"$python_bin" -u main.py "${args[@]}" "$@" 2>&1 | tee "$log"
printf '\nCompleted. Program binding: %s/scores/program/test/program_metrics.json\n' "$output"
if [[ ${PROGRAM_ONLY:-0} != "1" ]]; then
  printf 'Natural first-error metrics: %s/evaluation/summary.json\n' "$output"
fi
