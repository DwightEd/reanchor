#!/usr/bin/env bash
set -euo pipefail
BASE=/share/home/tm902089733300000/a903202310/lys
cd "$BASE/research/reanchor"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=''
export PYTHONPATH=src
phase="${1:?sanity or full}"
case "$phase" in
 sanity) extra=(--limit 12); target=outputs/s10_structural_sanity_20260914_v1;;
 full) extra=(); target=outputs/s10_structural_features_20260914_v1;;
 *) exit 2;;
esac
exec "$BASE/conda_envs/research/bin/python" -m decoding.structural_export \
 --population outputs/ragtruth_population_20260912 --output "$target" "${extra[@]}"
