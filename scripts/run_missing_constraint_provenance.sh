#!/usr/bin/env bash
set -euo pipefail
cd /share/home/tm902089733300000/a903202310/lys/research/reanchor
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=../graph:src
exec flock -n outputs/.o6_missing_constraint.lock \
  /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
  -u -m decoding.missing_constraint_provenance "$@"
