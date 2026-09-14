#!/usr/bin/env bash
set -euo pipefail
cd /share/home/tm902089733300000/a903202310/lys/research/reanchor
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=src
exec /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
  -m decoding.ownership_multilayer "$@"
