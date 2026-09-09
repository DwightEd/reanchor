#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 CAPTURE_ROOT OUTPUT_ROOT [DEVICE]" >&2
  exit 2
fi

CAPTURE_ROOT=$1
OUTPUT_ROOT=$2
DEVICE=${3:-cuda:0}
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

cd "$REPO_ROOT"
python -m pip install -e . --no-deps
python -m reanchor run \
  --capture "$CAPTURE_ROOT" \
  --output "$OUTPUT_ROOT" \
  --device "$DEVICE" \
  --window 10 \
  --local-floor 0.50 \
  --site-gain-floor 0.10 \
  --broad-gain-floor 0.05 \
  --broad-head-fraction 0.25 \
  --position-bins 4 \
  --alpha 0.05 \
  --min-calibration-sources 32 \
  --query-chunk 8 \
  --event-batch 2 \
  --bootstrap 1000
