#!/usr/bin/env bash
# Compatibility entry name now runs the NEW G0 pipeline.
# The old factorial experiment is explicitly available as run_audit.sh.
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/run_g0.sh" "$@"
