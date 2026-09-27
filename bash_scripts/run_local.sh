#!/usr/bin/env bash
# Explicit workstation / allocated-node alternative with bounded fit concurrency.
set -euo pipefail
cd "$(dirname "$0")/.."
python_exe="${BUCEX_PYTHON:-python3}";parallel="${BUCEX_MAX_JOBS:-4}"
tier="${1:-screen}";batch="${2:-posterior}";root="${BUCEX_RESULTS_ROOT:-results/serra_190_parallel}"
[[ "$parallel" =~ ^[1-9][0-9]*$ ]] || { echo 'BUCEX_MAX_JOBS must be positive' >&2; exit 2; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$python_exe" -m research.seasonal.jobs --tier "$tier" --batch "$batch" --verify --root "$root"
mkdir -p job_scripts/logs
"$python_exe" -m research.seasonal.jobs --tier "$tier" --batch "$batch" --list | awk '{print $1}' | \
  xargs -P "$parallel" -I '{}' bash -c '
    index="$1";python_exe="$2";tier="$3";batch="$4";root="$5"
    "$python_exe" -u -m research.seasonal.jobs --tier "$tier" --batch "$batch" --index "$index" --root "$root" \
      >"job_scripts/logs/local_${tier}_${batch}_${index}.log" 2>&1
  ' _ '{}' "$python_exe" "$tier" "$batch" "$root"
