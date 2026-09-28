#!/usr/bin/env bash
# Both schedulers enter through this non-login shell and retain submission environment.
set -euo pipefail
trap 'status=$?; printf "BUCEX startup/task failed: status=%s line=%s host=%s\n" "$status" "$LINENO" "${HOSTNAME:-unknown}" >&2; exit "$status"' ERR
cd -P "${BUCEX_PROJECT_ROOT:?missing project root}"
source bash_scripts/runtime.sh
index="${SLURM_ARRAY_TASK_ID:-${PBS_ARRAYID:-${PBS_ARRAY_INDEX:-}}}"
[[ "$index" =~ ^[1-9][0-9]*$ ]] || { echo 'Missing or invalid scheduler array index' >&2; exit 2; }
printf 'BUCEX task: host=%s root=%s tier=%s batch=%s resource=%s index=%s python=%s\n' \
    "${HOSTNAME:-unknown}" "$PWD" "${BUCEX_TIER:?missing tier}" "${BUCEX_BATCH:?missing batch}" \
    "${BUCEX_RESOURCE:?missing resource}" "$index" "$python_exe"
"$python_exe" -c 'import sys, bucex; print("Python:", sys.executable, "BUCEX:", bucex.__version__, flush=True)'
"$python_exe" -u -m research.seasonal.jobs --tier "$BUCEX_TIER" --batch "$BUCEX_BATCH" \
    --resource "$BUCEX_RESOURCE" --root "$root" --index "$index"
