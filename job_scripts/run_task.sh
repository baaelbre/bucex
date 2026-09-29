#!/bin/bash
set -euo pipefail
trap 'status=$?; printf "BUCEX startup/task failed: status=%s line=%s host=%s\n" "$status" "$LINENO" "${HOSTNAME:-unknown}" >&2; exit "$status"' ERR
printf 'BUCEX 1.9.6.1 starting on %s; Python requested: %s\n' "${HOSTNAME:-unknown}" "${BUCEX_PYTHON:-unset}"
cd -P "${BUCEX_PROJECT_ROOT:?missing project root}"
source bash_scripts/compute_environment.sh
source bash_scripts/runtime.sh
index="${SLURM_ARRAY_TASK_ID:-${PBS_ARRAYID:-${PBS_ARRAY_INDEX:-}}}"
[[ "$index" =~ ^[1-9][0-9]*$ ]] || { echo 'Missing or invalid scheduler array index' >&2; exit 2; }
args=(); [[ "${BUCEX_RETRY_FAILED:-0}" != 1 ]] || args+=(--retry-failed)
exec "$python_exe" -u -m research.seasonal.bundles --tier "${BUCEX_TIER:?}" --batch "${BUCEX_BATCH:?}" \
    --resource "${BUCEX_RESOURCE:?}" --root "$root" --index "$index" "${args[@]}"
