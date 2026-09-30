#!/bin/bash
set -euo pipefail
trap 'status=$?; printf "BUCEX startup/task failed: status=%s line=%s host=%s\n" "$status" "$LINENO" "${HOSTNAME:-unknown}" >&2; exit "$status"' ERR
printf 'BUCEX 1.9.8.3 starting on %s; Python requested: %s\n' "${HOSTNAME:-unknown}" "${BUCEX_PYTHON:-unset}"
cd -P "${BUCEX_PROJECT_ROOT:?missing project root}"
source bash_scripts/compute_environment.sh
source bash_scripts/runtime.sh
index="${SLURM_ARRAY_TASK_ID:-${PBS_ARRAYID:-${PBS_ARRAY_INDEX:-}}}"
[[ "$index" =~ ^[1-9][0-9]*$ ]] || { echo 'Missing or invalid scheduler array index' >&2; exit 2; }
args=(); [[ "${BUCEX_RETRY_FAILED:-0}" != 1 ]] || args+=(--retry-failed)
runner=research.seasonal.bundles
# Shared arrays have one fit per group; private arrays bundle six response fits.
if [[ "${BUCEX_BATCH:?}" == sweetspot* && "${BUCEX_RESOURCE:?}" == shared* ]]; then
    runner=research.seasonal.jobs
    mkdir -p "$root/$BUCEX_TIER/logs"
    task_id="$("$python_exe" -c 'import sys; from research.seasonal.job_plan import plan; g=plan(*sys.argv[1:4])[int(sys.argv[4])-1]; assert len(g["task_ids"])==1; print(g["task_ids"][0])' "$BUCEX_TIER" "$BUCEX_BATCH" "$BUCEX_RESOURCE" "$index")"
    exec > >(tee -a "$root/$BUCEX_TIER/logs/$task_id.log") 2>&1
fi
exec "$python_exe" -u -m "$runner" --tier "${BUCEX_TIER:?}" --batch "${BUCEX_BATCH:?}" \
    --resource "${BUCEX_RESOURCE:?}" --root "$root" --index "$index" "${args[@]}"
