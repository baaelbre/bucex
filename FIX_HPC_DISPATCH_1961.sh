#!/usr/bin/env bash
# Run from the BUCEX 1.9.6.1 checkout. Existing jobs and results are left intact.
set -euo pipefail
echo "This hotfix is only for the original 1.9.6.1 source. BUCEX 1.9.8.3 already includes shared/private dispatch; use RUN_SWEETSPOT_HPC.sh." >&2
exit 2
[[ -f research/seasonal/config/sweetspot.json ]] || { echo 'Run this from your BUCEX 1.9.6.1 project directory.' >&2; exit 2; }
: "${BUCEX_PYTHON:?Keep the working Gallade BUCEX_PYTHON setting.}"
: "${BUCEX_ENV_SETUP:?Keep the working Gallade BUCEX_ENV_SETUP setting.}"
[[ -r "$BUCEX_ENV_SETUP" ]] || { echo 'BUCEX_ENV_SETUP is not readable.' >&2; exit 2; }
case "${1:-}" in ''|--dry-run) ;; *) echo 'Only --dry-run is supported.' >&2; exit 2;; esac
cd -P .
fix_dir="$(mktemp -d "${PWD}_hpcfix_XXXXXX")"
/usr/bin/python3 - "$fix_dir" <<'PY_COPY'
from pathlib import Path
import shutil,sys
source=Path.cwd();target=Path(sys.argv[1])
for name in ('bucex','research','data','job_scripts','bash_scripts'):
    shutil.copytree(source/name,target/name,ignore=shutil.ignore_patterns('__pycache__','logs','.pytest_cache'))
for p in source.iterdir():
    if p.is_file() and (p.name.startswith(('RUN_','COLLECT_','SETUP_')) or p.name in ('pyproject.toml','README.md','LICENSE')):
        shutil.copy2(p,target/p.name)
PY_COPY
cd -P "$fix_dir"
cat > job_scripts/run_task.sh <<'SH_FIXED_RUNNER'
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
runner=research.seasonal.bundles
# Every focused array element is one pooled fit; its index is a task index.
# Use the direct runner (also used on BIOBOT), not the legacy private-fit bundler.
if [[ "${BUCEX_BATCH:?}" == sweetspot* ]]; then
    runner=research.seasonal.jobs
    mkdir -p "$root/$BUCEX_TIER/logs"
    task_id="$("$python_exe" -c 'import sys; from research.seasonal.job_plan import plan; g=plan(*sys.argv[1:4])[int(sys.argv[4])-1]; assert len(g["task_ids"])==1; print(g["task_ids"][0])' "$BUCEX_TIER" "$BUCEX_BATCH" "$BUCEX_RESOURCE" "$index")"
    exec > >(tee -a "$root/$BUCEX_TIER/logs/$task_id.log") 2>&1
fi
exec "$python_exe" -u -m "$runner" --tier "${BUCEX_TIER:?}" --batch "${BUCEX_BATCH:?}" \
    --resource "${BUCEX_RESOURCE:?}" --root "$root" --index "$index" "${args[@]}"
SH_FIXED_RUNNER
bash -n job_scripts/run_task.sh
results_base="${BUCEX_RESULTS_ROOT:-$PWD/results/serra_1961}"
mkdir -p "$(dirname "$results_base")"
export BUCEX_RESULTS_ROOT="$(mktemp -d "${results_base}_hpcfix_XXXXXX")"
export VSC_CLUSTER=gallade VSC_PROJECT="${VSC_PROJECT:-gvo00048}"
export BUCEX_SCHEDULER=slurm VSC_ARRAY_LIMIT="${VSC_ARRAY_LIMIT:-16}"
unset VSC_PARTITION
{
    printf 'cd -P %q\n' "$PWD"
    for key in BUCEX_PYTHON BUCEX_ENV_SETUP BUCEX_RESULTS_ROOT VSC_CLUSTER VSC_PROJECT BUCEX_SCHEDULER VSC_ARRAY_LIMIT; do
        printf 'export %s=%q\n' "$key" "${!key}"
    done
    printf 'unset VSC_PARTITION\n'
} > HPC_FIX_ENV.sh
printf 'Isolated checkout: %s\nResults and logs: %s\nResume environment: %s/HPC_FIX_ENV.sh\n' "$PWD" "$BUCEX_RESULTS_ROOT" "$PWD"
bash RUN_SWEETSPOT_HPC.sh --dry-run
if [[ "${1:-}" != --dry-run ]]; then
    bash RUN_SWEETSPOT_HPC.sh
fi
