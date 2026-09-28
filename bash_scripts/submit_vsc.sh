#!/usr/bin/env bash
# HPC-UGent qsub frontend. Default posterior batch excludes validation.
set -euo pipefail
cd -P "$(dirname "$0")/.."
tier="${1:-screen}"; if (( $# )); then shift; fi
batch=posterior
if (( $# )) && [[ "$1" != --dry-run ]]; then batch="$1"; shift; fi
dry=false
if (( $# )); then [[ "$1" == --dry-run && $# == 1 ]] || { echo 'Use [screen|paper] [posterior|reference|pre2019|validation|validation10|all] [--dry-run]' >&2; exit 2; }; dry=true; fi
source bash_scripts/runtime.sh
cap="${VSC_ARRAY_LIMIT:-8}"
[[ "$cap" =~ ^[1-9][0-9]*$ ]] || { echo 'VSC_ARRAY_LIMIT must be a positive integer' >&2; exit 2; }
"$python_exe" -m research.seasonal.jobs --tier "$tier" --batch "$batch" --verify --root "$root"
if ! "$dry"; then command -v qsub >/dev/null || { echo 'qsub not found; use run_local.sh or the native Slurm launcher on the appropriate host' >&2; exit 1; }; fi
mkdir -p job_scripts/logs
account=(); [[ -z "${VSC_PROJECT:-}" ]] || account=(-A "$VSC_PROJECT")
for resource in standard long; do
    count=$("$python_exe" -m research.seasonal.jobs --tier "$tier" --batch "$batch" --resource "$resource" --count)
    if (( count == 0 )); then continue; fi
    read -r cpus memory hours < <("$python_exe" -m research.seasonal.jobs --tier "$tier" --resource "$resource" --resource-info)
    cmd=(qsub -V "${account[@]}" -N "bx192_${tier:0:1}_${resource:0:3}" -t "1-${count}%${cap}"
         -l "nodes=1:ppn=${cpus},mem=${memory}gb,walltime=${hours}:00:00" -o "$PWD/job_scripts/logs" -e "$PWD/job_scripts/logs"
         -v "BUCEX_PROJECT_ROOT=$PWD,BUCEX_PYTHON=$python_exe,BUCEX_TIER=$tier,BUCEX_BATCH=$batch,BUCEX_RESOURCE=$resource,BUCEX_RESULTS_ROOT=$root"
         job_scripts/seasonal_array.pbs)
    if "$dry"; then printf '%q ' "${cmd[@]}"; printf '\n'; else "${cmd[@]}"; fi
done
