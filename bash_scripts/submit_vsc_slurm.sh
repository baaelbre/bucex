#!/usr/bin/env bash
# Native Slurm alternative; use the qsub launcher on HPC-UGent's qsub frontend.
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
if ! "$dry"; then command -v sbatch >/dev/null || { echo 'sbatch not found' >&2; exit 1; }; fi
mkdir -p job_scripts/logs
account=();partition=()
[[ -z "${VSC_PROJECT:-}" ]] || account=(--account="$VSC_PROJECT")
[[ -z "${VSC_PARTITION:-}" ]] || partition=(--partition="$VSC_PARTITION")
for resource in standard long; do
    count=$("$python_exe" -m research.seasonal.jobs --tier "$tier" --batch "$batch" --resource "$resource" --count)
    if (( count == 0 )); then continue; fi
    read -r cpus memory hours < <("$python_exe" -m research.seasonal.jobs --tier "$tier" --resource "$resource" --resource-info)
    cmd=(sbatch --parsable --chdir="$PWD" "${account[@]}" "${partition[@]}" --job-name="bx192_${tier}_${resource}"
         --array="1-${count}%${cap}" --nodes=1 --ntasks=1 --cpus-per-task="$cpus" --mem="${memory}G" --time="${hours}:00:00"
         --output="$PWD/job_scripts/logs/${tier}_${batch}_${resource}_%A_%a.log"
         --export="ALL,BUCEX_PROJECT_ROOT=$PWD,BUCEX_PYTHON=$python_exe,BUCEX_TIER=$tier,BUCEX_BATCH=$batch,BUCEX_RESOURCE=$resource,BUCEX_RESULTS_ROOT=$root"
         job_scripts/seasonal_array.slurm)
    if "$dry"; then printf '%q ' "${cmd[@]}"; printf '\n'; else "${cmd[@]}"; fi
done
