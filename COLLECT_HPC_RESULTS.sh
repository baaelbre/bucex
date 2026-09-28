#!/usr/bin/env bash
# Collection also runs on the compute architecture; no numerical imports on login.
set -euo pipefail
cd -P "$(dirname "$0")"
export BUCEX_PROJECT_ROOT="$PWD"
export BUCEX_TIER="${1:-screen}"
if [[ $# -gt 0 ]]; then shift; fi
export BUCEX_BATCH=experiments
if [[ $# -gt 0 && "$1" != --* ]]; then export BUCEX_BATCH="$1"; shift; fi
export BUCEX_COLLECT_FIGURES=0 BUCEX_REQUIRE_COMPLETE=0
dry=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --figures) export BUCEX_COLLECT_FIGURES=1;;
        --require-complete) export BUCEX_REQUIRE_COMPLETE=1;;
        --dry-run) dry=1;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2;;
    esac
    shift
done
case "$BUCEX_TIER" in screen|paper) ;; *) echo 'tier must be screen or paper' >&2; exit 2;; esac
case "$BUCEX_BATCH" in posterior|reference|pre2019|experiments|validation|validation10|all) ;; *) echo 'unknown batch' >&2; exit 2;; esac
export BUCEX_RESULTS_ROOT="${BUCEX_RESULTS_ROOT:-$PWD/results/serra_193_parallel}"
export BUCEX_PYTHON="${BUCEX_PYTHON:-$PWD/bucex_env_gallade_py311_193/bin/python}"
default_setup="$PWD/bucex_env_gallade_py311_193/environment.sh"
if [[ -z "${BUCEX_ENV_SETUP:-}" && -f "$default_setup" ]]; then export BUCEX_ENV_SETUP="$default_setup"; fi
mkdir -p job_scripts/logs
extra=()
[[ -z "${VSC_CLUSTER:-}" ]] || extra+=(--clusters="$VSC_CLUSTER")
[[ -z "${VSC_PROJECT:-}" ]] || extra+=(--account="$VSC_PROJECT")
[[ -z "${VSC_PARTITION:-}" ]] || extra+=(--partition="$VSC_PARTITION")
command=(sbatch --parsable "${extra[@]}" --chdir="$PWD" --job-name=bx193_collect
    --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=12G --time=02:00:00 --export=ALL
    --output="$PWD/job_scripts/logs/bx193_collect_%j.log" job_scripts/collect.slurm)
printf '%q ' "${command[@]}"; printf '\n'
if [[ "$dry" == 0 ]]; then "${command[@]}"; fi
