#!/usr/bin/env bash
# Collection also runs on the compute architecture; no numerical imports on login.
set -euo pipefail
cd -P "$(dirname "$0")"
export BUCEX_PROJECT_ROOT="$PWD"
export BUCEX_TIER="${1:-screen}"
if [[ $# -gt 0 ]]; then shift; fi
export BUCEX_BATCH=sweetspot_hpc
if [[ $# -gt 0 && "$1" != --* ]]; then export BUCEX_BATCH="$1"; shift; fi
export BUCEX_COLLECT_FIGURES=0 BUCEX_REQUIRE_COMPLETE=0
export BUCEX_FINISH=0
[[ "$BUCEX_BATCH" != sweetspot* ]] || export BUCEX_FINISH=1
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
case "$BUCEX_BATCH" in posterior|reference|comparison|experiments|influence|validation|validation10|all|sweetspot|sweetspot_hpc|sweetspot_posterior|sweetspot_validation|sweetspot_long|sweetspot_fixed) ;; *) echo 'unknown batch' >&2; exit 2;; esac
export BUCEX_RESULTS_ROOT="${BUCEX_RESULTS_ROOT:-$PWD/results/serra_1981}"
export BUCEX_PYTHON="${BUCEX_PYTHON:-$PWD/bucex_env_gallade_py311_1981/bin/python}"
default_setup="$PWD/bucex_env_gallade_py311_1981/environment.sh"
if [[ -z "${BUCEX_ENV_SETUP:-}" && -f "$default_setup" ]]; then export BUCEX_ENV_SETUP="$default_setup"; fi
export BUCEX_RESULTS_ROOT="$("${BUCEX_SUBMIT_PYTHON:-/usr/bin/python3}" job_scripts/paths.py "$BUCEX_RESULTS_ROOT")"
export BUCEX_PYTHON="$("${BUCEX_SUBMIT_PYTHON:-/usr/bin/python3}" job_scripts/paths.py --executable "$BUCEX_PYTHON")"
if [[ -n "${BUCEX_ENV_SETUP:-}" ]]; then export BUCEX_ENV_SETUP="$("${BUCEX_SUBMIT_PYTHON:-/usr/bin/python3}" job_scripts/paths.py "$BUCEX_ENV_SETUP")"; fi
mkdir -p "$BUCEX_RESULTS_ROOT/scheduler_logs"
extra=()
[[ -z "${VSC_CLUSTER:-}" ]] || extra+=(--clusters="$VSC_CLUSTER")
[[ -z "${VSC_PROJECT:-}" ]] || extra+=(--account="$VSC_PROJECT")
[[ -z "${VSC_PARTITION:-}" ]] || extra+=(--partition="$VSC_PARTITION")
command=(sbatch --parsable "${extra[@]}" --chdir="$PWD" --job-name=bx1981_collect
    --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=12G --time=02:00:00 --export=ALL
    --output="$BUCEX_RESULTS_ROOT/scheduler_logs/bx1981_collect_%j.log" job_scripts/collect.slurm)
printf '%q ' "${command[@]}"; printf '\n'
if [[ "$dry" == 0 ]]; then "${command[@]}"; fi
