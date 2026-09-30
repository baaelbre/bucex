#!/usr/bin/env bash
# Build a fresh environment on the target compute architecture, never on login.
set -euo pipefail
cd -P "$(dirname "$0")"
export BUCEX_PROJECT_ROOT="$PWD"
export VSC_CLUSTER="${VSC_CLUSTER:-gallade}"
export BUCEX_VENV="${BUCEX_VENV:-$PWD/bucex_env_gallade_py311_1981}"
export BUCEX_VENV="$("${BUCEX_SUBMIT_PYTHON:-/usr/bin/python3}" job_scripts/paths.py "$BUCEX_VENV")"
if [[ "$BUCEX_VENV" == /user/data/* ]]; then
    printf 'BUCEX_VENV still uses the login-only /user/data alias: %s\nUse a physical /kyukon/data path or a compute-visible scratch path.\n' "$BUCEX_VENV" >&2
    exit 2
fi
printf 'Physical project: %s\nPhysical environment: %s\n' "$PWD" "$BUCEX_VENV"
export BUCEX_PYTHON_MODULE="${BUCEX_PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
if [[ -e "$BUCEX_VENV" ]]; then
    if [[ -f "$BUCEX_VENV/environment.sh" && -x "$BUCEX_VENV/bin/python" ]]; then
        printf 'Environment already prepared: %s\n' "$BUCEX_VENV"
        exit 0
    fi
    printf 'Existing incomplete directory: %s. Choose a fresh BUCEX_VENV; this script will not overwrite it.\n' "$BUCEX_VENV" >&2
    exit 2
fi
mkdir -p job_scripts/logs
account=(); [[ -z "${VSC_PROJECT:-}" ]] || account=(--account="$VSC_PROJECT")
printf 'Preparing the environment on %s. This command waits for the setup job; logs: job_scripts/logs/bx1981_setup_*.log\n' "$VSC_CLUSTER"
sbatch --parsable --wait --clusters="$VSC_CLUSTER" "${account[@]}" --chdir="$PWD" \
    --job-name=bx1981_setup --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:30:00 \
    --export=ALL --output="$PWD/job_scripts/logs/bx1981_setup_%j.log" job_scripts/setup_env.slurm
[[ -f "$BUCEX_VENV/environment.sh" ]] || { echo 'Setup did not finish; inspect its log.' >&2; exit 2; }
printf 'Ready: %s\n' "$BUCEX_VENV"
