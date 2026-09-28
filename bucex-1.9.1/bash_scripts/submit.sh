#!/usr/bin/env bash
# Prefer native Slurm where available. BUCEX_SCHEDULER=slurm|pbs overrides auto.
set -euo pipefail
cd -P "$(dirname "$0")/.."
scheduler="${BUCEX_SCHEDULER:-auto}"
if [[ "$scheduler" == auto ]]; then
    if command -v sbatch >/dev/null; then scheduler=slurm
    elif command -v qsub >/dev/null; then scheduler=pbs
    else echo 'No scheduler found. Set BUCEX_SCHEDULER=slurm for a dry run, or use run_local.sh on biobot.' >&2; exit 2
    fi
fi
case "$scheduler" in
    slurm) exec bash bash_scripts/submit_vsc_slurm.sh "$@" ;;
    pbs) exec bash bash_scripts/submit_vsc.sh "$@" ;;
    *) echo 'BUCEX_SCHEDULER must be auto, slurm or pbs' >&2; exit 2 ;;
esac
