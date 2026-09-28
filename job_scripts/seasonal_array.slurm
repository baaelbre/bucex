#!/bin/bash
set -euo pipefail
cd -P "${BUCEX_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?submit from release root}}"
export BUCEX_PROJECT_ROOT="$PWD"
exec bash -l job_scripts/run_task.sh
