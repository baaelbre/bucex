#!/usr/bin/env bash
# Gallade: 46 experiment groups / 111 fits (full record and earlier origins).
set -euo pipefail
cd -P "$(dirname "$0")"
tier=screen
if [[ $# -gt 0 && "$1" != --* ]]; then tier="$1"; shift; fi
export BUCEX_SCHEDULER="${BUCEX_SCHEDULER:-slurm}"
export VSC_CLUSTER="${VSC_CLUSTER:-gallade}"
export VSC_ARRAY_LIMIT="${VSC_ARRAY_LIMIT:-16}"
export VSC_SEPARATE_ARRAY_LIMIT="${VSC_SEPARATE_ARRAY_LIMIT:-4}"
exec bash bash_scripts/submit.sh "$tier" sweetspot_hpc "$@"
