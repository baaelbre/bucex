#!/usr/bin/env bash
# Gallade: 25 full-record grid fits plus 50 earlier-origin hindcasts.
set -euo pipefail
cd -P "$(dirname "$0")"
tier=screen
if [[ $# -gt 0 && "$1" != --* ]]; then tier="$1"; shift; fi
export BUCEX_SCHEDULER="${BUCEX_SCHEDULER:-slurm}"
export VSC_CLUSTER="${VSC_CLUSTER:-gallade}"
export VSC_ARRAY_LIMIT="${VSC_ARRAY_LIMIT:-16}"
exec bash bash_scripts/submit.sh "$tier" sweetspot_hpc "$@"
