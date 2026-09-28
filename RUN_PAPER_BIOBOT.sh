#!/usr/bin/env bash
set -euo pipefail
cd -P "$(dirname "$0")"
batch=all
if [[ $# -gt 0 && "$1" != --* ]]; then batch="$1"; shift; fi
exec bash RUN_OVERNIGHT_BIOBOT.sh --tier paper --batch "$batch" "$@"
