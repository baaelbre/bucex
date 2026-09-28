#!/usr/bin/env bash
set -euo pipefail
cd -P "$(dirname "$0")"
exec bash bash_scripts/submit.sh screen all "$@"
