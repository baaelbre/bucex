#!/usr/bin/env bash
# Login-node planning uses only stdlib; compute Python is never executed here.
set -euo pipefail
cd -P "$(dirname "$0")/.."
exec "${BUCEX_SUBMIT_PYTHON:-/usr/bin/python3}" job_scripts/submit.py "$@"
