#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${HERE}/_run_example.sh" examples/12_uccle_gaussian.py examples/config/uccle_gaussian/01_txm.json "$@"
