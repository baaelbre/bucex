#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${HERE}/_run_example.sh" examples/11_uccle_phi.py examples/config/phi/uccle/01_txx_stationary.json "$@"
