#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${HERE}/_run_example.sh" examples/06_uccle_pgas.py examples/config/uccle.json "$@"
