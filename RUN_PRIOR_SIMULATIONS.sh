#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source bash_scripts/runtime.sh
tier="${1:-screen}"
if [[ $# -gt 0 ]]; then shift; fi
exec "$python_exe" -u -m research.seasonal.prior_simulations --root "$root" --tier "$tier" --suite sweetspot --draws "$([[ "$tier" == paper ]] && echo 5000 || echo 1000)" "$@"
