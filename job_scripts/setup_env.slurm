#!/bin/bash -l
set -euo pipefail
trap 'status=$?; printf "Environment setup failed: status=%s line=%s host=%s\n" "$status" "$LINENO" "${HOSTNAME:-unknown}" >&2; exit "$status"' ERR
printf 'BUCEX environment setup on %s\n' "${HOSTNAME:-unknown}"
cd -P "${BUCEX_PROJECT_ROOT:?}"
module swap "cluster/${VSC_CLUSTER:-gallade}"
module unload Python >/dev/null 2>&1 || true
module load "${BUCEX_PYTHON_MODULE:?}"
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
[[ ! -e "${BUCEX_VENV:?}" ]] || { echo 'Refusing to overwrite an existing environment directory' >&2; exit 2; }
printf 'Project: %s\nEnvironment: %s\n' "$PWD" "$BUCEX_VENV"
[[ -d "$(dirname "$BUCEX_VENV")" && -w "$(dirname "$BUCEX_VENV")" ]] || {
    echo 'Environment parent is not an existing writable compute-visible directory.' >&2; exit 2;
}
python -m venv "$BUCEX_VENV"
"$BUCEX_VENV/bin/python" -m pip install --upgrade pip
"$BUCEX_VENV/bin/python" -m pip install -e "$PWD[plot]"
"$BUCEX_VENV/bin/python" -c 'import sys,bucex,numpy,scipy,pandas,matplotlib; print(sys.executable,sys.version,bucex.__version__); assert bucex.__version__=="1.9.8.4"'
# Written only after successful installation/imports; used on subsequent nodes.
{
    printf '# Source only on a compute node for this cluster.\n'
    printf 'module swap %q\n' "cluster/${VSC_CLUSTER:-gallade}"
    printf 'module unload Python >/dev/null 2>&1 || true\n'
    printf 'module load %q\n' "$BUCEX_PYTHON_MODULE"
    printf 'export BUCEX_PYTHON=%q\n' "$BUCEX_VENV/bin/python"
} > "$BUCEX_VENV/environment.sh"
