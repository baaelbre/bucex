#!/usr/bin/env bash
# Submit independent PBS array tasks and all five full manuscript fits.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v qsub >/dev/null || { echo 'qsub was not found on this host' >&2; exit 1; }
python_exe="${BUCEX_PYTHON:-python3}"
"$python_exe" -c 'import bucex, scipy, pandas, matplotlib; print("BUCEX", bucex.__version__)'
mkdir -p job_scripts/logs
count="$("$python_exe" -m research.seasonal.jobs --count)"
concurrent="${BUCEX_MAX_JOBS:-8}"
mode="${1:-paper}"
if [[ "$mode" != paper && "$mode" != screen ]]; then
  echo 'Usage: bash bash_scripts/submit_biobot.sh [paper|screen]' >&2; exit 2
fi
if [[ "$mode" == paper ]]; then
  # Each array element retains the declared four-chain study budget.
  array_id="$(qsub -t "1-${count}%${concurrent}" \
    -o "$PWD/job_scripts/logs" \
    -v "BUCEX_PROJECT_ROOT=$PWD,BUCEX_PYTHON=$python_exe,BUCEX_TIER=paper" \
    job_scripts/seasonal_array.pbs)"
else
  # Triage only; these shorter two-chain fits cannot support manuscript claims.
  array_id="$(qsub -t "1-${count}%${concurrent}" -l walltime=03:00:00 \
    -o "$PWD/job_scripts/logs" \
    -v "BUCEX_PROJECT_ROOT=$PWD,BUCEX_PYTHON=$python_exe,BUCEX_TIER=screen" \
    job_scripts/seasonal_array.pbs)"
fi
echo "$mode array: $array_id ($count one-fit tasks, cap $concurrent)"
if [[ "$mode" == paper ]]; then
  for task in final_reference final_pre2019 final_constant_copula final_independence final_monthly_supplement; do
    id="$(qsub -o "$PWD/job_scripts/logs" \
      -v "BUCEX_PROJECT_ROOT=$PWD,BUCEX_PYTHON=$python_exe,BUCEX_TASK=$task" \
      job_scripts/seasonal_long.pbs)"
    echo "$task: $id"
  done
fi
