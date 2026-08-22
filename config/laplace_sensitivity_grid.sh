#!/usr/bin/env bash

# Edit this one file to change the sensitivity grid. The two PBS submission
# files and the run-all shell helper source it.

SENS_SEEDS=(
  13081997
  13081998
  13081999
  13082000
  13082001
)

# Short profile names keep result paths manageable. Exact values are also
# recorded in every run_config.json.
SENS_PROFILE_NAMES=(base l02 t05 t12 s035)
SENS_LEVEL_SLAB=(0.03 0.02 0.03 0.03 0.03)
SENS_TREND_SLAB=(0.0008 0.0008 0.0005 0.0012 0.0008)
SENS_SEASON_SLAB=(0.05 0.05 0.05 0.05 0.035)
SENS_INITIAL_SEASON_SD=(0.50 0.50 0.50 0.35 0.35)

# Each inference task runs one chain. Chains are combined in the second array.
SENS_CHAINS=4
SENS_DRAWS=1000
SENS_WARMUP=1000
SENS_N_TIME=1000
SENS_PERIOD=4

# qsub array throttles used by run_laplace_sensitivity_all.sh.
SENS_MAX_CONCURRENT_FITS=12
SENS_MAX_CONCURRENT_COMBINES=5

SENS_RESULTS_ROOT="results/laplace_sensitivity"

SENS_N_SEEDS=${#SENS_SEEDS[@]}
SENS_N_PROFILES=${#SENS_PROFILE_NAMES[@]}
SENS_N_FIT_TASKS=$((SENS_N_SEEDS * SENS_N_PROFILES * SENS_CHAINS))
SENS_N_COMBINE_TASKS=$((SENS_N_SEEDS * SENS_N_PROFILES))

if (( ${#SENS_LEVEL_SLAB[@]} != SENS_N_PROFILES \
   || ${#SENS_TREND_SLAB[@]} != SENS_N_PROFILES \
   || ${#SENS_SEASON_SLAB[@]} != SENS_N_PROFILES \
   || ${#SENS_INITIAL_SEASON_SD[@]} != SENS_N_PROFILES )); then
  echo "Sensitivity profile arrays must all have the same length." >&2
  return 2 2>/dev/null || exit 2
fi
