#!/usr/bin/env bash
# Full, sequential 1.8.8 manuscript queue. Execute from any directory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p results/serra_188_logs

run_logged() {
  local label="$1"
  shift
  "$@" 2>&1 | tee "results/serra_188_logs/${label}.log"
}
last_run() {
  local label="$1" path
  path="$(tail -n 1 "results/serra_188_logs/${label}.log" | tr -d '\r')"
  if [[ ! -f "${path}/config.json" && -f "${path}/../config.json" ]]; then
    path="${path}/.."
  fi
  if [[ ! -f "${path}/config.json" ]]; then
    echo "Cannot locate ${label} run from log: ${path}" >&2
    return 1
  fi
  printf '%s\n' "$path"
}

python -m pip install -e '.[plot,test]'
python -c 'import bucex; assert bucex.__version__ == "1.8.8"; print(bucex.__version__, bucex.__file__)'
python -m pytest -q
python -m research.seasonal.prepare --config research/seasonal/config/final.json --output results/serra_188_seasonal_data
python -m research.seasonal.preflight --config research/seasonal/config/final.json --output results/serra_188_final_plan
python -m research.seasonal.prior_effects --config research/seasonal/config/final.json
run_logged smoke python -u -m research.seasonal.fit --config research/seasonal/config/smoke.json

run_logged final python -u -m research.seasonal.fit --config research/seasonal/config/final.json
FINAL_RUN="$(last_run final)"
python -m research.seasonal.check_final --run "$FINAL_RUN"
python -m research.seasonal.dynamic_comparison --run "$FINAL_RUN" \
  --output results/serra_188_dynamic_comparison

# Full-record variants and both posterior and paired held-out assessments.
for specification in manuscript_sensitivity physical_sensitivity adequacy dependence_sensitivity; do
  python -m research.monthly.prior_assessment \
    --config "research/seasonal/config/${specification}.json" --stage plan \
    > "results/serra_188_logs/${specification}_plan.json"
  run_logged "$specification" python -u -m research.monthly.prior_assessment \
    --config "research/seasonal/config/${specification}.json" --stage all
done
PRIOR_RUN="$(last_run manuscript_sensitivity)"
PHYSICAL_RUN="$(last_run physical_sensitivity)"
ADEQUACY_RUN="$(last_run adequacy)"
DEPENDENCE_RUN="$(last_run dependence_sensitivity)"

# Dependence alternatives use the same 4 x (3000 warmup + 8000 retained) budget.
run_logged constant_copula python -u -m research.seasonal.fit \
  --config research/seasonal/config/constant_copula_full.json
run_logged independence python -u -m research.seasonal.fit \
  --config research/seasonal/config/independence_full.json

# Prospective 2019 forecast, followed by targeted risk sensitivity.
run_logged pre2019 python -u -m research.seasonal.fit \
  --config research/seasonal/config/pre2019.json
PRE2019_RUN="$(last_run pre2019)"
run_logged pre2019_sensitivity python -u -m research.monthly.sensitivity \
  --config research/seasonal/config/pre2019_sensitivity.json
PRE2019_SENSITIVITY="$(last_run pre2019_sensitivity)"

# Reviewer comment 5: seven nonoverlapping held-out five-year windows.
python -m research.seasonal.preflight --config research/seasonal/config/comment5.json \
  --output results/serra_188_comment5_plan
run_logged validation python -u -m research.monthly.validate \
  --config research/seasonal/config/comment5.json
VALIDATION_RUN="$(last_run validation)"

# Supplementary monthly full-record analysis and matched monthly/seasonal forecasts.
run_logged monthly python -u -m research.monthly.run \
  --config research/monthly/config/final.json
python -m research.seasonal.compare --config research/seasonal/config/compare_full.json --plan \
  > results/serra_188_logs/block_comparison_plan.txt
run_logged block_comparison python -u -m research.seasonal.compare \
  --config research/seasonal/config/compare_full.json
BLOCK_RUN="$(last_run block_comparison)"

# The main fit is strictly gated. Check numerical status of every supporting
# fit and fold as well, before treating their comparisons as manuscript evidence.
python -m research.seasonal.paper_gate \
  "$PRIOR_RUN" "$PHYSICAL_RUN" "$ADEQUACY_RUN" "$DEPENDENCE_RUN" \
  "$(last_run constant_copula)" "$(last_run independence)" \
  "$PRE2019_RUN" "$PRE2019_SENSITIVITY" "$VALIDATION_RUN" "$BLOCK_RUN"

python -m research.seasonal.manuscript_figures \
  --run "$FINAL_RUN" \
  --sensitivity "$PRIOR_RUN/sensitivity" "$ADEQUACY_RUN/sensitivity" \
  --validation "$VALIDATION_RUN/joint" \
  --block-comparison "$BLOCK_RUN" \
  --pre2019 "$PRE2019_RUN" "$PRE2019_SENSITIVITY" \
  --output results/serra_188_manuscript_figures
