#!/usr/bin/env bash

# Shared PBS-array contract for example 08. Scenario keys are colon-separated
# because commas delimit variables in ``qsub -v``.
LMH_DEFAULT_SCENARIO_KEYS="stationary:linear:random_walk:llt:dynamic_season:llt_season"

laplace_mh_validate_positive_integer() {
  local name="${1:?name is required}"
  local value="${2:-}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer; received ${value:-<empty>}." >&2
    return 2
  fi
}

laplace_mh_parse_scenarios() {
  local scenario_keys="${1:-${LMH_DEFAULT_SCENARIO_KEYS}}"
  IFS=: read -r -a LMH_SCENARIOS <<< "${scenario_keys}"
  if (( ${#LMH_SCENARIOS[@]} == 0 )); then
    echo "At least one scenario key is required." >&2
    return 2
  fi

  local seen=":"
  local scenario
  for scenario in "${LMH_SCENARIOS[@]}"; do
    case "${scenario}" in
      stationary|linear|random_walk|llt|dynamic_season|llt_season) ;;
      *)
        echo "Unknown Laplace-MH scenario key: ${scenario}." >&2
        return 2
        ;;
    esac
    if [[ "${seen}" == *":${scenario}:"* ]]; then
      echo "Duplicate Laplace-MH scenario key: ${scenario}." >&2
      return 2
    fi
    seen+="${scenario}:"
  done
  LMH_SCENARIO_KEYS="$(IFS=:; echo "${LMH_SCENARIOS[*]}")"
  LMH_N_SCENARIOS="${#LMH_SCENARIOS[@]}"
}

laplace_mh_map_task() {
  local task_id="${1:?task id is required}"
  local chains="${2:?chain count is required}"
  local scenario_keys="${3:-${LMH_DEFAULT_SCENARIO_KEYS}}"
  laplace_mh_validate_positive_integer "task id" "${task_id}"
  laplace_mh_validate_positive_integer "chains" "${chains}"
  laplace_mh_parse_scenarios "${scenario_keys}"

  local task_count=$((LMH_N_SCENARIOS * chains))
  if (( task_id > task_count )); then
    echo "Task ${task_id} exceeds the configured ${task_count} tasks." >&2
    return 2
  fi
  local zero_index=$((task_id - 1))
  local scenario_index=$((zero_index / chains))
  LMH_TASK_CHAIN=$((zero_index % chains + 1))
  LMH_TASK_SCENARIO="${LMH_SCENARIOS[${scenario_index}]}"
  LMH_TASK_SCENARIO_INDEX=$((scenario_index + 1))
  LMH_N_TASKS="${task_count}"
}

laplace_mh_run_signature() {
  local n_time="${1:?n_time is required}"
  local period="${2:?period is required}"
  local draws="${3:?draws is required}"
  local warmup="${4:?warmup is required}"
  local chains="${5:?chains is required}"
  local mh_steps="${6:?mh_steps is required}"
  echo "n${n_time}p${period}_d${draws}w${warmup}c${chains}m${mh_steps}"
}

laplace_mh_run_directory() {
  local results_root="${1:?results root is required}"
  local run_id="${2:?run id is required}"
  local signature="${3:?signature is required}"
  echo "${results_root}/08_simulation_laplace_mh/${run_id}__${signature}"
}
