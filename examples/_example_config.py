"""Load the small JSON configuration files used by the numbered examples.

The JSON file is the authoritative record of scientific settings. A custom
file can be selected with ``--config PATH`` or ``BUCEX_CONFIG=PATH``. The
existing ``BUCEX_*`` environment variables remain available for short HPC
runs and are applied after loading the file.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any


EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = EXAMPLE_DIR / "config"


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, received {value!r}.")


def _coerce(value: str, template: Any) -> Any:
    if isinstance(template, bool):
        return _parse_bool(value)
    if isinstance(template, int) and not isinstance(template, bool):
        return int(value)
    if isinstance(template, float):
        return float(value)
    if isinstance(template, list):
        if "," in value:
            separator = ","
        elif os.pathsep in value:
            separator = os.pathsep
        else:
            separator = ":"
        parts = [part.strip() for part in value.split(separator) if part.strip()]
        if not template:
            return parts
        return [_coerce(part, template[0]) for part in parts]
    if template is None:
        return value or None
    return value


def _get_path(config: dict[str, Any], path: str) -> tuple[dict[str, Any], str]:
    parts = path.split(".")
    current = config
    for part in parts[:-1]:
        value = current.get(part)
        if not isinstance(value, dict):
            raise KeyError(f"Configuration path {path!r} does not exist.")
        current = value
    if parts[-1] not in current:
        raise KeyError(f"Configuration path {path!r} does not exist.")
    return current, parts[-1]


def _config_argument() -> Path | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    arguments, _ = parser.parse_known_args()
    return arguments.config


def load_example_config(
    default_filename: str,
    environment_overrides: dict[str, str],
) -> tuple[dict[str, Any], Path]:
    """Load, copy, and environment-override one example configuration."""

    argument_path = _config_argument()
    environment_path = os.environ.get("BUCEX_CONFIG")
    selected = argument_path or (Path(environment_path) if environment_path else None)
    path = (selected or (DEFAULT_CONFIG_DIR / default_filename)).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Example configuration not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        loaded = json.load(stream)
    if not isinstance(loaded, dict):
        raise ValueError(f"Example configuration must be a JSON object: {path}")
    config = deepcopy(loaded)
    for variable, config_path in environment_overrides.items():
        if variable not in os.environ:
            continue
        parent, key = _get_path(config, config_path)
        parent[key] = _coerce(os.environ[variable], parent[key])
    return config, path


COMMON_OUTPUT_OVERRIDES = {
    "BUCEX_RESULTS_ROOT": "output.results_root",
    "BUCEX_OVERWRITE": "output.overwrite",
    "BUCEX_PROGRESS": "runtime.progress",
    "BUCEX_CHAIN_ONLY": "runtime.chain_only",
}


SIMULATION_OVERRIDES = {
    **COMMON_OUTPUT_OVERRIDES,
    "BUCEX_N_TIME": "simulation.n_time",
    "BUCEX_PERIOD": "simulation.period",
    "BUCEX_SIGMA": "simulation.sigma",
    "BUCEX_XI": "simulation.xi",
    "BUCEX_INITIAL_LEVEL": "simulation.initial_level",
    "BUCEX_LINEAR_SLOPE": "simulation.linear_slope",
    "BUCEX_RANDOM_WALK_SD": "simulation.random_walk_sd",
    "BUCEX_LOCAL_LEVEL_SD": "simulation.local_level_sd",
    "BUCEX_LOCAL_SLOPE_SD": "simulation.local_slope_sd",
    "BUCEX_LOCAL_INITIAL_SLOPE": "simulation.local_initial_slope",
    "BUCEX_DYNAMIC_SEASON_AMPLITUDE": "simulation.dynamic_season_amplitude",
    "BUCEX_FIXED_SEASON_AMPLITUDE": "simulation.fixed_season_amplitude",
    "BUCEX_SEASONAL_SD": "simulation.seasonal_sd",
    "BUCEX_SIMULATION_SEED": "simulation.seed",
    "BUCEX_SCENARIO_KEYS": "simulation.scenario_keys",
    "BUCEX_ALPHA_PRIOR_SD": "priors.alpha_sd",
    "BUCEX_BETA_PRIOR_MEAN": "priors.beta_mean",
    "BUCEX_BETA_PRIOR_SD": "priors.beta_sd",
    "BUCEX_INITIAL_SEASON_PRIOR_SD": "priors.seasonal_initial_sd",
    "BUCEX_SIGMA2_PRIOR_A": "priors.sigma2.a",
    "BUCEX_SIGMA2_PRIOR_B": "priors.sigma2.b",
    "BUCEX_XI_MAX_ABS": "priors.xi_max_abs",
    "BUCEX_LEVEL_SLAB_SD": "priors.innovation_slab_sd.level",
    "BUCEX_TREND_SLAB_SD": "priors.innovation_slab_sd.trend",
    "BUCEX_SEASON_SLAB_SD": "priors.innovation_slab_sd.season",
    "BUCEX_LEVEL_DYNAMIC_PROBABILITY": "priors.level_dynamic_probability",
    "BUCEX_TREND_PROBABILITIES": "priors.trend_probabilities",
    "BUCEX_SEASON_PROBABILITIES": "priors.season_probabilities",
    "BUCEX_DRAWS": "mcmc.draws",
    "BUCEX_WARMUP": "mcmc.warmup",
    "BUCEX_CHAINS": "mcmc.chains",
    "BUCEX_SEED": "mcmc.seed",
    "BUCEX_PARTICLES": "inference.pgas_particles",
    "BUCEX_LAPLACE_MH_STEPS": "inference.laplace_mh_steps",
    "BUCEX_PREDICTIVE_DRAWS": "figures.predictive_draws",
    "BUCEX_FOCUS_PHASE": "figures.focus_phase",
    "BUCEX_FORECAST_HORIZON": "figures.forecast_horizon",
    "BUCEX_FORECAST_HISTORY": "figures.forecast_history",
}


UCCLE_OVERRIDES = {
    **COMMON_OUTPUT_OVERRIDES,
    "BUCEX_DATA_DIR": "data.data_dir",
    "BUCEX_START": "data.start",
    "BUCEX_END": "data.end",
    "BUCEX_ALPHA_PRIOR_SD": "priors.alpha_sd",
    "BUCEX_BETA_PRIOR_MEAN": "priors.beta_mean",
    "BUCEX_BETA_PRIOR_SD": "priors.beta_sd",
    "BUCEX_INITIAL_SEASON_PRIOR_SD": "priors.seasonal_initial_sd",
    "BUCEX_SIGMA2_PRIOR_A": "priors.sigma2.a",
    "BUCEX_SIGMA2_PRIOR_B": "priors.sigma2.b",
    "BUCEX_XI_MAX_ABS": "priors.xi_max_abs",
    "BUCEX_LEVEL_SLAB_SD": "priors.innovation_slab_sd.level",
    "BUCEX_TREND_SLAB_SD": "priors.innovation_slab_sd.trend",
    "BUCEX_SEASON_SLAB_SD": "priors.innovation_slab_sd.season",
    "BUCEX_LEVEL_DYNAMIC_PROBABILITY": "priors.level_dynamic_probability",
    "BUCEX_TREND_PROBABILITIES": "priors.trend_probabilities",
    "BUCEX_SEASON_PROBABILITIES": "priors.season_probabilities",
    "BUCEX_DRAWS": "mcmc.draws",
    "BUCEX_WARMUP": "mcmc.warmup",
    "BUCEX_CHAINS": "mcmc.chains",
    "BUCEX_SEED": "mcmc.seed",
    "BUCEX_PARTICLES": "inference.pgas_particles",
    "BUCEX_LAPLACE_MH_STEPS": "inference.laplace_mh_steps",
    "BUCEX_PREDICTIVE_DRAWS": "figures.predictive_draws",
    "BUCEX_FOCUS_MONTH": "figures.focus_month",
    "BUCEX_FORECAST_HORIZON": "figures.forecast_horizon",
    "BUCEX_FORECAST_HISTORY": "figures.forecast_history",
}


CENTERED_IG_OVERRIDES = {
    **COMMON_OUTPUT_OVERRIDES,
    "BUCEX_N_TIME": "simulation.n_time",
    "BUCEX_SIGMA": "simulation.sigma",
    "BUCEX_XI": "simulation.xi",
    "BUCEX_INITIAL_LEVEL": "simulation.initial_level",
    "BUCEX_RANDOM_WALK_SD": "simulation.random_walk_sd",
    "BUCEX_SIMULATION_SEED": "simulation.seed",
    "BUCEX_INITIAL_LEVEL_PRIOR_MEAN": "priors.initial_level.mean",
    "BUCEX_INITIAL_LEVEL_PRIOR_SD": "priors.initial_level.sd",
    "BUCEX_LEVEL_IG_A": "priors.level_variance.shape",
    "BUCEX_LEVEL_IG_B": "priors.level_variance.scale",
    "BUCEX_SIGMA_IG_A": "priors.sigma_variance.shape",
    "BUCEX_SIGMA_IG_B": "priors.sigma_variance.scale",
    "BUCEX_XI_PRIOR_MEAN": "priors.shape.mean",
    "BUCEX_XI_PRIOR_SD": "priors.shape.sd",
    "BUCEX_ENGINE": "inference.engine",
    "BUCEX_PARTICLES": "inference.pgas_particles",
    "BUCEX_LAPLACE_MH_STEPS": "inference.laplace_mh_steps",
    "BUCEX_DRAWS": "mcmc.draws",
    "BUCEX_WARMUP": "mcmc.warmup",
    "BUCEX_CHAINS": "mcmc.chains",
    "BUCEX_SEED": "mcmc.seed",
    "BUCEX_PREDICTIVE_DRAWS": "figures.predictive_draws",
    "BUCEX_FORECAST_HORIZON": "figures.forecast_horizon",
    "BUCEX_FORECAST_HISTORY": "figures.forecast_history",
    "BUCEX_MAX_ACF_LAG": "figures.max_acf_lag",
}


def load_simulation_config() -> tuple[dict[str, Any], Path]:
    return load_example_config("simulation.json", SIMULATION_OVERRIDES)


def load_uccle_config() -> tuple[dict[str, Any], Path]:
    return load_example_config("uccle.json", UCCLE_OVERRIDES)


def load_centered_ig_config() -> tuple[dict[str, Any], Path]:
    return load_example_config("centered_ig.json", CENTERED_IG_OVERRIDES)
