"""Simulate selected structural GEV models with the public bucex API.

Every model is visible below: stationary, linear trend, random walk, local
linear trend, changing seasonality, and local linear trend plus a fixed cycle.
Run all six with ``python examples/02_structural_simulations.py`` or select a
subset with ``simulation.scenario_keys`` in a JSON configuration.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if (SOURCE_ROOT / "bucex").is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
EXAMPLE_ROOT = Path(__file__).resolve().parent
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

import bucex as bx


# Pick another JSON here for IDE/notebook runs; ``--config PATH`` overrides it.
DEFAULT_CONFIG_FILE = EXAMPLE_ROOT / "config" / "simulation.json"
CONFIG_PARSER = argparse.ArgumentParser()
CONFIG_PARSER.add_argument("--config", type=Path, default=DEFAULT_CONFIG_FILE)
CONFIG_ARGUMENTS, _ = CONFIG_PARSER.parse_known_args()
SETTINGS_PATH = CONFIG_ARGUMENTS.config.expanduser().resolve()
CONFIG = bx.load_config(SETTINGS_PATH)
SETTINGS_FILE = Path(
    CONFIG.get("_runner", {}).get("source_config", SETTINGS_PATH)
).resolve()
SIMULATION = CONFIG["simulation"]
FIGURES = CONFIG["figures"]

RESULTS_ROOT = Path(CONFIG["output"]["results_root"])
SCRIPT_NAME = Path(__file__).stem
RUN_TIMESTAMP = CONFIG["output"].get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
OVERWRITE = bool(CONFIG["output"]["overwrite"])

N_TIME = int(SIMULATION["n_time"])
PERIOD = int(SIMULATION["period"])
SIGMA = float(SIMULATION["sigma"])
XI = float(SIMULATION["xi"])
INITIAL_LEVEL = float(SIMULATION["initial_level"])
LINEAR_SLOPE = float(SIMULATION["linear_slope"])
RANDOM_WALK_SD = float(SIMULATION["random_walk_sd"])
LOCAL_LEVEL_SD = float(SIMULATION["local_level_sd"])
LOCAL_SLOPE_SD = float(SIMULATION["local_slope_sd"])
LOCAL_INITIAL_SLOPE = float(SIMULATION["local_initial_slope"])
DYNAMIC_SEASON_AMPLITUDE = float(SIMULATION["dynamic_season_amplitude"])
FIXED_SEASON_AMPLITUDE = float(SIMULATION["fixed_season_amplitude"])
SEASONAL_SD = float(SIMULATION["seasonal_sd"])
SIMULATION_SEED = int(SIMULATION["seed"])

FIGURE_FORMATS = tuple(FIGURES["formats"])
FIGURE_DPI = int(FIGURES["dpi"])
COLORS = {"navy": "#123B4A", "teal": "#1D7F7A", "grey": "#7A8589", "coral": "#D96C4F"}

RUN_SIGNATURE = f"n{N_TIME}p{PERIOD}_s{SIMULATION_SEED}"
OUTPUT_DIR = RESULTS_ROOT / SCRIPT_NAME / f"{RUN_TIMESTAMP}__{RUN_SIGNATURE}"


# Dummy-season states store period - 1 effects. Their next transition creates
# the missing effect as minus the sum, hence the one-position rotation here.
phase = np.arange(PERIOD, dtype=float)
dynamic_cycle = -DYNAMIC_SEASON_AMPLITUDE * np.cos(2.0 * np.pi * phase / PERIOD)
dynamic_cycle -= dynamic_cycle.mean()
fixed_cycle = -FIXED_SEASON_AMPLITUDE * np.cos(2.0 * np.pi * phase / PERIOD)
fixed_cycle -= fixed_cycle.mean()

# These are six ordinary declarative bucex models. The structural truth codes
# follow zero=0, fixed=1, dynamic=2, matching FitResult.component_probabilities.
ALL_SCENARIOS = (
    {
        "name": "stationary",
        "key": "stationary",
        "title": "Stationary",
        "model": bx.Model(
            bx.GEV(xi_bounds=(-0.5, 0.5)),
            (bx.LocalLinearTrend(level_mode="static", trend_mode="off"),),
            name="stationary",
        ),
        "params": {"sigma": SIGMA, "xi": XI},
        "initial_state": np.array([INITIAL_LEVEL]),
        "seed": SIMULATION_SEED,
        "structural_truth": {"level": 1, "slope": 0, "seasonal": 0},
    },
    {
        "name": "linear_trend",
        "key": "linear",
        "title": "Linear trend",
        "model": bx.Model(
            bx.GEV(xi_bounds=(-0.5, 0.5)),
            (bx.LocalLinearTrend(level_mode="static", trend_mode="static"),),
            name="linear trend",
        ),
        "params": {"sigma": SIGMA, "xi": XI},
        "initial_state": np.array([INITIAL_LEVEL, LINEAR_SLOPE]),
        "seed": SIMULATION_SEED + 1,
        "structural_truth": {"level": 1, "slope": 1, "seasonal": 0},
    },
    {
        "name": "random_walk",
        "key": "random_walk",
        "title": "Random walk",
        "model": bx.Model(
            bx.GEV(xi_bounds=(-0.5, 0.5)),
            (bx.LocalLinearTrend(level_mode="dynamic", trend_mode="off"),),
            name="random walk",
        ),
        "params": {"sigma": SIGMA, "xi": XI, "sd.level": RANDOM_WALK_SD},
        "initial_state": np.array([INITIAL_LEVEL]),
        "seed": SIMULATION_SEED + 2,
        "structural_truth": {"level": 2, "slope": 0, "seasonal": 0},
    },
    {
        "name": "local_linear_trend",
        "key": "llt",
        "title": "Local linear trend",
        "model": bx.Model(
            bx.GEV(xi_bounds=(-0.5, 0.5)),
            (bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),),
            name="local linear trend",
        ),
        "params": {"sigma": SIGMA, "xi": XI, "sd.level": LOCAL_LEVEL_SD, "sd.slope": LOCAL_SLOPE_SD},
        "initial_state": np.array([INITIAL_LEVEL, LOCAL_INITIAL_SLOPE]),
        "seed": SIMULATION_SEED + 3,
        "structural_truth": {"level": 2, "slope": 2, "seasonal": 0},
    },
    {
        "name": "stationary_dynamic_season",
        "key": "dynamic_season",
        "title": "Stationary + changing seasonality",
        "model": bx.Model(
            bx.GEV(xi_bounds=(-0.5, 0.5)),
            (
                bx.LocalLinearTrend(level_mode="static", trend_mode="off"),
                bx.DummySeasonal(period=PERIOD, mode="dynamic"),
            ),
            name="stationary plus changing seasonality",
        ),
        "params": {"sigma": SIGMA, "xi": XI, "sd.seasonal": SEASONAL_SD},
        "initial_state": np.r_[INITIAL_LEVEL, dynamic_cycle[1:]],
        "seed": SIMULATION_SEED + 4,
        "structural_truth": {"level": 1, "slope": 0, "seasonal": 2},
    },
    {
        "name": "local_linear_trend_fixed_season",
        "key": "llt_season",
        "title": "Local linear trend + fixed seasonality",
        "model": bx.Model(
            bx.GEV(xi_bounds=(-0.5, 0.5)),
            (
                bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
                bx.DummySeasonal(period=PERIOD, mode="static"),
            ),
            name="local linear trend plus fixed seasonality",
        ),
        "params": {"sigma": SIGMA, "xi": XI, "sd.level": LOCAL_LEVEL_SD, "sd.slope": LOCAL_SLOPE_SD},
        "initial_state": np.r_[INITIAL_LEVEL, LOCAL_INITIAL_SLOPE, fixed_cycle[1:]],
        # This matches the local-linear-trend innovations; the fixed seasonal
        # cycle is the only additional signal.
        "seed": SIMULATION_SEED + 3,
        "structural_truth": {"level": 2, "slope": 2, "seasonal": 1},
    },
)

SCENARIO_INDEX = {
    scenario["key"]: index for index, scenario in enumerate(ALL_SCENARIOS)
}
_requested_scenario_keys = tuple(
    str(value).strip()
    for value in SIMULATION.get("scenario_keys", ())
    if str(value).strip()
)
if len(set(_requested_scenario_keys)) != len(_requested_scenario_keys):
    raise ValueError("simulation.scenario_keys must not contain duplicates.")
_unknown_scenario_keys = tuple(
    key for key in _requested_scenario_keys if key not in SCENARIO_INDEX
)
if _unknown_scenario_keys:
    raise ValueError(
        "Unknown simulation.scenario_keys: "
        + ", ".join(_unknown_scenario_keys)
        + ". Available keys: "
        + ", ".join(SCENARIO_INDEX)
    )
SCENARIOS = (
    tuple(
        ALL_SCENARIOS[SCENARIO_INDEX[key]] for key in _requested_scenario_keys
    )
    if _requested_scenario_keys
    else ALL_SCENARIOS
)


def main() -> None:
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "legend.frameon": False})
    config_path = OUTPUT_DIR / "run_config.json"
    if config_path.exists() and not OVERWRITE:
        raise FileExistsError(
            f"Refusing to overwrite {config_path}; set output.overwrite=1 to rerun."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_config = {
        "script": SCRIPT_NAME,
        "created_at": datetime.now().astimezone().isoformat(),
        "run_timestamp": RUN_TIMESTAMP,
        "run_signature": RUN_SIGNATURE,
        "output_directory": str(OUTPUT_DIR),
        "bucex_version": bx.__version__,
        "settings_file": str(SETTINGS_FILE),
        "simulation": {
            "n_time": N_TIME,
            "period": PERIOD,
            "sigma": SIGMA,
            "xi": XI,
            "initial_level": INITIAL_LEVEL,
            "linear_slope": LINEAR_SLOPE,
            "random_walk_sd": RANDOM_WALK_SD,
            "local_level_sd": LOCAL_LEVEL_SD,
            "local_slope_sd": LOCAL_SLOPE_SD,
            "local_initial_slope": LOCAL_INITIAL_SLOPE,
            "dynamic_season_amplitude": DYNAMIC_SEASON_AMPLITUDE,
            "fixed_season_amplitude": FIXED_SEASON_AMPLITUDE,
            "seasonal_sd": SEASONAL_SD,
            "seed": SIMULATION_SEED,
        },
        "scenarios": [
            {
                "name": scenario["name"],
                "key": scenario["key"],
                "model": scenario["model"].to_dict(),
                "params": scenario["params"],
                "initial_state": scenario["initial_state"].tolist(),
                "seed": scenario["seed"],
                "structural_truth": scenario["structural_truth"],
            }
            for scenario in SCENARIOS
        ],
        "figures": {"formats": list(FIGURE_FORMATS), "dpi": FIGURE_DPI},
    }
    config_path.write_text(
        json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8"
    )

    simulation_dir = OUTPUT_DIR / "simulations"
    figure_dir = OUTPUT_DIR / "figures"
    simulation_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    catalog = []

    for scenario in SCENARIOS:
        simulation = bx.simulate(
            scenario["model"],
            N_TIME,
            scenario["params"],
            initial_state=scenario["initial_state"],
            seed=scenario["seed"],
        )
        state_names = scenario["model"].state_names
        states = simulation.states[1:]
        level = states[:, state_names.index("level")]
        slope = states[:, state_names.index("slope")] if "slope" in state_names else np.zeros(N_TIME)
        seasonal_name = next((name for name in state_names if name.startswith("seasonal[")), None)
        seasonal = states[:, state_names.index(seasonal_name)] if seasonal_name else np.zeros(N_TIME)
        table = pd.DataFrame(
            {
                "time": np.arange(1, N_TIME + 1),
                "cycle": np.arange(N_TIME) // PERIOD + 1,
                "phase": np.arange(N_TIME) % PERIOD + 1,
                "y": simulation.y,
                "eta": simulation.eta,
                "level": level,
                "slope": slope,
                "seasonal": seasonal,
            }
        )
        truth = {
            "name": scenario["name"],
            "key": scenario["key"],
            "title": scenario["title"],
            "n_time": N_TIME,
            "period": PERIOD,
            "model": scenario["model"].to_dict(),
            "params": scenario["params"],
            "parameter_truth": {
                "sigma": SIGMA,
                "xi": XI,
                "sd.level": scenario["params"].get("sd.level", 0.0),
                "sd.slope": scenario["params"].get("sd.slope", 0.0),
                "sd.seasonal": scenario["params"].get("sd.seasonal", 0.0),
            },
            "structural_truth": scenario["structural_truth"],
            "initial_state": scenario["initial_state"].tolist(),
            "seed": scenario["seed"],
        }
        data_path = simulation_dir / f"{scenario['key']}.csv"
        truth_path = data_path.with_suffix(".json")
        if not OVERWRITE and (data_path.exists() or truth_path.exists()):
            raise FileExistsError(f"Refusing to overwrite {data_path}; set output.overwrite=1.")
        table.to_csv(data_path, index=False)
        truth_path.write_text(json.dumps(truth, indent=2, sort_keys=True), encoding="utf-8")
        catalog.append({"name": scenario["name"], "key": scenario["key"], **scenario["structural_truth"], **truth["parameter_truth"], "seed": scenario["seed"]})

        # Each simulated series is a separate figure, as used in the talk.
        figure, axis = plt.subplots(figsize=(11, 4.2))
        axis.scatter(table["time"], table["y"], s=7, alpha=0.24, color=COLORS["grey"], label=r"$y_t$")
        axis.plot(table["time"], table["eta"], color=COLORS["teal"], linewidth=1.8, label=r"$\mu_t$")
        time_series_title = bx.config_title(CONFIG, "time_series")
        if time_series_title is not None:
            axis.set_title(time_series_title.format(**scenario))
        axis.set_xlabel("observation")
        axis.set_ylabel("GEV location / °C")
        axis.grid(axis="y", alpha=0.35)
        axis.legend()
        figure.tight_layout()
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"{scenario['key']}_series.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        # The only multi-panel simulation graphic is the allowed structural
        # decomposition into level, slope, and seasonal effect.
        figure, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
        axes[0].plot(table["time"], table["level"], color=COLORS["teal"])
        axes[0].set_ylabel("level")
        axes[1].plot(table["time"], table["slope"], color=COLORS["coral"])
        axes[1].axhline(0.0, color="0.5", linewidth=0.7)
        axes[1].set_ylabel("slope")
        axes[2].plot(table["time"], table["seasonal"], color=COLORS["navy"])
        axes[2].axhline(0.0, color="0.5", linewidth=0.7)
        axes[2].set_ylabel("season")
        axes[2].set_xlabel("observation")
        for axis in axes:
            axis.grid(axis="y", alpha=0.35)
        decomposition_title = bx.config_title(CONFIG, "decomposition")
        if decomposition_title is not None:
            figure.suptitle(decomposition_title.format(**scenario), weight="bold")
        figure.tight_layout()
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"{scenario['key']}_decomposition.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        if seasonal_name is not None:
            figure, axis = plt.subplots(figsize=(9, 4.5))
            for current_phase in range(1, PERIOD + 1):
                selected = table["phase"] == current_phase
                axis.plot(
                    table.loc[selected, "cycle"],
                    table.loc[selected, "seasonal"],
                    linewidth=1.35,
                    label=f"phase {current_phase}",
                )
            season_title = bx.config_title(CONFIG, "seasonality")
            if season_title is not None:
                axis.set_title(season_title.format(**scenario))
            axis.set_xlabel("cycle")
            axis.set_ylabel("seasonal effect")
            axis.legend(ncol=PERIOD)
            axis.grid(axis="y", alpha=0.35)
            figure.tight_layout()
            for extension in FIGURE_FORMATS:
                figure.savefig(figure_dir / f"{scenario['key']}_season.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(figure)

        print(f"Simulated {scenario['name']}: {data_path}")

    catalog_path = OUTPUT_DIR / "tables" / "scenarios.csv"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(catalog).to_csv(catalog_path, index=False)
    print(f"Structural simulation outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
