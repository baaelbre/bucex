"""Fit all six structural simulations with exact Laplace-MH state updates.

This is the method-matched counterpart of ``03_simulation_laplace.py``. It
uses the same simulations, model, priors, MCMC defaults, tables, and figures;
only the latent-state engine and its exact-correction diagnostics differ.

The scientific settings come from the same ``config/simulation.json`` as
examples 02, 03, and 04; the exact-correction fit and diagnostics stay visible.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import io
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
SIMULATION = CONFIG["simulation"]
PRIOR_SETTINGS = CONFIG["priors"]
MCMC_SETTINGS = CONFIG["mcmc"]
INFERENCE = CONFIG["inference"]
FIGURES = CONFIG["figures"]
RUNTIME = CONFIG["runtime"]

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

ALPHA_PRIOR_SD = float(PRIOR_SETTINGS["alpha_sd"])
BETA_PRIOR_MEAN = float(PRIOR_SETTINGS["beta_mean"])
BETA_PRIOR_SD = float(PRIOR_SETTINGS["beta_sd"])
INITIAL_SEASON_PRIOR_SD = float(PRIOR_SETTINGS["seasonal_initial_sd"])
SIGMA2_PRIOR_A = float(PRIOR_SETTINGS["sigma2"]["a"])
SIGMA2_PRIOR_B = float(PRIOR_SETTINGS["sigma2"]["b"])
XI_PRIOR_BOUNDS = tuple(PRIOR_SETTINGS["xi_bounds"])
XI_MAX_ABS = float(PRIOR_SETTINGS["xi_max_abs"])
INNOVATION_SLAB_SD = dict(PRIOR_SETTINGS["innovation_slab_sd"])
LEVEL_DYNAMIC_PROBABILITY = float(PRIOR_SETTINGS["level_dynamic_probability"])
TREND_PROBABILITIES = tuple(PRIOR_SETTINGS["trend_probabilities"])
SEASON_PROBABILITIES = tuple(PRIOR_SETTINGS["season_probabilities"])

PARAMETERIZATION = str(INFERENCE["parameterization"])
ASIS = bool(INFERENCE["asis"])
DRAWS = int(MCMC_SETTINGS["draws"])
WARMUP = int(MCMC_SETTINGS["warmup"])
CHAINS = int(MCMC_SETTINGS["chains"])
SEED = int(MCMC_SETTINGS["seed"])
MH_STEPS = int(INFERENCE["laplace_mh_steps"])
PROGRESS = bool(RUNTIME["progress"])
CHAIN_ONLY = bool(RUNTIME["chain_only"])
COMBINE_RUNS = tuple(
    Path(value) for value in RUNTIME.get("combine_runs", ()) if value
)

FIGURE_FORMATS = tuple(FIGURES["formats"])
FIGURE_DPI = int(FIGURES["dpi"])
DIAGNOSTIC_FIGURES = bool(FIGURES["diagnostics"])
INTERVAL_PROBABILITY = float(FIGURES["interval_probability"])
PHASE_LABELS = tuple(f"phase {index + 1}" for index in range(PERIOD))
PREDICTIVE_DRAWS = int(FIGURES["predictive_draws"])
FOCUS_PHASE = int(FIGURES["focus_phase"])
if not 1 <= FOCUS_PHASE <= PERIOD:
    raise ValueError(f"figures.focus_phase must be between 1 and {PERIOD}.")
FORECAST_HORIZON = int(FIGURES["forecast_horizon"])
FORECAST_HISTORY = int(FIGURES["forecast_history"])

RUN_SIGNATURE = f"n{N_TIME}p{PERIOD}_d{DRAWS}w{WARMUP}c{CHAINS}m{MH_STEPS}"
OUTPUT_DIR = RESULTS_ROOT / SCRIPT_NAME / f"{RUN_TIMESTAMP}__{RUN_SIGNATURE}"

phase = np.arange(PERIOD, dtype=float)
dynamic_cycle = -DYNAMIC_SEASON_AMPLITUDE * np.cos(2.0 * np.pi * phase / PERIOD)
dynamic_cycle -= dynamic_cycle.mean()
fixed_cycle = -FIXED_SEASON_AMPLITUDE * np.cos(2.0 * np.pi * phase / PERIOD)
fixed_cycle -= fixed_cycle.mean()

ALL_SCENARIOS = (
    {
        "name": "stationary",
        "key": "stationary",
        "model": bx.Model(bx.GEV(), (bx.LocalLinearTrend(level_mode="static", trend_mode="off"),), name="stationary"),
        "params": {"sigma": SIGMA, "xi": XI},
        "initial_state": np.array([INITIAL_LEVEL]),
        "seed": SIMULATION_SEED,
        "structural_truth": {"level": 1, "slope": 0, "seasonal": 0},
    },
    {
        "name": "linear_trend",
        "key": "linear",
        "model": bx.Model(bx.GEV(), (bx.LocalLinearTrend(level_mode="static", trend_mode="static"),), name="linear trend"),
        "params": {"sigma": SIGMA, "xi": XI},
        "initial_state": np.array([INITIAL_LEVEL, LINEAR_SLOPE]),
        "seed": SIMULATION_SEED + 1,
        "structural_truth": {"level": 1, "slope": 1, "seasonal": 0},
    },
    {
        "name": "random_walk",
        "key": "random_walk",
        "model": bx.Model(bx.GEV(), (bx.LocalLinearTrend(level_mode="dynamic", trend_mode="off"),), name="random walk"),
        "params": {"sigma": SIGMA, "xi": XI, "sd.level": RANDOM_WALK_SD},
        "initial_state": np.array([INITIAL_LEVEL]),
        "seed": SIMULATION_SEED + 2,
        "structural_truth": {"level": 2, "slope": 0, "seasonal": 0},
    },
    {
        "name": "local_linear_trend",
        "key": "llt",
        "model": bx.Model(bx.GEV(), (bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),), name="local linear trend"),
        "params": {"sigma": SIGMA, "xi": XI, "sd.level": LOCAL_LEVEL_SD, "sd.slope": LOCAL_SLOPE_SD},
        "initial_state": np.array([INITIAL_LEVEL, LOCAL_INITIAL_SLOPE]),
        "seed": SIMULATION_SEED + 3,
        "structural_truth": {"level": 2, "slope": 2, "seasonal": 0},
    },
    {
        "name": "stationary_dynamic_season",
        "key": "dynamic_season",
        "model": bx.Model(
            bx.GEV(),
            (bx.LocalLinearTrend(level_mode="static", trend_mode="off"), bx.DummySeasonal(PERIOD, mode="dynamic")),
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
        "model": bx.Model(
            bx.GEV(),
            (bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"), bx.DummySeasonal(PERIOD, mode="static")),
            name="local linear trend plus fixed seasonality",
        ),
        "params": {"sigma": SIGMA, "xi": XI, "sd.level": LOCAL_LEVEL_SD, "sd.slope": LOCAL_SLOPE_SD},
        "initial_state": np.r_[INITIAL_LEVEL, LOCAL_INITIAL_SLOPE, fixed_cycle[1:]],
        "seed": SIMULATION_SEED + 3,
        "structural_truth": {"level": 2, "slope": 2, "seasonal": 1},
    },
)

ALL_SCENARIO_INDEX = {
    scenario["key"]: index for index, scenario in enumerate(ALL_SCENARIOS)
}
_requested_scenario_keys = tuple(
    str(value).strip() for value in SIMULATION["scenario_keys"] if str(value).strip()
)
if _requested_scenario_keys:
    unknown_scenario_keys = tuple(
        key for key in _requested_scenario_keys if key not in ALL_SCENARIO_INDEX
    )
    if unknown_scenario_keys:
        raise ValueError(
            "Unknown simulation.scenario_keys: "
            + ", ".join(unknown_scenario_keys)
            + ". Available keys: "
            + ", ".join(ALL_SCENARIO_INDEX)
        )
    if len(set(_requested_scenario_keys)) != len(_requested_scenario_keys):
        raise ValueError("simulation.scenario_keys must not contain duplicates.")
    SCENARIOS = tuple(
        ALL_SCENARIOS[ALL_SCENARIO_INDEX[key]] for key in _requested_scenario_keys
    )
else:
    SCENARIOS = ALL_SCENARIOS

# Fit one encompassing model to every scenario. SSVS decides whether each
# process is zero, fixed, or dynamic; no scenario-specific model is supplied.
FIT_MODEL = bx.Model(
    bx.GEV(xi_bounds=XI_PRIOR_BOUNDS),
    (
        bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
        bx.DummySeasonal(period=PERIOD, mode="dynamic"),
    ),
    name="GEV unobserved-components model",
)

PRIOR_SETTINGS = {
    "alpha_sd": ALPHA_PRIOR_SD,
    "beta_mean": BETA_PRIOR_MEAN,
    "beta_sd": BETA_PRIOR_SD,
    "seasonal_initial_sd": INITIAL_SEASON_PRIOR_SD,
    "sigma2": {"a": SIGMA2_PRIOR_A, "b": SIGMA2_PRIOR_B},
    "xi_bounds": list(XI_PRIOR_BOUNDS),
    "xi_max_abs": XI_MAX_ABS,
    "innovation_slab_sd": INNOVATION_SLAB_SD,
    "level_dynamic_probability": LEVEL_DYNAMIC_PROBABILITY,
    "trend_probabilities": list(TREND_PROBABILITIES),
    "season_probabilities": list(SEASON_PROBABILITIES),
}


def expected_truth_for(scenario: dict[str, object]) -> dict[str, object]:
    """Return the complete reproducibility record for one simulation."""

    params = scenario["params"]
    return {
        "name": scenario["name"],
        "n_time": N_TIME,
        "period": PERIOD,
        "model": scenario["model"].to_dict(),
        "params": params,
        "parameter_truth": {
            "sigma": SIGMA,
            "xi": XI,
            "sd.level": params.get("sd.level", 0.0),
            "sd.slope": params.get("sd.slope", 0.0),
            "sd.seasonal": params.get("sd.seasonal", 0.0),
        },
        "structural_truth": scenario["structural_truth"],
        "initial_state": scenario["initial_state"].tolist(),
        "seed": scenario["seed"],
    }


def simulate_scenario_table(
    scenario: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Simulate one scenario and reproduce the canonical CSV representation."""

    simulation = bx.simulate(
        scenario["model"],
        N_TIME,
        scenario["params"],
        initial_state=scenario["initial_state"],
        seed=scenario["seed"],
    )
    state_names = scenario["model"].state_names
    states = simulation.states[1:]
    seasonal_name = next(
        (name for name in state_names if name.startswith("seasonal[")), None
    )
    table = pd.DataFrame(
        {
            "time": np.arange(1, N_TIME + 1),
            "cycle": np.arange(N_TIME) // PERIOD + 1,
            "phase": np.arange(N_TIME) % PERIOD + 1,
            "y": simulation.y,
            "eta": simulation.eta,
            "level": states[:, state_names.index("level")],
            "slope": (
                states[:, state_names.index("slope")]
                if "slope" in state_names
                else np.zeros(N_TIME)
            ),
            "seasonal": (
                states[:, state_names.index(seasonal_name)]
                if seasonal_name
                else np.zeros(N_TIME)
            ),
        }
    )
    # Every standalone and PBS-array fit sees exactly the float representation
    # written by the finalizer, without requiring concurrent jobs to share a
    # simulation CSV while they are sampling.
    csv_buffer = io.StringIO()
    table.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return pd.read_csv(csv_buffer), expected_truth_for(scenario)


def priors_for(y: np.ndarray) -> bx.FSGEVPriors:
    """Construct the method-matched SSVS prior used by examples 03 and 08."""

    return bx.ssvs_gev_priors(
        period=PERIOD,
        alpha_mean=float(np.median(y)),
        alpha_sd=ALPHA_PRIOR_SD,
        beta_mean=BETA_PRIOR_MEAN,
        beta_sd=BETA_PRIOR_SD,
        seasonal_initial_sd=INITIAL_SEASON_PRIOR_SD,
        sigma2_prior=bx.InverseGammaPrior(SIGMA2_PRIOR_A, SIGMA2_PRIOR_B),
        xi_prior=bx.UniformPrior(*XI_PRIOR_BOUNDS),
        xi_max_abs=XI_MAX_ABS,
        innovation_slab_sd=INNOVATION_SLAB_SD,
        level_dynamic_probability=LEVEL_DYNAMIC_PROBABILITY,
        trend_probabilities=TREND_PROBABILITIES,
        season_probabilities=SEASON_PROBABILITIES,
    )


def fit_scenario(
    scenario: dict[str, object],
    table: pd.DataFrame,
    truth: dict[str, object],
    *,
    seed: int,
    chains: int,
    progress: bool,
) -> bx.FitResult:
    """Fit one structural scenario with the exact Laplace-MH kernel."""

    fit = bx.fit(
        table["y"].to_numpy(float),
        model=FIT_MODEL,
        priors=priors_for(table["y"].to_numpy(float)),
        engine="laplace_mh",
        parameterization=PARAMETERIZATION,
        asis=ASIS,
        mcmc=bx.MCMC(
            draws=DRAWS,
            warmup=WARMUP,
            chains=chains,
            seed=seed,
            progress=progress,
        ),
        laplace=bx.Laplace(mh_steps=MH_STEPS),
        name=scenario["name"],
    )
    fit.metadata.update(
        {
            "example": "simulation_laplace_mh",
            "truth": truth,
            "prior_settings": PRIOR_SETTINGS,
        }
    )
    return fit


def main() -> None:
    if COMBINE_RUNS and len(COMBINE_RUNS) != CHAINS:
        raise ValueError(
            f"runtime.combine_runs contains {len(COMBINE_RUNS)} runs, "
            f"but mcmc.chains={CHAINS}."
        )
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "legend.frameon": False})
    labels = {0: "zero", 1: "fixed", 2: "dynamic"}
    algorithm_rows = []

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
        "settings_file": str(SETTINGS_PATH),
        "engine": "laplace_mh",
        "target": "exact posterior",
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
        "fit_model": FIT_MODEL.to_dict(),
        "priors": PRIOR_SETTINGS,
        "inference": {
            "parameterization": PARAMETERIZATION,
            "asis": ASIS,
            "laplace_mh_steps": MH_STEPS,
        },
        "mcmc": {
            "draws": DRAWS,
            "warmup": WARMUP,
            "chains": CHAINS,
            "seed": SEED,
        },
        "laplace": {"mh_steps": MH_STEPS},
        "scenarios": [scenario["name"] for scenario in SCENARIOS],
        "scenario_directories": {
            scenario["name"]: scenario["key"] for scenario in SCENARIOS
        },
        "figures": {
            "formats": list(FIGURE_FORMATS),
            "dpi": FIGURE_DPI,
            "diagnostics": DIAGNOSTIC_FIGURES,
            "interval_probability": INTERVAL_PROBABILITY,
            "predictive_draws": PREDICTIVE_DRAWS,
            "forecast_horizon": FORECAST_HORIZON,
            "forecast_history": FORECAST_HISTORY,
            "focus_phase": FOCUS_PHASE,
        },
        "chain_only": CHAIN_ONLY,
        "combined_chain_runs": [str(path) for path in COMBINE_RUNS],
    }
    config_path.write_text(
        json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8"
    )

    for scenario in SCENARIOS:
        number = ALL_SCENARIO_INDEX[scenario["key"]]
        data_path = OUTPUT_DIR / "simulations" / f"{scenario['key']}.csv"
        truth_path = data_path.with_suffix(".json")
        expected_truth = expected_truth_for(scenario)

        if data_path.is_file() and truth_path.is_file() and not OVERWRITE:
            table = pd.read_csv(data_path)
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
            for key in ("name", "n_time", "period", "model", "params", "initial_state", "seed"):
                if truth.get(key) != expected_truth[key]:
                    raise ValueError(f"{truth_path} has different {key}; set output.overwrite=1.")
        else:
            if not OVERWRITE and (data_path.exists() or truth_path.exists()):
                raise FileExistsError(f"Only one simulation artifact exists for {scenario['name']}; set output.overwrite=1.")
            table, truth = simulate_scenario_table(scenario)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(data_path, index=False)
            truth_path.write_text(json.dumps(truth, indent=2, sort_keys=True), encoding="utf-8")
            # Fit the canonical CSV representation. A later standalone PGAS
            # process reads these exact floats, and warm starts deliberately
            # require bit-identical observations.
            table = pd.read_csv(data_path)

        fit_path = OUTPUT_DIR / "fits" / scenario["key"] / "combined.bucex"
        if fit_path.is_file() and not OVERWRITE:
            laplace_fit = bx.FitResult.load(fit_path)
            if (
                laplace_fit.plan.engine != "laplace_mh"
                or laplace_fit.n_time != N_TIME
                or laplace_fit.n_chains != CHAINS
                or laplace_fit.draws_per_chain != DRAWS
                or laplace_fit.family != FIT_MODEL.family
                or laplace_fit.model.period != FIT_MODEL.period
                or laplace_fit.state_names != FIT_MODEL.state_names
                or laplace_fit.compiled.noise_names != FIT_MODEL.noise_names
                or laplace_fit.metadata.get("prior_settings") != PRIOR_SETTINGS
            ):
                raise ValueError(f"{fit_path} does not match the current model, prior, or MCMC settings.")
            print(f"Reusing {fit_path}")
        elif COMBINE_RUNS:
            source_paths = [
                run_dir / "fits" / scenario["key"] / "combined.bucex"
                for run_dir in COMBINE_RUNS
            ]
            missing = [path for path in source_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Cannot combine Laplace-MH chains; missing fit file(s):\n  "
                    + "\n  ".join(str(path) for path in missing)
                )
            laplace_fit = bx.combine_fits(
                [bx.FitResult.load(path) for path in source_paths]
            )
            laplace_fit.metadata["combined_chain_sources"] = [
                str(path) for path in source_paths
            ]
            fit_path.parent.mkdir(parents=True, exist_ok=True)
            laplace_fit.save(fit_path)
        else:
            laplace_fit = fit_scenario(
                scenario,
                table,
                truth,
                seed=SEED + 100 * number,
                chains=CHAINS,
                progress=PROGRESS,
            )
            fit_path.parent.mkdir(parents=True, exist_ok=True)
            laplace_fit.save(fit_path)

        if CHAIN_ONLY:
            print(f"One-chain Laplace-MH fit complete: {scenario['name']}")
            continue

        # Tables are written explicitly so the example shows what every
        # FitResult method returns.
        table_dir = OUTPUT_DIR / "tables" / scenario["key"]
        figure_dir = OUTPUT_DIR / "figures" / scenario["key"]
        table_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        diagnostics = laplace_fit.diagnostics()
        engine_diagnostics = diagnostics["engine"]
        algorithm_rows.append({"scenario": scenario["name"], **engine_diagnostics})

        parameters = pd.DataFrame.from_dict(laplace_fit.static_summary(), orient="index").rename_axis("parameter")
        parameters.to_csv(table_dir / "parameters.csv")
        diagnostics["parameters"].to_csv(table_dir / "diagnostics.csv")
        pd.DataFrame([{"metric": key, "value": value} for key, value in diagnostics["engine"].items()]).to_csv(
            table_dir / "algorithm.csv", index=False
        )

        eta_draws = laplace_fit.eta_draws(original_scale=True)
        lower, median, upper = np.quantile(eta_draws, [0.05, 0.50, 0.95], axis=0)
        pd.DataFrame(
            {"time": table["time"], "phase": table["phase"], "observed": laplace_fit.observed, "lower": lower, "median": median, "upper": upper, "truth": table["eta"]}
        ).to_csv(table_dir / "trajectory.csv", index=False)

        selection = laplace_fit.component_probabilities().reset_index()
        selection["truth_code"] = selection["process"].map(scenario["structural_truth"])
        selection["truth_state"] = selection["truth_code"].map(labels)
        selection["probability_true_state"] = [row[labels[int(row["truth_code"])]] for _, row in selection.iterrows()]
        selection.to_csv(table_dir / "selection.csv", index=False)
        laplace_fit.structural_model_probabilities().to_csv(table_dir / "models.csv", index=False)
        laplace_fit.component_transition_summary().reset_index().to_csv(table_dir / "switching.csv", index=False)

        # Replicate the fitted observations, then propagate the complete state
        # equation into the future. Both operations use the FitResult API.
        predictive = laplace_fit.posterior_predictive(
            draws=PREDICTIVE_DRAWS,
            seed=SEED + 20_000 + number,
        )
        predictive.summary(level=INTERVAL_PROBABILITY).to_csv(
            table_dir / "posterior_predictive.csv", index=False
        )
        forecast = laplace_fit.forecast(
            FORECAST_HORIZON,
            draws=PREDICTIVE_DRAWS,
            seed=SEED + 30_000 + number,
        )
        forecast.summary(level=INTERVAL_PROBABILITY).to_csv(table_dir / "forecast.csv", index=False)
        forecast.summary(level=INTERVAL_PROBABILITY, phase=FOCUS_PHASE).to_csv(
            table_dir / f"forecast_phase_{FOCUS_PHASE:02d}.csv", index=False
        )
        forecast.summary(level=INTERVAL_PROBABILITY, target="level").to_csv(
            table_dir / "forecast_level.csv", index=False
        )
        (table_dir / "summary.json").write_text(
            json.dumps(
                {
                    "fit": str(fit_path),
                    "series": laplace_fit.series_name,
                    "n_time": laplace_fit.n_time,
                    "n_chains": laplace_fit.n_chains,
                    "draws_per_chain": laplace_fit.draws_per_chain,
                    "plan": laplace_fit.plan.to_dict(),
                    "engine_diagnostics": diagnostics["engine"],
                    "prior_settings": PRIOR_SETTINGS,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        figure, axis = laplace_fit.plot("predictor", credible_interval=INTERVAL_PROBABILITY)
        axis.plot(np.arange(N_TIME), table["eta"], color="#123B4A", linestyle="--", linewidth=1.2, label=r"$\mu_t$")
        title = bx.config_title(CONFIG, "predictor")
        if title is not None:
            axis.set_title(title.format(**scenario))
        axis.legend()
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"trajectory.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        selected_phase = table["phase"].to_numpy(int) == FOCUS_PHASE
        figure, axis = laplace_fit.plot(
            "predictor", credible_interval=INTERVAL_PROBABILITY, phase=FOCUS_PHASE
        )
        axis.plot(
            np.arange(N_TIME)[selected_phase],
            table.loc[selected_phase, "eta"],
            color="#123B4A",
            linestyle="--",
            linewidth=1.2,
            label=r"$\mu_t$",
        )
        title = bx.config_title(CONFIG, "phase_predictor")
        if title is not None:
            axis.set_title(title.format(**scenario, phase=PHASE_LABELS[FOCUS_PHASE - 1]))
        axis.legend()
        for extension in FIGURE_FORMATS:
            figure.savefig(
                figure_dir / f"trajectory_phase_{FOCUS_PHASE:02d}.{extension}",
                dpi=FIGURE_DPI,
                bbox_inches="tight",
            )
        plt.close(figure)

        axis = predictive.plot(
            level=INTERVAL_PROBABILITY,
            observed=laplace_fit.observed,
            title=bx.config_title(CONFIG, "posterior_predictive"),
            ylabel="temperature / °C",
        )
        figure = axis.figure
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"posterior_predictive.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        axis = forecast.plot(
            level=INTERVAL_PROBABILITY,
            history=laplace_fit.observed,
            history_dates=np.arange(N_TIME),
            history_points=FORECAST_HISTORY,
            title=bx.config_title(CONFIG, "forecast"),
            ylabel="temperature / °C",
        )
        figure = axis.figure
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"forecast.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        axis = forecast.plot(
            level=INTERVAL_PROBABILITY,
            phase=FOCUS_PHASE,
            phase_label=PHASE_LABELS[FOCUS_PHASE - 1],
            history=laplace_fit.observed,
            history_dates=np.arange(N_TIME),
            history_points=FORECAST_HISTORY,
            title=bx.config_title(CONFIG, "phase_forecast"),
            ylabel="temperature / °C",
        )
        figure = axis.figure
        for extension in FIGURE_FORMATS:
            figure.savefig(
                figure_dir / f"forecast_phase_{FOCUS_PHASE:02d}.{extension}",
                dpi=FIGURE_DPI,
                bbox_inches="tight",
            )
        plt.close(figure)

        level_history = np.median(laplace_fit.state_original("level"), axis=0)
        axis = forecast.plot(
            level=INTERVAL_PROBABILITY,
            target="level",
            history=level_history,
            history_dates=np.arange(N_TIME),
            history_points=FORECAST_HISTORY,
            title=bx.config_title(CONFIG, "level_forecast"),
            ylabel="latent level / °C",
        )
        figure = axis.figure
        for extension in FIGURE_FORMATS:
            figure.savefig(
                figure_dir / f"forecast_level.{extension}",
                dpi=FIGURE_DPI,
                bbox_inches="tight",
            )
        plt.close(figure)

        figure, axis = laplace_fit.plot(
            "level", credible_interval=INTERVAL_PROBABILITY, truth=table["level"].to_numpy()
        )
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"level.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, axis = laplace_fit.plot(
            "level",
            credible_interval=INTERVAL_PROBABILITY,
            truth=table["level"].to_numpy(),
            show_observed=False,
        )
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"level_no_observations.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, axis = laplace_fit.plot(
            "slope", credible_interval=INTERVAL_PROBABILITY, truth=table["slope"].to_numpy()
        )
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"slope.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, axis = laplace_fit.plot("component_probabilities")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"selection.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, _ = laplace_fit.plot(
            "process_sds", truths=truth["parameter_truth"]
        )
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"process_sd.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, _ = laplace_fit.plot("parameter_densities", parameters=("sigma", "xi"), truths=truth["parameter_truth"])
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"gev.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, axis = laplace_fit.plot("season", labels=PHASE_LABELS, show_interval=False)
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"season.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        if DIAGNOSTIC_FIGURES:
            for kind, filename in (("traces", "sd_traces"), ("acf", "acf")):
                figure, _ = laplace_fit.plot(kind)
                for extension in FIGURE_FORMATS:
                    figure.savefig(figure_dir / f"{filename}.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
                plt.close(figure)
        state_acceptance = float(engine_diagnostics.get("state_acceptance", np.nan))
        if np.isfinite(state_acceptance) and state_acceptance < 0.10:
            print(
                f"warning: {scenario['name']} has low whole-trajectory acceptance "
                f"({state_acceptance:.3f})."
            )
        print(f"Laplace-MH fit complete: {scenario['name']}")

    if CHAIN_ONLY:
        print(f"One-chain Laplace-MH simulation outputs: {OUTPUT_DIR}")
        return

    diagnostics_path = OUTPUT_DIR / "tables" / "laplace_mh_diagnostics.csv"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(algorithm_rows).to_csv(diagnostics_path, index=False)
    print(f"Laplace-MH simulation outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
