"""Fit the Uccle monthly mean-temperature series with exact Gaussian FFBS.

TXm and TNm use the same local-linear-trend, dummy-seasonal, and structural
SSVS specification as the four extreme-temperature analyses.  The observation
model is Gaussian, so its latent-state update is exact and needs neither a
Laplace approximation nor a Metropolis correction.  Every setting is read
from the selected JSON below.
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
DEFAULT_CONFIG_FILE = (
    EXAMPLE_ROOT / "config" / "uccle_gaussian" / "01_txm.json"
)
CONFIG_PARSER = argparse.ArgumentParser()
CONFIG_PARSER.add_argument("--config", type=Path, default=DEFAULT_CONFIG_FILE)
CONFIG_ARGUMENTS, _ = CONFIG_PARSER.parse_known_args()
SETTINGS_PATH = CONFIG_ARGUMENTS.config.expanduser().resolve()
CONFIG = bx.load_config(SETTINGS_PATH)
SETTINGS_FILE = Path(
    CONFIG.get("_runner", {}).get("source_config", SETTINGS_PATH)
).resolve()
DATA = CONFIG["data"]
MODEL_SETTINGS = CONFIG["model"]
PRIOR_SETTINGS = CONFIG["priors"]
MCMC_SETTINGS = CONFIG["mcmc"]
INFERENCE = CONFIG["inference"]
FIGURES = CONFIG["figures"]
RUNTIME = CONFIG["runtime"]

RESULTS_ROOT = Path(CONFIG["output"]["results_root"])
SCRIPT_NAME = Path(__file__).stem
RUN_TIMESTAMP = CONFIG["output"].get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
OVERWRITE = bool(CONFIG["output"]["overwrite"])

# Data. ``None`` uses the complete series shipped with bucex.
DATA_DIR = None if DATA.get("data_dir") is None else Path(DATA["data_dir"])
START = str(DATA["start"])
END = DATA["end"] or None
SERIES = tuple(DATA["series"])
PERIOD = int(DATA["period"])
if (
    str(MODEL_SETTINGS["observation_family"]).lower() != "gaussian"
    or str(MODEL_SETTINGS["time_varying_parameter"]).lower() != "mean"
    or str(MODEL_SETTINGS["trend_component"]).lower() != "local_linear_trend"
    or str(MODEL_SETTINGS["seasonal_component"]).lower() != "dummy"
):
    raise ValueError(
        "Example 12 requires a Gaussian mean model with a local-linear trend "
        "and dummy seasonality."
    )

ALPHA_PRIOR_MEAN = PRIOR_SETTINGS["alpha_mean"]
ALPHA_PRIOR_SD = float(PRIOR_SETTINGS["alpha_sd"])
BETA_PRIOR_MEAN = float(PRIOR_SETTINGS["beta_mean"])
BETA_PRIOR_SD = float(PRIOR_SETTINGS["beta_sd"])
INITIAL_SEASON_PRIOR_SD = float(PRIOR_SETTINGS["seasonal_initial_sd"])
SIGMA2_PRIOR_A = float(PRIOR_SETTINGS["sigma2"]["a"])
SIGMA2_PRIOR_B = float(PRIOR_SETTINGS["sigma2"]["b"])
INNOVATION_SLAB_SD = dict(PRIOR_SETTINGS["innovation_slab_sd"])
LEVEL_DYNAMIC_PROBABILITY = float(PRIOR_SETTINGS["level_dynamic_probability"])
TREND_PROBABILITIES = tuple(PRIOR_SETTINGS["trend_probabilities"])
SEASON_PROBABILITIES = tuple(PRIOR_SETTINGS["season_probabilities"])

PARAMETERIZATION = str(INFERENCE["parameterization"])
ASIS = bool(INFERENCE["asis"])
ENGINE = str(INFERENCE["engine"])
if ENGINE != "ffbs":
    raise ValueError("Example 12 is exact Gaussian inference and requires inference.engine='ffbs'.")
DRAWS = int(MCMC_SETTINGS["draws"])
WARMUP = int(MCMC_SETTINGS["warmup"])
CHAINS = int(MCMC_SETTINGS["chains"])
SEED = int(MCMC_SETTINGS["seed"])
PROGRESS = bool(RUNTIME["progress"])
CHAIN_ONLY = bool(RUNTIME["chain_only"])
COMBINE_RUNS = tuple(
    Path(value) for value in RUNTIME.get("combine_runs", ()) if value
)

FIGURE_FORMATS = tuple(FIGURES["formats"])
FIGURE_DPI = int(FIGURES["dpi"])
DIAGNOSTIC_FIGURES = bool(FIGURES["diagnostics"])
INTERVAL_PROBABILITY = float(FIGURES["interval_probability"])
PREDICTIVE_DRAWS = int(FIGURES["predictive_draws"])
FORECAST_HORIZON = int(FIGURES["forecast_horizon"])
FORECAST_HISTORY = int(FIGURES["forecast_history"])
FOCUS_MONTH = int(FIGURES["focus_month"])
if not 1 <= FOCUS_MONTH <= PERIOD:
    raise ValueError("figures.focus_month must be between 1 and 12.")
MONTH_LABELS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

RUN_SIGNATURE = (
    f"y{START.removesuffix('-01-01')}-{(END or 'latest').removesuffix('-12-31')}"
    f"_d{DRAWS}w{WARMUP}c{CHAINS}"
)
OUTPUT_DIR = RESULTS_ROOT / SCRIPT_NAME / f"{RUN_TIMESTAMP}__{RUN_SIGNATURE}"


MODEL = bx.Model(
    bx.Gaussian(),
    (
        bx.LocalLinearTrend(
            level_mode=str(MODEL_SETTINGS["level_mode"]),
            trend_mode=str(MODEL_SETTINGS["trend_mode"]),
        ),
        bx.DummySeasonal(
            period=PERIOD,
            mode=str(MODEL_SETTINGS["seasonal_mode"]),
        ),
    ),
    name="monthly Gaussian unobserved-components model",
)

PRIOR_SETTINGS = {
    "alpha_mean": ALPHA_PRIOR_MEAN,
    "alpha_sd": ALPHA_PRIOR_SD,
    "beta_mean": BETA_PRIOR_MEAN,
    "beta_sd": BETA_PRIOR_SD,
    "seasonal_initial_sd": INITIAL_SEASON_PRIOR_SD,
    "sigma2": {"a": SIGMA2_PRIOR_A, "b": SIGMA2_PRIOR_B},
    "innovation_slab_sd": INNOVATION_SLAB_SD,
    "level_dynamic_probability": LEVEL_DYNAMIC_PROBABILITY,
    "trend_probabilities": list(TREND_PROBABILITIES),
    "season_probabilities": list(SEASON_PROBABILITIES),
}


def main() -> None:
    if COMBINE_RUNS and len(COMBINE_RUNS) != CHAINS:
        raise ValueError(
            f"runtime.combine_runs contains {len(COMBINE_RUNS)} runs, "
            f"but mcmc.chains={CHAINS}."
        )
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "legend.frameon": False})
    selection_rows = []

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
        "engine": ENGINE,
        "target": "exact Gaussian posterior",
        "data": {
            "data_dir": None if DATA_DIR is None else str(DATA_DIR),
            "start": START,
            "end": END,
            "series": list(SERIES),
            "period": PERIOD,
        },
        "model": MODEL.to_dict(),
        "priors": PRIOR_SETTINGS,
        "inference": {
            "parameterization": PARAMETERIZATION,
            "asis": ASIS,
        },
        "mcmc": {
            "draws": DRAWS,
            "warmup": WARMUP,
            "chains": CHAINS,
            "seed": SEED,
        },
        "figures": {
            "formats": list(FIGURE_FORMATS),
            "dpi": FIGURE_DPI,
            "diagnostics": DIAGNOSTIC_FIGURES,
            "interval_probability": INTERVAL_PROBABILITY,
            "predictive_draws": PREDICTIVE_DRAWS,
            "forecast_horizon": FORECAST_HORIZON,
            "forecast_history": FORECAST_HISTORY,
            "focus_month": FOCUS_MONTH,
        },
        "chain_only": CHAIN_ONLY,
        "combined_chain_runs": [str(path) for path in COMBINE_RUNS],
    }
    config_path.write_text(
        json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8"
    )

    for number, name in enumerate(SERIES):
        values = bx.load_uccle_series(
            name, DATA_DIR, start=START, end=END
        )
        focus_phase = (FOCUS_MONTH - int(values.index[0].month)) % PERIOD + 1
        focus_label = MONTH_LABELS[FOCUS_MONTH - 1]
        if bx.UCCLE_INFO[name]["family"] != "gaussian":
            raise ValueError(
                f"{name} is not a Gaussian Uccle series; choose TXm or TNm."
            )
        alpha_mean = (
            float(np.nanmedian(values.to_numpy(float)))
            if ALPHA_PRIOR_MEAN in {"series_median", "transformed_series_median"}
            else float(ALPHA_PRIOR_MEAN)
        )

        priors = bx.ssvs_gaussian_priors(
            period=PERIOD,
            alpha_mean=alpha_mean,
            alpha_sd=ALPHA_PRIOR_SD,
            beta_mean=BETA_PRIOR_MEAN,
            beta_sd=BETA_PRIOR_SD,
            seasonal_initial_sd=INITIAL_SEASON_PRIOR_SD,
            sigma2_prior=bx.InverseGammaPrior(SIGMA2_PRIOR_A, SIGMA2_PRIOR_B),
            innovation_slab_sd=INNOVATION_SLAB_SD,
            level_dynamic_probability=LEVEL_DYNAMIC_PROBABILITY,
            trend_probabilities=TREND_PROBABILITIES,
            season_probabilities=SEASON_PROBABILITIES,
        )

        fit_path = OUTPUT_DIR / "fits" / name / "combined.bucex"
        if fit_path.is_file() and not OVERWRITE:
            fit = bx.FitResult.load(fit_path)
            if (
                fit.n_time != len(values)
                or fit.n_chains != CHAINS
                or fit.draws_per_chain != DRAWS
                or fit.family != MODEL.family
                or fit.model.period != MODEL.period
                or fit.state_names != MODEL.state_names
                or fit.compiled.noise_names != MODEL.noise_names
                or fit.metadata.get("prior_settings") != PRIOR_SETTINGS
            ):
                raise ValueError(f"{fit_path} does not match the current model, prior, data, or MCMC settings.")
            print(f"Reusing {fit_path}")
        elif COMBINE_RUNS:
            source_paths = [
                run_dir / "fits" / name / "combined.bucex"
                for run_dir in COMBINE_RUNS
            ]
            missing = [path for path in source_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Cannot combine Gaussian Uccle chains; missing fit file(s):\n  "
                    + "\n  ".join(str(path) for path in missing)
                )
            fit = bx.combine_fits(
                [bx.FitResult.load(path) for path in source_paths]
            )
            fit.metadata["combined_chain_sources"] = [
                str(path) for path in source_paths
            ]
            fit_path.parent.mkdir(parents=True, exist_ok=True)
            fit.save(fit_path)
        else:
            fit = bx.fit(
                values,
                model=MODEL,
                priors=priors,
                engine=ENGINE,
                parameterization=PARAMETERIZATION,
                asis=ASIS,
                mcmc=bx.MCMC(draws=DRAWS, warmup=WARMUP, chains=CHAINS, seed=SEED + 100 * number, progress=PROGRESS),
                name=name,
            )
            fit.metadata.update(
                {
                    "example": "uccle_gaussian",
                    "description": bx.UCCLE_INFO[name]["description"],
                    "start": START,
                    "end": END,
                    "prior_settings": PRIOR_SETTINGS,
                }
            )
            fit_path.parent.mkdir(parents=True, exist_ok=True)
            fit.save(fit_path)

        if CHAIN_ONLY:
            print(f"One-chain Uccle Gaussian fit complete: {name}")
            continue

        table_dir = OUTPUT_DIR / "tables" / name
        figure_dir = OUTPUT_DIR / "figures" / name
        table_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        diagnostics = fit.diagnostics()

        pd.DataFrame.from_dict(fit.static_summary(), orient="index").rename_axis("parameter").to_csv(table_dir / "parameters.csv")
        diagnostics["parameters"].to_csv(table_dir / "diagnostics.csv")
        pd.DataFrame([{"metric": key, "value": value} for key, value in diagnostics["engine"].items()]).to_csv(table_dir / "algorithm.csv", index=False)
        eta_draws = fit.eta_draws(original_scale=True)
        lower, median, upper = np.quantile(eta_draws, [0.05, 0.50, 0.95], axis=0)
        pd.DataFrame({"date": values.index, "month": values.index.month, "observed": values.to_numpy(), "lower": lower, "median": median, "upper": upper}).to_csv(
            table_dir / "trajectory.csv", index=False
        )
        selection = fit.component_probabilities().reset_index()
        selection.insert(0, "series", name)
        selection.insert(1, "engine", ENGINE)
        selection.to_csv(table_dir / "selection.csv", index=False)
        selection_rows.append(selection)
        fit.structural_model_probabilities().to_csv(table_dir / "models.csv", index=False)
        fit.component_transition_summary().reset_index().to_csv(table_dir / "switching.csv", index=False)

        predictive = fit.posterior_predictive(
            draws=PREDICTIVE_DRAWS,
            seed=SEED + 20_000 + number,
        )
        predictive.summary(level=INTERVAL_PROBABILITY).to_csv(
            table_dir / "posterior_predictive.csv", index=False
        )
        forecast = fit.forecast(
            FORECAST_HORIZON,
            draws=PREDICTIVE_DRAWS,
            seed=SEED + 30_000 + number,
        )
        forecast.summary(level=INTERVAL_PROBABILITY).to_csv(table_dir / "forecast.csv", index=False)
        focus_forecast = forecast.summary(level=INTERVAL_PROBABILITY, phase=focus_phase)
        focus_forecast.insert(1, "calendar_month", FOCUS_MONTH)
        focus_forecast.insert(2, "calendar_month_label", focus_label)
        focus_forecast.to_csv(
            table_dir / f"forecast_{focus_label.lower()}.csv", index=False
        )
        forecast.summary(level=INTERVAL_PROBABILITY, target="level").to_csv(
            table_dir / "forecast_level.csv", index=False
        )
        (table_dir / "summary.json").write_text(
            json.dumps(
                {
                    "fit": str(fit_path),
                    "series": name,
                    "description": bx.UCCLE_INFO[name]["description"],
                    "family": "gaussian",
                    "start": str(values.index.min().date()),
                    "end": str(values.index.max().date()),
                    "n_time": fit.n_time,
                    "n_chains": fit.n_chains,
                    "draws_per_chain": fit.draws_per_chain,
                    "plan": fit.plan.to_dict(),
                    "engine_diagnostics": diagnostics["engine"],
                    "prior_settings": PRIOR_SETTINGS,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        figure, axis = fit.plot("predictor", credible_interval=INTERVAL_PROBABILITY)
        axis.set_ylabel("monthly mean / °C")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"trajectory.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, axis = fit.plot(
            "predictor", credible_interval=INTERVAL_PROBABILITY, phase=focus_phase
        )
        axis.set_ylabel("monthly mean / °C")
        for extension in FIGURE_FORMATS:
            figure.savefig(
                figure_dir / f"trajectory_{focus_label.lower()}.{extension}",
                dpi=FIGURE_DPI,
                bbox_inches="tight",
            )
        plt.close(figure)

        axis = predictive.plot(
            level=INTERVAL_PROBABILITY,
            observed=fit.observed,
            title=bx.config_title(CONFIG, "posterior_predictive"),
            ylabel="temperature / °C",
        )
        figure = axis.figure
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"posterior_predictive.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        axis = forecast.plot(
            level=INTERVAL_PROBABILITY,
            history=fit.observed,
            history_dates=values.index,
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
            phase=focus_phase,
            phase_label=focus_label,
            history=fit.observed,
            history_dates=values.index,
            history_points=FORECAST_HISTORY,
            title=bx.config_title(CONFIG, "phase_forecast"),
            ylabel="temperature / °C",
        )
        figure = axis.figure
        for extension in FIGURE_FORMATS:
            figure.savefig(
                figure_dir / f"forecast_{focus_label.lower()}.{extension}",
                dpi=FIGURE_DPI,
                bbox_inches="tight",
            )
        plt.close(figure)

        level_history = np.median(fit.state_original("level"), axis=0)
        axis = forecast.plot(
            level=INTERVAL_PROBABILITY,
            target="level",
            history=level_history,
            history_dates=values.index,
            history_points=FORECAST_HISTORY,
            title=bx.config_title(CONFIG, "level_forecast"),
            ylabel="latent mean level / °C",
        )
        figure = axis.figure
        for extension in FIGURE_FORMATS:
            figure.savefig(
                figure_dir / f"forecast_level.{extension}",
                dpi=FIGURE_DPI,
                bbox_inches="tight",
            )
        plt.close(figure)

        figure, axis = fit.plot("level", credible_interval=INTERVAL_PROBABILITY)
        axis.set_ylabel("mean level / °C")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"level.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, axis = fit.plot(
            "level", credible_interval=INTERVAL_PROBABILITY, show_observed=False
        )
        axis.set_ylabel("mean level / °C")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"level_no_observations.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        trend_states = np.asarray(fit.parameter("state_trend"), dtype=int)
        slope_condition = "dynamic" if np.any(trend_states == 2) else None
        figure, axis = fit.plot(
            "slope", credible_interval=INTERVAL_PROBABILITY,
            scale="decade",
            unit="slope / °C per decade",
            condition_on=slope_condition,
            show_fixed=bool(np.any(trend_states == 1)),
        )
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"slope.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, axis = fit.plot("component_probabilities")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"selection.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, _ = fit.plot("process_sds")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"process_sd.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, _ = fit.plot("parameter_densities", parameters=("sigma",))
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"gaussian.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, axis = fit.plot("season", show_interval=False)
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"season.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        if DIAGNOSTIC_FIGURES:
            for kind, filename in (("traces", "sd_traces"), ("acf", "acf")):
                figure, _ = fit.plot(kind)
                for extension in FIGURE_FORMATS:
                    figure.savefig(figure_dir / f"{filename}.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
                plt.close(figure)
        print(f"Exact Gaussian FFBS fit complete: {name}")

    if CHAIN_ONLY:
        print(f"One-chain Uccle Gaussian outputs: {OUTPUT_DIR}")
        return

    selection_path = OUTPUT_DIR / "tables" / "selection_all.csv"
    pd.concat(selection_rows, ignore_index=True).to_csv(selection_path, index=False)
    print(f"Uccle Gaussian outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
