"""Fit all four Uccle extremes with Laplace-initialized PGAS.

It uses the same ``config/uccle.json`` as examples 05 and 09 and keeps the
Laplace initialization and exact PGAS fit calls explicit.
"""
from __future__ import annotations

from datetime import datetime
import json
import os
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
from _example_config import load_uccle_config


# One calibrated file is shared by all three Uccle inference engines.
CONFIG, SETTINGS_PATH = load_uccle_config()
DATA = CONFIG["data"]
PRIOR_SETTINGS = CONFIG["priors"]
MCMC_SETTINGS = CONFIG["mcmc"]
INFERENCE = CONFIG["inference"]
FIGURES = CONFIG["figures"]
RUNTIME = CONFIG["runtime"]

RESULTS_ROOT = Path(CONFIG["output"]["results_root"])
SCRIPT_NAME = Path(__file__).stem
RUN_TIMESTAMP = os.environ.get("BUCEX_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OVERWRITE = bool(CONFIG["output"]["overwrite"])

# Data.
DATA_DIR = Path(DATA["data_dir"]) if DATA["data_dir"] else None
START = str(DATA["start"])
END = DATA["end"] or None
SERIES = tuple(DATA["series"])
PERIOD = int(DATA["period"])

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

DRAWS = int(MCMC_SETTINGS["draws"])
WARMUP = int(MCMC_SETTINGS["warmup"])
CHAINS = int(MCMC_SETTINGS["chains"])
PARTICLES = int(INFERENCE["pgas_particles"])
SEED = int(MCMC_SETTINGS["seed"])
PROGRESS = bool(RUNTIME["progress"])
CHAIN_ONLY = bool(RUNTIME["chain_only"])
COMBINE_RUNS = tuple(
    Path(value)
    for value in os.environ.get("BUCEX_COMBINE_RUNS", "").split(os.pathsep)
    if value
)

FIGURE_FORMATS = tuple(FIGURES["formats"])
FIGURE_DPI = int(FIGURES["dpi"])
DIAGNOSTIC_FIGURES = bool(FIGURES["diagnostics"])
PREDICTIVE_DRAWS = int(FIGURES["predictive_draws"])
FORECAST_HORIZON = int(FIGURES["forecast_horizon"])
FORECAST_HISTORY = int(FIGURES["forecast_history"])
FOCUS_MONTH = int(FIGURES["focus_month"])
if not 1 <= FOCUS_MONTH <= PERIOD:
    raise ValueError("BUCEX_FOCUS_MONTH must be between 1 and 12.")
MONTH_LABELS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

RUN_SIGNATURE = (
    f"y{START.removesuffix('-01-01')}-{(END or 'latest').removesuffix('-12-31')}"
    f"_d{DRAWS}w{WARMUP}c{CHAINS}p{PARTICLES}"
)
OUTPUT_DIR = RESULTS_ROOT / SCRIPT_NAME / f"{RUN_TIMESTAMP}__{RUN_SIGNATURE}"


MODEL = bx.Model(
    bx.GEV(xi_bounds=XI_PRIOR_BOUNDS),
    (bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"), bx.DummySeasonal(PERIOD, mode="dynamic")),
    name="monthly GEV unobserved-components model",
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


def main() -> None:
    if COMBINE_RUNS and len(COMBINE_RUNS) != CHAINS:
        raise ValueError(
            f"BUCEX_COMBINE_RUNS contains {len(COMBINE_RUNS)} runs, "
            f"but BUCEX_CHAINS={CHAINS}."
        )
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "legend.frameon": False})
    selection_rows = []

    config_path = OUTPUT_DIR / "run_config.json"
    if config_path.exists() and not OVERWRITE:
        raise FileExistsError(
            f"Refusing to overwrite {config_path}; set BUCEX_OVERWRITE=1 to rerun."
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
        "engine": "pgas",
        "initializer": "laplace",
        "data": {
            "data_dir": None if DATA_DIR is None else str(DATA_DIR),
            "start": START,
            "end": END,
            "series": list(SERIES),
            "period": PERIOD,
        },
        "model": MODEL.to_dict(),
        "priors": PRIOR_SETTINGS,
        "mcmc": {
            "draws": DRAWS,
            "warmup": WARMUP,
            "chains": CHAINS,
            "particles": PARTICLES,
            "seed": SEED,
        },
        "figures": {
            "formats": list(FIGURE_FORMATS),
            "dpi": FIGURE_DPI,
            "diagnostics": DIAGNOSTIC_FIGURES,
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
        values = bx.load_uccle_series(name, DATA_DIR, start=START, end=END)
        focus_phase = (FOCUS_MONTH - int(values.index[0].month)) % PERIOD + 1
        focus_label = MONTH_LABELS[FOCUS_MONTH - 1]
        tail = bx.UCCLE_INFO[name]["tail"]
        sign = -1.0 if tail == "min" else 1.0
        transformed = sign * values.to_numpy(float)
        priors = bx.ssvs_gev_priors(
            period=PERIOD,
            alpha_mean=float(np.median(transformed)),
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

        laplace_path = OUTPUT_DIR / "fits" / name / "laplace" / "combined.bucex"
        if laplace_path.is_file() and not OVERWRITE:
            laplace_fit = bx.FitResult.load(laplace_path)
            if (
                laplace_fit.n_time != len(values)
                or laplace_fit.n_chains != CHAINS
                or laplace_fit.draws_per_chain != DRAWS
                or laplace_fit.family != MODEL.family
                or laplace_fit.model.period != MODEL.period
                or laplace_fit.state_names != MODEL.state_names
                or laplace_fit.compiled.noise_names != MODEL.noise_names
                or laplace_fit.metadata.get("prior_settings") != PRIOR_SETTINGS
            ):
                raise ValueError(f"{laplace_path} does not match the current settings.")
        elif COMBINE_RUNS:
            source_paths = [
                run_dir / "fits" / name / "laplace" / "combined.bucex"
                for run_dir in COMBINE_RUNS
            ]
            missing = [path for path in source_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Cannot combine Uccle Laplace initializers; missing fit file(s):\n  "
                    + "\n  ".join(str(path) for path in missing)
                )
            laplace_fit = bx.combine_fits(
                [bx.FitResult.load(path) for path in source_paths]
            )
            laplace_fit.metadata["combined_chain_sources"] = [
                str(path) for path in source_paths
            ]
            laplace_path.parent.mkdir(parents=True, exist_ok=True)
            laplace_fit.save(laplace_path)
        else:
            laplace_fit = bx.fit(
                values,
                model=MODEL,
                priors=priors,
                engine="laplace",
                parameterization="fruehwirth_schnatter",
                asis=False,
                mcmc=bx.MCMC(draws=DRAWS, warmup=WARMUP, chains=CHAINS, seed=SEED + 100 * number, progress=PROGRESS),
                name=name,
                tail=tail,
            )
            laplace_fit.metadata.update(
                {
                    "example": "uccle_laplace_initializer",
                    "description": bx.UCCLE_INFO[name]["description"],
                    "start": START,
                    "end": END,
                    "prior_settings": PRIOR_SETTINGS,
                }
            )
            laplace_path.parent.mkdir(parents=True, exist_ok=True)
            laplace_fit.save(laplace_path)

        pgas_path = OUTPUT_DIR / "fits" / name / "pgas" / "combined.bucex"
        if pgas_path.is_file() and not OVERWRITE:
            pgas_fit = bx.FitResult.load(pgas_path)
            if (
                pgas_fit.n_time != len(values)
                or pgas_fit.n_chains != CHAINS
                or pgas_fit.draws_per_chain != DRAWS
                or pgas_fit.family != MODEL.family
                or pgas_fit.model.period != MODEL.period
                or pgas_fit.state_names != MODEL.state_names
                or pgas_fit.compiled.noise_names != MODEL.noise_names
                or pgas_fit.metadata.get("prior_settings") != PRIOR_SETTINGS
                or int(pgas_fit.metadata.get("particles", PARTICLES)) != PARTICLES
            ):
                raise ValueError(f"{pgas_path} does not match the current settings.")
            print(f"Reusing {pgas_path}")
        elif COMBINE_RUNS:
            source_paths = [
                run_dir / "fits" / name / "pgas" / "combined.bucex"
                for run_dir in COMBINE_RUNS
            ]
            missing = [path for path in source_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Cannot combine Uccle PGAS chains; missing fit file(s):\n  "
                    + "\n  ".join(str(path) for path in missing)
                )
            pgas_fit = bx.combine_fits(
                [bx.FitResult.load(path) for path in source_paths]
            )
            pgas_fit.metadata["combined_chain_sources"] = [
                str(path) for path in source_paths
            ]
            pgas_fit.metadata["warm_start_source"] = str(laplace_path)
            pgas_path.parent.mkdir(parents=True, exist_ok=True)
            pgas_fit.save(pgas_path)
        else:
            pgas_fit = bx.fit(
                values,
                model=MODEL,
                priors=laplace_fit.priors,
                engine="pgas",
                parameterization="fruehwirth_schnatter",
                asis=False,
                mcmc=bx.MCMC(draws=DRAWS, warmup=WARMUP, chains=CHAINS, seed=SEED + 10_000 + 100 * number, progress=PROGRESS),
                particles=bx.Particles(n=PARTICLES, proposal="guided"),
                name=name,
                tail=tail,
                init=laplace_fit,
            )
            pgas_fit.metadata.update(
                {
                    "example": "uccle_pgas",
                    "description": bx.UCCLE_INFO[name]["description"],
                    "warm_start_source": str(laplace_path),
                    "start": START,
                    "end": END,
                    "prior_settings": PRIOR_SETTINGS,
                    "particles": PARTICLES,
                }
            )
            pgas_path.parent.mkdir(parents=True, exist_ok=True)
            pgas_fit.save(pgas_path)

        if CHAIN_ONLY:
            print(f"One-chain Uccle PGAS fit complete: {name}")
            continue

        table_dir = OUTPUT_DIR / "tables" / name
        figure_dir = OUTPUT_DIR / "figures" / name
        table_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        diagnostics = pgas_fit.diagnostics()

        pd.DataFrame.from_dict(pgas_fit.static_summary(), orient="index").rename_axis("parameter").to_csv(table_dir / "parameters.csv")
        diagnostics["parameters"].to_csv(table_dir / "diagnostics.csv")
        pd.DataFrame([{"metric": key, "value": value} for key, value in diagnostics["engine"].items()]).to_csv(table_dir / "algorithm.csv", index=False)
        eta_draws = pgas_fit.eta_draws(original_scale=True)
        lower, median, upper = np.quantile(eta_draws, [0.05, 0.50, 0.95], axis=0)
        pd.DataFrame({"date": values.index, "month": values.index.month, "observed": values.to_numpy(), "lower": lower, "median": median, "upper": upper}).to_csv(
            table_dir / "trajectory.csv", index=False
        )
        selection = pgas_fit.component_probabilities().reset_index()
        selection.insert(0, "series", name)
        selection.insert(1, "engine", "pgas")
        selection.to_csv(table_dir / "selection.csv", index=False)
        pgas_fit.structural_model_probabilities().to_csv(table_dir / "models.csv", index=False)
        pgas_fit.component_transition_summary().reset_index().to_csv(table_dir / "switching.csv", index=False)

        predictive = pgas_fit.posterior_predictive(
            draws=PREDICTIVE_DRAWS,
            seed=SEED + 20_000 + number,
        )
        predictive.summary(level=0.90).to_csv(
            table_dir / "posterior_predictive.csv", index=False
        )
        forecast = pgas_fit.forecast(
            FORECAST_HORIZON,
            draws=PREDICTIVE_DRAWS,
            seed=SEED + 30_000 + number,
        )
        forecast.summary(level=0.90).to_csv(table_dir / "forecast.csv", index=False)
        focus_forecast = forecast.summary(level=0.90, phase=focus_phase)
        focus_forecast.insert(1, "calendar_month", FOCUS_MONTH)
        focus_forecast.insert(2, "calendar_month_label", focus_label)
        focus_forecast.to_csv(
            table_dir / f"forecast_{focus_label.lower()}.csv", index=False
        )
        forecast.summary(level=0.90, target="level").to_csv(
            table_dir / "forecast_level.csv", index=False
        )
        (table_dir / "summary.json").write_text(
            json.dumps(
                {
                    "fit": str(pgas_path),
                    "warm_start": str(laplace_path),
                    "series": name,
                    "description": bx.UCCLE_INFO[name]["description"],
                    "tail": tail,
                    "start": str(values.index.min().date()),
                    "end": str(values.index.max().date()),
                    "n_time": pgas_fit.n_time,
                    "n_chains": pgas_fit.n_chains,
                    "draws_per_chain": pgas_fit.draws_per_chain,
                    "particles": PARTICLES,
                    "plan": pgas_fit.plan.to_dict(),
                    "engine_diagnostics": diagnostics["engine"],
                    "prior_settings": PRIOR_SETTINGS,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        laplace_selection = laplace_fit.component_probabilities().reset_index()
        laplace_selection.insert(0, "series", name)
        laplace_selection.insert(1, "engine", "laplace")
        selection_rows.extend((laplace_selection, selection))

        figure, axis = pgas_fit.plot("predictor", credible_interval=0.90)
        axis.set_title(f"{name}: posterior latent predictor")
        axis.set_ylabel("GEV location / °C")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"trajectory.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, axis = pgas_fit.plot(
            "predictor", credible_interval=0.90, phase=focus_phase
        )
        axis.set_title(f"{name}: {focus_label} GEV-location trajectory")
        axis.set_ylabel("GEV location / °C")
        for extension in FIGURE_FORMATS:
            figure.savefig(
                figure_dir / f"trajectory_{focus_label.lower()}.{extension}",
                dpi=FIGURE_DPI,
                bbox_inches="tight",
            )
        plt.close(figure)

        axis = predictive.plot(
            level=0.90,
            observed=pgas_fit.observed,
            title=f"{name}: posterior predictive check",
            ylabel="temperature / °C",
        )
        figure = axis.figure
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"posterior_predictive.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        axis = forecast.plot(
            level=0.90,
            history=pgas_fit.observed,
            history_dates=values.index,
            history_points=FORECAST_HISTORY,
            title=f"{name}: posterior predictive forecast",
            ylabel="temperature / °C",
        )
        figure = axis.figure
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"forecast.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        axis = forecast.plot(
            level=0.90,
            phase=focus_phase,
            phase_label=focus_label,
            history=pgas_fit.observed,
            history_dates=values.index,
            history_points=FORECAST_HISTORY,
            title=f"{name}: {focus_label} posterior predictive forecast",
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

        level_history = np.median(pgas_fit.state_original("level"), axis=0)
        axis = forecast.plot(
            level=0.90,
            target="level",
            history=level_history,
            history_dates=values.index,
            history_points=FORECAST_HISTORY,
            title=f"{name}: seasonally adjusted level forecast",
            ylabel="latent GEV level / °C",
        )
        figure = axis.figure
        for extension in FIGURE_FORMATS:
            figure.savefig(
                figure_dir / f"forecast_level.{extension}",
                dpi=FIGURE_DPI,
                bbox_inches="tight",
            )
        plt.close(figure)

        figure, axis = pgas_fit.plot("level", credible_interval=0.90)
        axis.set_title(f"{name}: posterior latent level")
        axis.set_ylabel("GEV level / °C")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"level.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, axis = pgas_fit.plot(
            "level", credible_interval=0.90, show_observed=False
        )
        axis.set_title(f"{name}: posterior latent level")
        axis.set_ylabel("GEV level / °C")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"level_no_observations.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        trend_states = np.asarray(pgas_fit.parameter("state_trend"), dtype=int)
        slope_condition = "dynamic" if np.any(trend_states == 2) else None
        figure, axis = pgas_fit.plot(
            "slope", credible_interval=0.90,
            scale="decade",
            unit="slope / °C per decade",
            condition_on=slope_condition,
            show_fixed=bool(np.any(trend_states == 1)),
        )
        axis.set_title("Posterior slope")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"slope.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, axis = pgas_fit.plot("component_probabilities")
        axis.set_title(f"{name}: structural selection")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"selection.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, _ = pgas_fit.plot("process_sds")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"process_sd.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, _ = pgas_fit.plot("parameter_densities", parameters=("sigma", "xi"))
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"gev.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, axis = pgas_fit.plot("season", show_interval=False)
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"season.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        try:
            figure, _ = pgas_fit.plot("endpoint")
        except ValueError:
            print(f"{name}: no finite endpoint in the retained posterior draws.")
        else:
            for extension in FIGURE_FORMATS:
                figure.savefig(figure_dir / f"endpoint.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(figure)

        if DIAGNOSTIC_FIGURES:
            for kind, filename in (("traces", "sd_traces"), ("acf", "acf")):
                figure, _ = pgas_fit.plot(kind)
                for extension in FIGURE_FORMATS:
                    figure.savefig(figure_dir / f"{filename}.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
                plt.close(figure)
        print(f"PGAS fit complete: {name}")

    if CHAIN_ONLY:
        print(f"One-chain Uccle PGAS outputs: {OUTPUT_DIR}")
        return

    selection_table = pd.concat(selection_rows, ignore_index=True)
    selection_path = OUTPUT_DIR / "tables" / "selection_all.csv"
    selection_table.to_csv(selection_path, index=False)

    # One compact comparison of the two engines. Each row is a series/process
    # pair and each bar is the posterior probability of a dynamic component.
    comparison = selection_table.pivot_table(index=["series", "process"], columns="engine", values="dynamic")
    figure, axis = plt.subplots(figsize=(11, 5.2))
    positions = np.arange(len(comparison))
    width = 0.38
    if "laplace" in comparison:
        axis.bar(positions - width / 2, comparison["laplace"], width, label="Laplace")
    if "pgas" in comparison:
        axis.bar(positions + width / 2, comparison["pgas"], width, label="PGAS")
    axis.set_xticks(positions, [f"{series}\n{process}" for series, process in comparison.index])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("posterior P(dynamic)")
    axis.set_title("Uccle structural selection: Laplace and PGAS")
    axis.legend()
    figure.tight_layout()
    comparison_dir = OUTPUT_DIR / "figures"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for extension in FIGURE_FORMATS:
        figure.savefig(comparison_dir / f"selection_engines.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    print(f"Uccle PGAS outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
