"""Fit TXx, TXn, TNx, and TNn with exact Laplace-MH state updates.

This is the method-matched counterpart of ``05_uccle_laplace.py``. It uses the
same data, model, calibrated priors, MCMC defaults, tables, and figures; only
the latent-state engine and its exact-correction diagnostics differ.

The model, prior, result tables, and figures are all visible in this file.
Run with ``python examples/09_uccle_laplace_mh.py``.
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

import bucex as bx


# Results. The folder name keeps only the data range, slab scales, and sampling
# effort. run_config.json contains the full data, prior, and sampler settings.
RESULTS_ROOT = Path(os.environ.get("BUCEX_RESULTS_ROOT", "results"))
SCRIPT_NAME = Path(__file__).stem
RUN_TIMESTAMP = os.environ.get("BUCEX_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OVERWRITE = os.environ.get("BUCEX_OVERWRITE", "0").lower() in {"1", "true", "yes"}

# Data.
DATA_DIR: Path | None = (
    Path(os.environ["BUCEX_DATA_DIR"]) if "BUCEX_DATA_DIR" in os.environ else None
)
START = os.environ.get("BUCEX_START", "1892-01-01")
END: str | None = os.environ.get("BUCEX_END") or None
SERIES = ("TXx", "TXn", "TNx", "TNn")
PERIOD = 12

# Prior hyperparameters. A zero-centred fixed-slope prior is appropriate for
# both upper-tail series and sign-transformed lower-tail series: a common
# +0.2/120 degree-per-month mean would have the wrong internal sign for the
# lower-tail fits. The 0.0015 SD corresponds to 0.18 degrees per decade and
# regularizes implausibly steep fixed trends without forcing the slope to zero.
ALPHA_PRIOR_SD = float(os.environ.get("BUCEX_ALPHA_PRIOR_SD", "3.2"))
BETA_PRIOR_MEAN = float(os.environ.get("BUCEX_BETA_PRIOR_MEAN", "0.0"))
BETA_PRIOR_SD = float(os.environ.get("BUCEX_BETA_PRIOR_SD", "0.0015"))
INITIAL_SEASON_PRIOR_SD = float(
    os.environ.get("BUCEX_INITIAL_SEASON_PRIOR_SD", "2.25")
)
SIGMA2_PRIOR_A = float(os.environ.get("BUCEX_SIGMA2_PRIOR_A", "2.0"))
SIGMA2_PRIOR_B = float(os.environ.get("BUCEX_SIGMA2_PRIOR_B", "2.0"))
XI_PRIOR_BOUNDS = (-0.50, 0.50)
XI_MAX_ABS = float(os.environ.get("BUCEX_XI_MAX_ABS", "0.50"))
INNOVATION_SLAB_SD = {
    "level": float(os.environ.get("BUCEX_LEVEL_SLAB_SD", "0.03")),
    "trend": float(os.environ.get("BUCEX_TREND_SLAB_SD", "0.00010")),
    "season": float(os.environ.get("BUCEX_SEASON_SLAB_SD", "0.05")),
}
LEVEL_DYNAMIC_PROBABILITY = float(
    os.environ.get("BUCEX_LEVEL_DYNAMIC_PROBABILITY", "0.50")
)
# For the observed monthly record we treat slope and season as scientifically
# present, then select between fixed and dynamic behavior. Use colon-separated
# overrides (for example 0.2:0.4:0.4) for a sensitivity analysis through qsub.
TREND_PROBABILITIES = tuple(
    float(value)
    for value in os.environ.get("BUCEX_TREND_PROBABILITIES", "0:0.5:0.5").split(":")
)
SEASON_PROBABILITIES = tuple(
    float(value)
    for value in os.environ.get("BUCEX_SEASON_PROBABILITIES", "0:0.5:0.5").split(":")
)

# MCMC. These defaults are deliberately identical to example 05.
DRAWS = int(os.environ.get("BUCEX_DRAWS", "1000"))
WARMUP = int(os.environ.get("BUCEX_WARMUP", "1000"))
CHAINS = int(os.environ.get("BUCEX_CHAINS", "1"))
SEED = int(os.environ.get("BUCEX_SEED", "56000"))
MH_STEPS = int(os.environ.get("BUCEX_LAPLACE_MH_STEPS", "1"))
PROGRESS = os.environ.get("BUCEX_PROGRESS", "1").lower() not in {"0", "false", "no"}
CHAIN_ONLY = os.environ.get("BUCEX_CHAIN_ONLY", "0").lower() in {"1", "true", "yes"}
COMBINE_RUNS = tuple(
    Path(value)
    for value in os.environ.get("BUCEX_COMBINE_RUNS", "").split(os.pathsep)
    if value
)

FIGURE_FORMATS = ("pdf", "png")
FIGURE_DPI = 180
DIAGNOSTIC_FIGURES = False
PREDICTIVE_DRAWS = int(os.environ.get("BUCEX_PREDICTIVE_DRAWS", "500"))
FORECAST_HORIZON = int(os.environ.get("BUCEX_FORECAST_HORIZON", "120"))
FORECAST_HISTORY = int(os.environ.get("BUCEX_FORECAST_HISTORY", "360"))
FOCUS_MONTH = int(os.environ.get("BUCEX_FOCUS_MONTH", "7"))
if not 1 <= FOCUS_MONTH <= PERIOD:
    raise ValueError("BUCEX_FOCUS_MONTH must be between 1 and 12.")
MONTH_LABELS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

RUN_SIGNATURE = (
    f"y{START.removesuffix('-01-01')}-{(END or 'latest').removesuffix('-12-31')}"
    f"_d{DRAWS}w{WARMUP}c{CHAINS}m{MH_STEPS}"
)
OUTPUT_DIR = RESULTS_ROOT / SCRIPT_NAME / f"{RUN_TIMESTAMP}__{RUN_SIGNATURE}"


MODEL = bx.Model(
    bx.GEV(xi_bounds=XI_PRIOR_BOUNDS),
    (
        bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
        bx.DummySeasonal(period=PERIOD, mode="dynamic"),
    ),
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
    algorithm_rows = []

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
        "engine": "laplace_mh",
        "target": "exact posterior",
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
            "seed": SEED,
        },
        "laplace": {"mh_steps": MH_STEPS},
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

        fit_path = OUTPUT_DIR / "fits" / name / "combined.bucex"
        if fit_path.is_file() and not OVERWRITE:
            laplace_fit = bx.FitResult.load(fit_path)
            if (
                laplace_fit.plan.engine != "laplace_mh"
                or laplace_fit.n_time != len(values)
                or laplace_fit.n_chains != CHAINS
                or laplace_fit.draws_per_chain != DRAWS
                or laplace_fit.family != MODEL.family
                or laplace_fit.model.period != MODEL.period
                or laplace_fit.state_names != MODEL.state_names
                or laplace_fit.compiled.noise_names != MODEL.noise_names
                or laplace_fit.metadata.get("prior_settings") != PRIOR_SETTINGS
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
                    "Cannot combine Uccle Laplace-MH chains; missing fit file(s):\n  "
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
            laplace_fit = bx.fit(
                values,
                model=MODEL,
                priors=priors,
                engine="laplace_mh",
                parameterization="fruehwirth_schnatter",
                asis=False,
                mcmc=bx.MCMC(draws=DRAWS, warmup=WARMUP, chains=CHAINS, seed=SEED + 100 * number, progress=PROGRESS),
                laplace=bx.Laplace(mh_steps=MH_STEPS),
                name=name,
                tail=tail,
            )
            laplace_fit.metadata.update(
                {
                    "example": "uccle_laplace_mh",
                    "description": bx.UCCLE_INFO[name]["description"],
                    "start": START,
                    "end": END,
                    "prior_settings": PRIOR_SETTINGS,
                }
            )
            fit_path.parent.mkdir(parents=True, exist_ok=True)
            laplace_fit.save(fit_path)

        if CHAIN_ONLY:
            print(f"One-chain Uccle Laplace-MH fit complete: {name}")
            continue

        table_dir = OUTPUT_DIR / "tables" / name
        figure_dir = OUTPUT_DIR / "figures" / name
        table_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        diagnostics = laplace_fit.diagnostics()
        engine_diagnostics = diagnostics["engine"]
        algorithm_rows.append({"series": name, **engine_diagnostics})

        pd.DataFrame.from_dict(laplace_fit.static_summary(), orient="index").rename_axis("parameter").to_csv(table_dir / "parameters.csv")
        diagnostics["parameters"].to_csv(table_dir / "diagnostics.csv")
        pd.DataFrame([{"metric": key, "value": value} for key, value in diagnostics["engine"].items()]).to_csv(table_dir / "algorithm.csv", index=False)
        eta_draws = laplace_fit.eta_draws(original_scale=True)
        lower, median, upper = np.quantile(eta_draws, [0.05, 0.50, 0.95], axis=0)
        pd.DataFrame({"date": values.index, "month": values.index.month, "observed": values.to_numpy(), "lower": lower, "median": median, "upper": upper}).to_csv(
            table_dir / "trajectory.csv", index=False
        )
        selection = laplace_fit.component_probabilities().reset_index()
        selection.insert(0, "series", name)
        selection.insert(1, "engine", "laplace")
        selection.to_csv(table_dir / "selection.csv", index=False)
        selection_rows.append(selection)
        laplace_fit.structural_model_probabilities().to_csv(table_dir / "models.csv", index=False)
        laplace_fit.component_transition_summary().reset_index().to_csv(table_dir / "switching.csv", index=False)

        predictive = laplace_fit.posterior_predictive(
            draws=PREDICTIVE_DRAWS,
            seed=SEED + 20_000 + number,
        )
        predictive.summary(level=0.90).to_csv(
            table_dir / "posterior_predictive.csv", index=False
        )
        forecast = laplace_fit.forecast(
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
                    "fit": str(fit_path),
                    "series": name,
                    "description": bx.UCCLE_INFO[name]["description"],
                    "tail": tail,
                    "start": str(values.index.min().date()),
                    "end": str(values.index.max().date()),
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

        figure, axis = laplace_fit.plot("predictor", credible_interval=0.90)
        axis.set_title(f"{name}: posterior latent predictor (Laplace-MH)")
        axis.set_ylabel("GEV location / °C")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"trajectory.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, axis = laplace_fit.plot(
            "predictor", credible_interval=0.90, phase=focus_phase
        )
        axis.set_title(f"{name}: {focus_label} GEV-location trajectory (Laplace-MH)")
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
            observed=laplace_fit.observed,
            title=f"{name}: posterior predictive check (Laplace-MH)",
            ylabel="temperature / °C",
        )
        figure = axis.figure
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"posterior_predictive.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        axis = forecast.plot(
            level=0.90,
            history=laplace_fit.observed,
            history_dates=values.index,
            history_points=FORECAST_HISTORY,
            title=f"{name}: posterior predictive forecast (Laplace-MH)",
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
            history=laplace_fit.observed,
            history_dates=values.index,
            history_points=FORECAST_HISTORY,
            title=f"{name}: {focus_label} posterior predictive forecast (Laplace-MH)",
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

        level_history = np.median(laplace_fit.state_original("level"), axis=0)
        axis = forecast.plot(
            level=0.90,
            target="level",
            history=level_history,
            history_dates=values.index,
            history_points=FORECAST_HISTORY,
            title=f"{name}: seasonally adjusted level forecast (Laplace-MH)",
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

        figure, axis = laplace_fit.plot("level", credible_interval=0.90)
        axis.set_title(f"{name}: posterior latent level (Laplace-MH)")
        axis.set_ylabel("GEV level / °C")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"level.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, axis = laplace_fit.plot(
            "level", credible_interval=0.90, show_observed=False
        )
        axis.set_title(f"{name}: posterior latent level (Laplace-MH)")
        axis.set_ylabel("GEV level / °C")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"level_no_observations.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        trend_states = np.asarray(laplace_fit.parameter("state_trend"), dtype=int)
        slope_condition = "dynamic" if np.any(trend_states == 2) else None
        figure, axis = laplace_fit.plot(
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

        figure, axis = laplace_fit.plot("component_probabilities")
        axis.set_title(f"{name}: structural selection (Laplace-MH)")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"selection.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, _ = laplace_fit.plot("process_sds", title=f"{name}: prior to posterior (Laplace-MH)")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"process_sd.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, _ = laplace_fit.plot("parameter_densities", parameters=("sigma", "xi"))
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"gev.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, axis = laplace_fit.plot("season", show_interval=False)
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"season.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        try:
            figure, _ = laplace_fit.plot("endpoint")
        except ValueError:
            print(f"{name}: no finite endpoint in the retained posterior draws.")
        else:
            for extension in FIGURE_FORMATS:
                figure.savefig(figure_dir / f"endpoint.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
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
                f"warning: {name} has low whole-trajectory acceptance "
                f"({state_acceptance:.3f})."
            )
        print(f"Laplace-MH fit complete: {name}")

    if CHAIN_ONLY:
        print(f"One-chain Uccle Laplace-MH outputs: {OUTPUT_DIR}")
        return

    selection_path = OUTPUT_DIR / "tables" / "selection_all.csv"
    pd.concat(selection_rows, ignore_index=True).to_csv(selection_path, index=False)
    pd.DataFrame(algorithm_rows).to_csv(
        OUTPUT_DIR / "tables" / "laplace_mh_diagnostics.csv", index=False
    )
    print(f"Uccle Laplace-MH outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
