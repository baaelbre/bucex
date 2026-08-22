"""Fit a random-walk GEV with centred states and inverse-gamma priors.

This is the deliberately difficult reference analysis discussed in the paper:
the latent level is sampled in the centred parameterisation, while its small
innovation variance has an inverse-gamma prior.  The script therefore saves
the trace, ACF, ESS, and R-hat diagnostics needed to see whether the apparently
reasonable latent trajectory is supported by a well-mixing Markov chain.

Run with ``python examples/07_centered_ig_random_walk_gev.py``.
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


# Results. The directory name stays short; run_config.json records every
# simulation, prior, approximation, and MCMC setting.
RESULTS_ROOT = Path(os.environ.get("BUCEX_RESULTS_ROOT", "results"))
SCRIPT_NAME = Path(__file__).stem
RUN_TIMESTAMP = os.environ.get("BUCEX_RUN_ID") or datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)
OVERWRITE = os.environ.get("BUCEX_OVERWRITE", "0").lower() in {
    "1",
    "true",
    "yes",
}

# Random-walk GEV truth. The process SD is deliberately small relative to the
# GEV scale, which is the scientifically interesting slow-signal regime.
N_TIME = int(os.environ.get("BUCEX_N_TIME", "1000"))
SIGMA = float(os.environ.get("BUCEX_SIGMA", "1.50"))
XI = float(os.environ.get("BUCEX_XI", "-0.30"))
INITIAL_LEVEL = float(os.environ.get("BUCEX_INITIAL_LEVEL", "25.0"))
RANDOM_WALK_SD = float(os.environ.get("BUCEX_RANDOM_WALK_SD", "0.05"))
SIMULATION_SEED = int(os.environ.get("BUCEX_SIMULATION_SEED", "13081997"))

# Initial-state prior. This is separate from the inverse-gamma prior on the
# process variance q_level.
INITIAL_LEVEL_PRIOR_MEAN = float(
    os.environ.get("BUCEX_INITIAL_LEVEL_PRIOR_MEAN", "25.0")
)
INITIAL_LEVEL_PRIOR_SD = float(
    os.environ.get("BUCEX_INITIAL_LEVEL_PRIOR_SD", "3.2")
)

# Inverse-gamma priors use shape/scale form: q ~ IG(a, b). If a > 1, the
# prior mean is b / (a - 1), while its mode is b / (a + 1). Here the process
# prior is calibrated to the simulation scale. Change these four numbers at
# the top of the file for the sensitivity experiment; very small a and b
# reproduce the traditional diffuse-IG specification.
LEVEL_VARIANCE_IG_A = float(os.environ.get("BUCEX_LEVEL_IG_A", "2.0"))
LEVEL_VARIANCE_IG_B = float(os.environ.get("BUCEX_LEVEL_IG_B", "0.0025"))
SIGMA_VARIANCE_IG_A = float(os.environ.get("BUCEX_SIGMA_IG_A", "2.0"))
SIGMA_VARIANCE_IG_B = float(os.environ.get("BUCEX_SIGMA_IG_B", "2.25"))

# A weakly regularising, zero-centred shape prior avoids making the finite
# support bounds the complete prior specification.
XI_PRIOR_MEAN = float(os.environ.get("BUCEX_XI_PRIOR_MEAN", "0.0"))
XI_PRIOR_SD = float(os.environ.get("BUCEX_XI_PRIOR_SD", "0.20"))
XI_PRIOR_BOUNDS = (-0.50, 0.50)

# MCMC and particle settings. Four independent chains are important here: the
# purpose of this example is to diagnose the centred/IG geometry, not only draw
# a smooth path. PGAS keeps the GEV observation update exact; it also lets us
# distinguish parameter mixing from particle-path degeneracy.
DRAWS = int(os.environ.get("BUCEX_DRAWS", "1000"))
WARMUP = int(os.environ.get("BUCEX_WARMUP", "1000"))
CHAINS = int(os.environ.get("BUCEX_CHAINS", "4"))
PARTICLES = int(os.environ.get("BUCEX_PARTICLES", "256"))
SEED = int(os.environ.get("BUCEX_SEED", "13081997"))
PROGRESS = os.environ.get("BUCEX_PROGRESS", "1").lower() not in {
    "0",
    "false",
    "no",
}
CHAIN_ONLY = os.environ.get("BUCEX_CHAIN_ONLY", "0").lower() in {
    "1",
    "true",
    "yes",
}
COMBINE_RUNS = tuple(
    Path(value)
    for value in os.environ.get("BUCEX_COMBINE_RUNS", "").split(os.pathsep)
    if value
)

# Output settings.
PREDICTIVE_DRAWS = int(os.environ.get("BUCEX_PREDICTIVE_DRAWS", "500"))
FORECAST_HORIZON = int(os.environ.get("BUCEX_FORECAST_HORIZON", "40"))
FORECAST_HISTORY = int(os.environ.get("BUCEX_FORECAST_HISTORY", "200"))
MAX_ACF_LAG = int(os.environ.get("BUCEX_MAX_ACF_LAG", "100"))
FIGURE_FORMATS = ("pdf", "png")
FIGURE_DPI = 180

RUN_SIGNATURE = (
    f"n{N_TIME}_p{PARTICLES}_d{DRAWS}w{WARMUP}c{CHAINS}_s{SIMULATION_SEED}"
)
OUTPUT_DIR = RESULTS_ROOT / SCRIPT_NAME / f"{RUN_TIMESTAMP}__{RUN_SIGNATURE}"


# One random-walk level and one direct GEV observation equation. There is no
# auxiliary Gaussian observation layer in this model.
MODEL = bx.Model(
    bx.GEV(xi_bounds=XI_PRIOR_BOUNDS),
    (
        bx.LocalLevel(
            mode="dynamic",
            initial_mean=INITIAL_LEVEL_PRIOR_MEAN,
            initial_sd=INITIAL_LEVEL_PRIOR_SD,
        ),
    ),
    name="centred random-walk GEV",
)

# q_level = sd.level**2 and sigma**2 both receive explicit inverse-gamma
# priors. This is intentionally not an FS/non-centred shrinkage prior.
PRIORS = bx.Priors(
    process={
        "level": bx.InverseGammaVariance(
            shape=LEVEL_VARIANCE_IG_A,
            scale=LEVEL_VARIANCE_IG_B,
        )
    },
    observation_sd=bx.InverseGammaVariance(
        shape=SIGMA_VARIANCE_IG_A,
        scale=SIGMA_VARIANCE_IG_B,
    ),
    shape=bx.TruncatedNormalPrior(
        mean=XI_PRIOR_MEAN,
        sd=XI_PRIOR_SD,
        lower=XI_PRIOR_BOUNDS[0],
        upper=XI_PRIOR_BOUNDS[1],
    ),
    profile="centred_inverse_gamma",
    metadata={
        "purpose": "centred random-walk GEV benchmark",
        "process_variance_parameter": "q_level = sd.level**2",
    },
)

PRIOR_SETTINGS = {
    "initial_level": {
        "mean": INITIAL_LEVEL_PRIOR_MEAN,
        "sd": INITIAL_LEVEL_PRIOR_SD,
    },
    "level_variance": {
        "distribution": "inverse_gamma",
        "shape": LEVEL_VARIANCE_IG_A,
        "scale": LEVEL_VARIANCE_IG_B,
    },
    "sigma_variance": {
        "distribution": "inverse_gamma",
        "shape": SIGMA_VARIANCE_IG_A,
        "scale": SIGMA_VARIANCE_IG_B,
    },
    "xi": {
        "distribution": "truncated_normal",
        "mean": XI_PRIOR_MEAN,
        "sd": XI_PRIOR_SD,
        "bounds": list(XI_PRIOR_BOUNDS),
    },
}


def main() -> None:
    if COMBINE_RUNS and len(COMBINE_RUNS) != CHAINS:
        raise ValueError(
            f"BUCEX_COMBINE_RUNS contains {len(COMBINE_RUNS)} runs, "
            f"but BUCEX_CHAINS={CHAINS}."
        )

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )

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
        "model": MODEL.to_dict(),
        "simulation": {
            "n_time": N_TIME,
            "initial_level": INITIAL_LEVEL,
            "random_walk_sd": RANDOM_WALK_SD,
            "random_walk_variance": RANDOM_WALK_SD**2,
            "sigma": SIGMA,
            "xi": XI,
            "seed": SIMULATION_SEED,
        },
        "priors": PRIOR_SETTINGS,
        "inference": {
            "engine": "pgas",
            "parameterization": "centered",
            "asis": False,
            "particles": PARTICLES,
            "proposal": "guided",
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
            "predictive_draws": PREDICTIVE_DRAWS,
            "forecast_horizon": FORECAST_HORIZON,
            "forecast_history": FORECAST_HISTORY,
            "max_acf_lag": MAX_ACF_LAG,
        },
        "chain_only": CHAIN_ONLY,
        "combined_chain_runs": [str(path) for path in COMBINE_RUNS],
    }
    config_path.write_text(
        json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Simulate from the same public model used for fitting.
    simulation = bx.simulate(
        MODEL,
        N_TIME,
        {"sd.level": RANDOM_WALK_SD, "sigma": SIGMA, "xi": XI},
        initial_state=[INITIAL_LEVEL],
        seed=SIMULATION_SEED,
    )
    time = np.arange(N_TIME)
    true_level = simulation.states[1:, MODEL.state_names.index("level")]
    simulation_table = pd.DataFrame(
        {
            "time": time,
            "observed": simulation.y,
            "true_level": true_level,
        }
    )
    table_dir = OUTPUT_DIR / "tables"
    figure_dir = OUTPUT_DIR / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    simulation_table.to_csv(table_dir / "simulation.csv", index=False)

    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.scatter(time, simulation.y, s=10, color="0.55", alpha=0.5, label="observed")
    axis.plot(time, true_level, color="#123B4A", linewidth=1.3, label="true level")
    axis.set_xlabel("time")
    axis.set_ylabel("GEV location / °C")
    axis.set_title("Random-walk GEV simulation")
    axis.legend()
    for extension in FIGURE_FORMATS:
        figure.savefig(
            figure_dir / f"simulation.{extension}",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
    plt.close(figure)

    # Fit or combine independent chain runs. The key lines are intentionally
    # explicit: centered states, no ASIS rescue, and the PRIORS object above.
    fit_path = OUTPUT_DIR / "fits" / "combined.bucex"
    if fit_path.is_file() and not OVERWRITE:
        fit = bx.FitResult.load(fit_path)
        if (
            fit.n_time != N_TIME
            or fit.n_chains != CHAINS
            or fit.draws_per_chain != DRAWS
            or fit.plan.engine != "pgas"
            or fit.plan.parameterization != "centered"
            or fit.plan.asis
            or fit.metadata.get("prior_settings") != PRIOR_SETTINGS
        ):
            raise ValueError(
                f"{fit_path} does not match the current model, prior, or MCMC settings."
            )
        print(f"Reusing {fit_path}")
    elif COMBINE_RUNS:
        source_paths = [run_dir / "fits" / "combined.bucex" for run_dir in COMBINE_RUNS]
        missing = [path for path in source_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Cannot combine centred-IG chains; missing fit file(s):\n  "
                + "\n  ".join(str(path) for path in missing)
            )
        fit = bx.combine_fits([bx.FitResult.load(path) for path in source_paths])
        fit.metadata["combined_chain_sources"] = [str(path) for path in source_paths]
        fit_path.parent.mkdir(parents=True, exist_ok=True)
        fit.save(fit_path)
    else:
        fit = bx.fit(
            simulation.y,
            model=MODEL,
            priors=PRIORS,
            engine="pgas",
            parameterization="centered",
            asis=False,
            particles=bx.Particles(n=PARTICLES, proposal="guided"),
            mcmc=bx.MCMC(
                draws=DRAWS,
                warmup=WARMUP,
                chains=CHAINS,
                seed=SEED,
                progress=PROGRESS,
            ),
            name="centered inverse-gamma random-walk GEV",
        )
        fit.metadata.update(
            {
                "example": "centered_ig_random_walk_gev",
                "truth": {
                    "sd.level": RANDOM_WALK_SD,
                    "q_level": RANDOM_WALK_SD**2,
                    "sigma": SIGMA,
                    "xi": XI,
                },
                "prior_settings": PRIOR_SETTINGS,
            }
        )
        fit_path.parent.mkdir(parents=True, exist_ok=True)
        fit.save(fit_path)

    if CHAIN_ONLY:
        print(f"One-chain centred-IG fit complete: {fit_path}")
        return

    # Parameter and algorithm diagnostics. Poor ESS, large R-hat, or slowly
    # decaying ACFs are the result of interest in this benchmark.
    diagnostics = fit.diagnostics()
    static_summary = pd.DataFrame.from_dict(
        fit.static_summary(), orient="index"
    ).rename_axis("parameter")
    static_summary.to_csv(table_dir / "parameters.csv")
    diagnostics["parameters"].to_csv(table_dir / "diagnostics.csv")
    pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in diagnostics["engine"].items()
        ]
    ).to_csv(table_dir / "algorithm.csv", index=False)

    sd_draws = np.asarray(fit.parameter("sd.level", combine_chains=False))
    q_draws = sd_draws**2
    q_summary = pd.DataFrame(
        {
            "chain": np.arange(1, fit.n_chains + 1),
            "mean": q_draws.mean(axis=1),
            "median": np.median(q_draws, axis=1),
            "lower_90": np.quantile(q_draws, 0.05, axis=1),
            "upper_90": np.quantile(q_draws, 0.95, axis=1),
        }
    )
    q_summary.to_csv(table_dir / "process_variance_by_chain.csv", index=False)

    eta_draws = fit.eta_draws(original_scale=True)
    lower, median, upper = np.quantile(eta_draws, [0.05, 0.50, 0.95], axis=0)
    pd.DataFrame(
        {
            "time": time,
            "observed": fit.observed,
            "lower": lower,
            "median": median,
            "upper": upper,
            "true_level": true_level,
        }
    ).to_csv(table_dir / "trajectory.csv", index=False)

    predictive = fit.posterior_predictive(draws=PREDICTIVE_DRAWS, seed=SEED + 1)
    predictive.summary(level=0.90).to_csv(
        table_dir / "posterior_predictive.csv", index=False
    )
    forecast = fit.forecast(
        FORECAST_HORIZON,
        draws=PREDICTIVE_DRAWS,
        seed=SEED + 2,
    )
    forecast.summary(level=0.90).to_csv(table_dir / "forecast.csv", index=False)

    (table_dir / "summary.json").write_text(
        json.dumps(
            {
                "fit": str(fit_path),
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

    # Posterior state and parameter figures.
    figure, axis = fit.plot(
        "level",
        credible_interval=0.90,
        truth=true_level,
    )
    axis.set_title("Posterior random-walk level")
    for extension in FIGURE_FORMATS:
        figure.savefig(
            figure_dir / f"level.{extension}",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
    plt.close(figure)

    figure, axis = fit.plot(
        "level",
        credible_interval=0.90,
        truth=true_level,
        show_observed=False,
    )
    axis.set_title("Posterior random-walk level")
    for extension in FIGURE_FORMATS:
        figure.savefig(
            figure_dir / f"level_no_observations.{extension}",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
    plt.close(figure)

    figure, _ = fit.plot(
        "process_sds",
        truths={"sd.level": RANDOM_WALK_SD},
        title="Random-walk innovation SD: prior to posterior",
    )
    for extension in FIGURE_FORMATS:
        figure.savefig(
            figure_dir / f"process_sd.{extension}",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
    plt.close(figure)

    figure, _ = fit.plot(
        "parameter_densities",
        parameters=("sigma", "xi"),
        truths={"sigma": SIGMA, "xi": XI},
    )
    for extension in FIGURE_FORMATS:
        figure.savefig(
            figure_dir / f"gev_parameters.{extension}",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
    plt.close(figure)

    # Trace and autocorrelation plots are always produced in this diagnostic
    # example, rather than hidden behind a presentation-only switch.
    diagnostic_parameters = ("sd.level", "sigma", "xi")
    figure, _ = fit.plot(
        "traces",
        parameters=diagnostic_parameters,
        truths={"sd.level": RANDOM_WALK_SD, "sigma": SIGMA, "xi": XI},
    )
    for extension in FIGURE_FORMATS:
        figure.savefig(
            figure_dir / f"parameter_traces.{extension}",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
    plt.close(figure)

    figure, _ = fit.plot(
        "acf",
        parameters=diagnostic_parameters,
        max_lag=min(MAX_ACF_LAG, max(DRAWS - 1, 1)),
    )
    for extension in FIGURE_FORMATS:
        figure.savefig(
            figure_dir / f"parameter_acf.{extension}",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4.5))
    for chain_index, chain in enumerate(q_draws):
        axis.plot(chain, linewidth=0.8, alpha=0.8, label=f"chain {chain_index + 1}")
    axis.axhline(
        RANDOM_WALK_SD**2,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="truth",
    )
    axis.set_xlabel("retained draw")
    axis.set_ylabel(r"level innovation variance $q_\mu$")
    axis.set_title("Centered process-variance trace")
    axis.legend(ncol=min(fit.n_chains + 1, 5))
    for extension in FIGURE_FORMATS:
        figure.savefig(
            figure_dir / f"process_variance_trace.{extension}",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
    plt.close(figure)

    axis = predictive.plot(
        level=0.90,
        observed=fit.observed,
        title="Posterior predictive check",
        ylabel="temperature / °C",
    )
    for extension in FIGURE_FORMATS:
        axis.figure.savefig(
            figure_dir / f"posterior_predictive.{extension}",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
    plt.close(axis.figure)

    axis = forecast.plot(
        level=0.90,
        history=fit.observed,
        history_dates=time,
        history_points=FORECAST_HISTORY,
        title="Posterior predictive forecast",
        ylabel="temperature / °C",
    )
    for extension in FIGURE_FORMATS:
        axis.figure.savefig(
            figure_dir / f"forecast.{extension}",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
    plt.close(axis.figure)

    print("Centered random-walk GEV with inverse-gamma priors complete")
    print(f"Outputs: {OUTPUT_DIR}")
    print(diagnostics["parameters"].loc[list(diagnostic_parameters)])


if __name__ == "__main__":
    main()
