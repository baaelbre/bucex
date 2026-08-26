"""Fit Uccle extremes with stationary, linear, RW, or SSVS log scale.

Pick the JSON below for an IDE run, or pass ``--config PATH``.  The script has
no hidden scientific defaults: every model, prior, computational, and output
choice is read from the selected JSON.
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
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import bucex as bx


# Change this one line to pick another readable JSON in an IDE.
DEFAULT_CONFIG_FILE = (
    Path(__file__).parent / "config" / "phi" / "uccle" / "01_txx_stationary.json"
)
SCRIPT_NAME = Path(__file__).stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_FILE)
    settings_path = parser.parse_args().config.expanduser().resolve()
    config = bx.load_config(settings_path)
    source_settings_path = Path(
        config.get("_runner", {}).get("source_config", settings_path)
    ).resolve()

    data_settings = config["data"]
    model_settings = config["model"]
    prior_settings = config["priors"]
    mcmc_settings = config["mcmc"]
    inference = config["inference"]
    runtime = config["runtime"]
    output = config["output"]
    figures = config["figures"]

    data_dir = (
        SOURCE_ROOT / "data"
        if data_settings["data_dir"] is None
        else Path(data_settings["data_dir"])
    )
    series_names = tuple(map(str, data_settings["series"]))
    period = int(data_settings["period"])
    fit_phi = str(model_settings["phi"])
    components = (
        bx.LocalLinearTrend(
            level_mode=str(model_settings["level_mode"]),
            trend_mode=str(model_settings["trend_mode"]),
        ),
        bx.DummySeasonal(
            period=period,
            mode=str(model_settings["seasonal_mode"]),
        ),
    )
    model = bx.Model(
        bx.GEV(xi_bounds=tuple(prior_settings["xi_bounds"]), phi=fit_phi),
        components,
        name=f"Uccle GEV with {fit_phi} log scale",
    )
    phi_prior = bx.PhiPrior(
        linear=bx.NormalPrior(
            float(prior_settings["phi"]["linear"]["mean"]),
            float(prior_settings["phi"]["linear"]["sd"]),
        ),
        rw_variance=bx.InverseGammaPrior(
            float(prior_settings["phi"]["rw_variance"]["a"]),
            float(prior_settings["phi"]["rw_variance"]["b"]),
        ),
        model_probabilities=prior_settings["phi"]["model_probabilities"],
    )

    run_id = output.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    signature = (
        f"phi-{fit_phi}_{'-'.join(name.lower() for name in series_names)}"
        f"_d{int(mcmc_settings['draws'])}w{int(mcmc_settings['warmup'])}"
        f"c{int(mcmc_settings['chains'])}"
    )
    output_dir = Path(output["results_root"]) / SCRIPT_NAME / f"{run_id}__{signature}"
    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not bool(output["overwrite"])
    ):
        raise FileExistsError(
            f"{output_dir} already exists. Choose a new output.run_id or set overwrite=true."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    bx.save_config(config, output_dir / "run_config.json")

    summaries = []
    for series_number, series_name in enumerate(series_names):
        values = bx.load_uccle_series(
            series_name,
            data_dir,
            start=data_settings["start"],
            end=data_settings["end"],
        )
        tail = bx.UCCLE_INFO[series_name]["tail"]
        sign = -1.0 if tail == "min" else 1.0
        transformed = sign * values.to_numpy(float)
        alpha_setting = prior_settings["alpha_mean"]
        alpha_mean = (
            float(np.median(transformed))
            if alpha_setting == "transformed_series_median"
            else float(alpha_setting)
        )
        priors = bx.ssvs_gev_priors(
            period=period,
            alpha_mean=alpha_mean,
            alpha_sd=float(prior_settings["alpha_sd"]),
            beta_mean=float(prior_settings["beta_mean"]),
            beta_sd=float(prior_settings["beta_sd"]),
            seasonal_initial_sd=float(prior_settings["seasonal_initial_sd"]),
            sigma2_prior=bx.InverseGammaPrior(
                float(prior_settings["sigma2"]["a"]),
                float(prior_settings["sigma2"]["b"]),
            ),
            xi_prior=bx.UniformPrior(*map(float, prior_settings["xi_bounds"])),
            xi_max_abs=float(prior_settings["xi_max_abs"]),
            innovation_slab_sd={
                name: float(value)
                for name, value in prior_settings["innovation_slab_sd"].items()
            },
            level_dynamic_probability=float(
                prior_settings["level_dynamic_probability"]
            ),
            trend_probabilities=tuple(
                map(float, prior_settings["trend_probabilities"])
            ),
            season_probabilities=tuple(
                map(float, prior_settings["season_probabilities"])
            ),
            phi_prior=phi_prior,
        )

        fit_path = output_dir / "fits" / f"{series_name}.bucex"
        if runtime.get("combine_runs"):
            fit = bx.combine_fits(
                [
                    bx.FitResult.load(Path(directory) / "fits" / f"{series_name}.bucex")
                    for directory in runtime["combine_runs"]
                ]
            )
        else:
            phi_options = inference["phi"]
            fit = bx.fit(
                values,
                model=model,
                priors=priors,
                engine=str(inference["engine"]),
                parameterization=str(inference["parameterization"]),
                asis=bool(inference["asis"]),
                mcmc=bx.MCMC(
                    draws=int(mcmc_settings["draws"]),
                    warmup=int(mcmc_settings["warmup"]),
                    thin=int(mcmc_settings["thin"]),
                    chains=int(mcmc_settings["chains"]),
                    seed=int(mcmc_settings["seed"]) + 100 * series_number,
                    progress=bool(runtime["progress"]),
                ),
                laplace=bx.Laplace(
                    max_iterations=int(inference["laplace"]["max_iterations"]),
                    tolerance=float(inference["laplace"]["tolerance"]),
                    curvature_floor=float(inference["laplace"]["curvature_floor"]),
                    maximum_variance=float(inference["laplace"]["maximum_variance"]),
                    draw_attempts=int(inference["laplace"]["draw_attempts"]),
                    mh_steps=int(inference["laplace"]["mh_steps"]),
                ),
                state_kwargs={
                    "phi_step_intercept": float(phi_options["step_intercept"]),
                    "phi_step_linear": float(phi_options["step_linear"]),
                    "phi_laplace_max_iterations": int(
                        phi_options["laplace_max_iterations"]
                    ),
                    "phi_laplace_tolerance": float(phi_options["laplace_tolerance"]),
                    "phi_curvature_floor": float(phi_options["curvature_floor"]),
                    "phi_maximum_variance": float(phi_options["maximum_variance"]),
                    "phi_shift_limit": float(phi_options["shift_limit"]),
                    "phi_laplace_mh_steps": int(phi_options["mh_steps"]),
                    "phi_draw_attempts": int(phi_options["draw_attempts"]),
                },
                dates=values.index.to_numpy(),
                name=series_name,
                tail=tail,
            )
        fit.metadata.update(
            example="uccle_phi",
            settings_file=str(source_settings_path),
            fitted_phi=fit_phi,
            description=bx.UCCLE_INFO[series_name]["description"],
        )
        fit_path.parent.mkdir(parents=True, exist_ok=True)
        fit.save(fit_path)
        print(f"Saved {fit_path}")

        if bool(runtime["chain_only"]):
            continue

        phi_draws = fit.phi_draws()
        lower, median, upper = np.quantile(phi_draws, [0.05, 0.5, 0.95], axis=0)
        pd.DataFrame(
            {
                "date": values.index,
                "lower": lower,
                "median": median,
                "upper": upper,
                "sigma_median": np.exp(median),
            }
        ).to_csv(output_dir / f"phi_{series_name}.csv", index=False)
        probabilities = fit.phi_model_probabilities() if fit_phi == "ssvs" else None
        summaries.append(
            {
                "series": series_name,
                "phi": fit_phi,
                "probability_stationary": (
                    probabilities["stationary"] if probabilities else None
                ),
                "probability_linear": probabilities["linear"]
                if probabilities
                else None,
                "probability_rw": probabilities["rw"] if probabilities else None,
                "phi_slope_median": (
                    float(np.median(fit.parameter("phi_slope")))
                    if "phi_slope" in fit.parameter_draws
                    else None
                ),
                "phi_rw_sd_median": (
                    float(np.median(fit.parameter("phi_rw_sd")))
                    if "phi_rw_sd" in fit.parameter_draws
                    else None
                ),
            }
        )

        if bool(figures["enabled"]):
            figure, axis = plt.subplots(figsize=(10, 4))
            axis.fill_between(values.index, lower, upper, alpha=0.2)
            axis.plot(values.index, median, color="C0")
            axis.set(ylabel=r"$\phi_t=\log(\sigma_t)$", xlabel="date")
            title = bx.config_title(config, series_name)
            if title:
                axis.set_title(title)
            figure.savefig(
                output_dir / f"phi_{series_name}.{figures['format']}",
                dpi=int(figures["dpi"]),
                bbox_inches="tight",
            )
            plt.close(figure)

    if not bool(runtime["chain_only"]):
        pd.DataFrame(summaries).to_csv(output_dir / "summary.csv", index=False)
        (output_dir / "summary.json").write_text(
            json.dumps(summaries, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
