"""Fit a simulated GEV series with stationary or time-varying log scale.

Pick the JSON below for an IDE run, or pass ``--config PATH``.  Every model,
prior, MCMC, Laplace, plotting, and output setting is read directly from JSON.
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
    Path(__file__).parent / "config" / "phi" / "simulation_stationary.json"
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

    simulation = config["simulation"]
    model_settings = config["model"]
    prior_settings = config["priors"]
    mcmc_settings = config["mcmc"]
    inference = config["inference"]
    runtime = config["runtime"]
    output = config["output"]

    n_time = int(simulation["n_time"])
    period = int(simulation["period"])
    truth_phi = str(simulation["phi"]["mode"])
    reference_phi = float(np.log(float(simulation["phi"]["reference_sigma"])))
    basis = (
        np.zeros(n_time)
        if n_time == 1
        else (np.arange(n_time) - 0.5 * (n_time - 1)) / (n_time - 1)
    )
    if truth_phi == "stationary":
        phi_truth = np.full(n_time, reference_phi)
    elif truth_phi == "linear":
        phi_truth = reference_phi + float(simulation["phi"]["linear_change"]) * basis
    elif truth_phi == "rw":
        phi_rng = np.random.default_rng(int(simulation["seed"]) + 1)
        phi_truth = reference_phi + np.cumsum(
            phi_rng.normal(scale=float(simulation["phi"]["rw_sd"]), size=n_time)
        )
        phi_truth -= float(np.mean(phi_truth) - reference_phi)
    else:
        raise ValueError("simulation.phi.mode must be stationary, linear, or rw.")

    phase = np.arange(period, dtype=float)
    seasonal = -float(simulation["seasonal_amplitude"]) * np.cos(
        2.0 * np.pi * phase / period
    )
    seasonal -= seasonal.mean()
    components = (
        bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
        bx.DummySeasonal(period=period, mode="dynamic"),
    )
    simulation_model = bx.Model(bx.GEV(), components, name="dynamic-location GEV truth")
    simulated = bx.simulate(
        simulation_model,
        n_time,
        {
            "sigma": np.exp(phi_truth),
            "xi": float(simulation["xi"]),
            "sd.level": float(simulation["location_sd"]["level"]),
            "sd.slope": float(simulation["location_sd"]["slope"]),
            "sd.seasonal": float(simulation["location_sd"]["seasonal"]),
        },
        initial_state=np.r_[
            float(simulation["initial_level"]),
            float(simulation["initial_slope"]),
            seasonal[1:],
        ],
        seed=int(simulation["seed"]),
    )

    fit_phi = str(model_settings["phi"])
    fit_model = bx.Model(
        bx.GEV(xi_bounds=tuple(prior_settings["xi_bounds"]), phi=fit_phi),
        components,
        name=f"GEV with {fit_phi} log scale",
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
    priors = bx.ssvs_gev_priors(
        period=period,
        alpha_mean=float(np.median(simulated.y)),
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
        level_dynamic_probability=float(prior_settings["level_dynamic_probability"]),
        trend_probabilities=tuple(map(float, prior_settings["trend_probabilities"])),
        season_probabilities=tuple(map(float, prior_settings["season_probabilities"])),
        phi_prior=phi_prior,
    )

    run_id = output.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    signature = (
        f"phi-{fit_phi}_n{n_time}p{period}_d{int(mcmc_settings['draws'])}"
        f"w{int(mcmc_settings['warmup'])}c{int(mcmc_settings['chains'])}"
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
    pd.DataFrame(
        {
            "time": np.arange(n_time),
            "y": simulated.y,
            "mu": simulated.eta,
            "phi": phi_truth,
            "sigma": np.exp(phi_truth),
        }
    ).to_csv(output_dir / "simulation.csv", index=False)

    if runtime.get("combine_runs"):
        source_fits = [
            bx.FitResult.load(Path(directory) / "fit.bucex")
            for directory in runtime["combine_runs"]
        ]
        fit = bx.combine_fits(source_fits)
    else:
        phi_options = inference["phi"]
        fit = bx.fit(
            simulated.y,
            model=fit_model,
            priors=priors,
            engine=str(inference["engine"]),
            parameterization=str(inference["parameterization"]),
            asis=bool(inference["asis"]),
            mcmc=bx.MCMC(
                draws=int(mcmc_settings["draws"]),
                warmup=int(mcmc_settings["warmup"]),
                thin=int(mcmc_settings["thin"]),
                chains=int(mcmc_settings["chains"]),
                seed=int(mcmc_settings["seed"]),
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
            name=f"simulation: truth={truth_phi}, fit={fit_phi}",
        )
    fit.metadata.update(
        example="simulation_phi",
        settings_file=str(source_settings_path),
        truth_phi=truth_phi,
        fitted_phi=fit_phi,
    )
    fit.save(output_dir / "fit.bucex")
    print(f"Saved {output_dir / 'fit.bucex'}")

    if bool(runtime["chain_only"]):
        return

    phi_draws = fit.phi_draws()
    lower, median, upper = np.quantile(phi_draws, [0.05, 0.5, 0.95], axis=0)
    pd.DataFrame(
        {
            "time": np.arange(n_time),
            "truth": phi_truth,
            "lower": lower,
            "median": median,
            "upper": upper,
        }
    ).to_csv(output_dir / "phi_summary.csv", index=False)
    summary = {
        "truth_phi": truth_phi,
        "fitted_phi": fit_phi,
        "phi_model_probabilities": (
            fit.phi_model_probabilities() if fit_phi == "ssvs" else None
        ),
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
        "acceptance": fit.acceptance,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if bool(config["figures"]["enabled"]):
        figure, axis = plt.subplots(figsize=(9, 4))
        axis.fill_between(np.arange(n_time), lower, upper, alpha=0.2)
        axis.plot(phi_truth, color="black", linewidth=1.0, label="truth")
        axis.plot(median, color="C0", label="posterior median")
        axis.set(xlabel="time", ylabel=r"$\phi_t=\log(\sigma_t)$")
        title = bx.config_title(config, "phi")
        if title:
            axis.set_title(title)
        axis.legend()
        figure.savefig(
            output_dir / f"phi.{config['figures']['format']}",
            dpi=int(config["figures"]["dpi"]),
            bbox_inches="tight",
        )
        plt.close(figure)


if __name__ == "__main__":
    main()
