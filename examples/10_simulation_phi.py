"""Fit one simulated GEV series with a configurable log-scale model.

Pick the JSON below for an IDE run, or pass ``--config PATH``. The JSON
separates the data-generating truth from the fitted location and scale models.
After fitting, this script writes the same tables and figures as the earlier
simulation examples, plus log-scale-specific summaries.
"""

from __future__ import annotations
# ruff: noqa: I001

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

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
    truth_location = simulation["location"]
    model_settings = config["model"]
    fitted_location = model_settings["location"]
    prior_settings = config["priors"]
    mcmc_settings = config["mcmc"]
    inference = config["inference"]
    figures = config["figures"]
    runtime = config["runtime"]
    output = config["output"]

    # The component names are settings, not silent Python defaults. The
    # focused example deliberately supports one clear location family.
    for label, location in (
        ("simulation.location", truth_location),
        ("model.location", fitted_location),
    ):
        if (
            str(location["trend_component"]).lower() != "local_linear_trend"
            or str(location["seasonal_component"]).lower() != "dummy"
        ):
            raise ValueError(
                f"{label} currently supports local_linear_trend plus dummy "
                "seasonality."
            )
    if str(fitted_location["structure"]).lower() != "ssvs":
        raise ValueError("model.location.structure must be 'ssvs'.")

    n_time = int(simulation["n_time"])
    period = int(simulation["period"])
    fit_phi = str(model_settings["phi"])
    truth_phi = str(simulation["phi"]["mode"])
    reference_sigma = float(simulation["phi"]["reference_sigma"])
    reference_phi = float(np.log(reference_sigma))

    # The centered linear basis makes linear_change the complete first-to-last
    # change in phi=log(sigma), rather than a change per observation.
    basis = (
        np.zeros(n_time)
        if n_time == 1
        else (np.arange(n_time) - 0.5 * (n_time - 1)) / (n_time - 1)
    )
    if truth_phi == "stationary":
        phi_truth = np.full(n_time, reference_phi)
    elif truth_phi == "linear":
        phi_truth = (
            reference_phi
            + float(simulation["phi"]["linear_change"]) * basis
        )
    elif truth_phi == "rw":
        phi_rng = np.random.default_rng(int(simulation["seed"]) + 1)
        phi_truth = reference_phi + np.cumsum(
            phi_rng.normal(
                scale=float(simulation["phi"]["rw_sd"]), size=n_time
            )
        )
        phi_truth -= float(np.mean(phi_truth) - reference_phi)
    else:
        raise ValueError("simulation.phi.mode must be stationary, linear, or rw.")

    phase = np.arange(period, dtype=float)
    seasonal_cycle = -float(truth_location["seasonal_amplitude"]) * np.cos(
        2.0 * np.pi * phase / period
    )
    seasonal_cycle -= seasonal_cycle.mean()

    truth_trend = bx.LocalLinearTrend(
        level_mode=str(truth_location["level_mode"]),
        trend_mode=str(truth_location["trend_mode"]),
    )
    truth_season = bx.DummySeasonal(
        period=period,
        mode=str(truth_location["seasonal_mode"]),
    )
    simulation_model = bx.Model(
        bx.GEV(),
        (truth_trend, truth_season),
        name="configured simulation truth",
    )
    simulation_params = {
        "sigma": np.exp(phi_truth),
        "xi": float(simulation["xi"]),
        "sd.level": float(truth_location["level_innovation_sd"]),
        "sd.slope": float(truth_location["trend_innovation_sd"]),
        "sd.seasonal": float(truth_location["seasonal_innovation_sd"]),
    }
    initial_state = [float(truth_location["initial_level"])]
    if str(truth_location["trend_mode"]) != "off":
        initial_state.append(float(truth_location["initial_slope"]))
    if str(truth_location["seasonal_mode"]) != "off":
        initial_state.extend(seasonal_cycle[1:].tolist())

    simulated = bx.simulate(
        simulation_model,
        n_time,
        simulation_params,
        initial_state=np.asarray(initial_state),
        seed=int(simulation["seed"]),
    )
    states = simulated.states[1:]
    state_names = simulation_model.state_names
    seasonal_name = next(
        (name for name in state_names if name.startswith("seasonal[")), None
    )
    simulation_table = pd.DataFrame(
        {
            "time": np.arange(1, n_time + 1),
            "cycle": np.arange(n_time) // period + 1,
            "phase": np.arange(n_time) % period + 1,
            "y": simulated.y,
            "eta": simulated.eta,
            "level": states[:, state_names.index("level")],
            "slope": (
                states[:, state_names.index("slope")]
                if "slope" in state_names
                else np.zeros(n_time)
            ),
            "seasonal": (
                states[:, state_names.index(seasonal_name)]
                if seasonal_name is not None
                else np.zeros(n_time)
            ),
            "phi": phi_truth,
            "sigma": np.exp(phi_truth),
        }
    )

    fitted_components = (
        bx.LocalLinearTrend(
            level_mode=str(fitted_location["level_mode"]),
            trend_mode=str(fitted_location["trend_mode"]),
        ),
        bx.DummySeasonal(
            period=period,
            mode=str(fitted_location["seasonal_mode"]),
        ),
    )
    fit_model = bx.Model(
        bx.GEV(xi_bounds=tuple(prior_settings["xi_bounds"]), phi=fit_phi),
        fitted_components,
        name=f"GEV with structural location SSVS and {fit_phi} log scale",
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
    alpha_setting = prior_settings["alpha_mean"]
    alpha_mean = (
        float(np.median(simulated.y))
        if alpha_setting == "simulated_series_median"
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

    draws = int(mcmc_settings["draws"])
    warmup = int(mcmc_settings["warmup"])
    chains = int(mcmc_settings["chains"])
    run_id = output.get("run_id") or datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    signature = (
        f"phi-{fit_phi}_n{n_time}p{period}_d{draws}w{warmup}c{chains}"
    )
    output_dir = (
        Path(output["results_root"]) / SCRIPT_NAME / f"{run_id}__{signature}"
    )
    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not bool(output["overwrite"])
    ):
        raise FileExistsError(
            f"{output_dir} already exists. Choose a new output.run_id or set "
            "overwrite=true."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    bx.save_config(config, output_dir / "run_config.json")

    case_key = f"phi_{fit_phi}"
    simulation_path = output_dir / "simulations" / f"{case_key}.csv"
    truth_path = simulation_path.with_suffix(".json")
    simulation_path.parent.mkdir(parents=True, exist_ok=True)
    simulation_table.to_csv(simulation_path, index=False)

    structural_truth = {
        "level": 2 if str(truth_location["level_mode"]) == "dynamic" else 1,
        "slope": {"off": 0, "static": 1, "dynamic": 2}[
            str(truth_location["trend_mode"])
        ],
        "seasonal": {"off": 0, "static": 1, "dynamic": 2}[
            str(truth_location["seasonal_mode"])
        ],
    }
    parameter_truth = {
        "xi": float(simulation["xi"]),
        "sd.level": (
            float(truth_location["level_innovation_sd"])
            if structural_truth["level"] == 2
            else 0.0
        ),
        "sd.slope": (
            float(truth_location["trend_innovation_sd"])
            if structural_truth["slope"] == 2
            else 0.0
        ),
        "sd.seasonal": (
            float(truth_location["seasonal_innovation_sd"])
            if structural_truth["seasonal"] == 2
            else 0.0
        ),
        "sigma": reference_sigma,
        "phi_intercept": reference_phi,
        "phi_slope": (
            float(simulation["phi"]["linear_change"])
            if truth_phi == "linear"
            else 0.0
        ),
        "phi_rw_sd": (
            float(simulation["phi"]["rw_sd"]) if truth_phi == "rw" else 0.0
        ),
    }
    truth_record = {
        "name": case_key,
        "n_time": n_time,
        "period": period,
        "model": simulation_model.to_dict(),
        "initial_state": initial_state,
        "location": truth_location,
        "phi": simulation["phi"],
        "xi": float(simulation["xi"]),
        "structural_truth": structural_truth,
        "parameter_truth": parameter_truth,
        "seed": int(simulation["seed"]),
    }
    truth_path.write_text(
        json.dumps(truth_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fit_path = output_dir / "fits" / case_key / "combined.bucex"
    if runtime.get("combine_runs"):
        source_paths = [
            Path(directory) / "fits" / case_key / "combined.bucex"
            for directory in runtime["combine_runs"]
        ]
        missing = [path for path in source_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Cannot combine chains; missing fit file(s):\n  "
                + "\n  ".join(str(path) for path in missing)
            )
        fit = bx.combine_fits([bx.FitResult.load(path) for path in source_paths])
        fit.metadata["combined_chain_sources"] = [
            str(path) for path in source_paths
        ]
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
                draws=draws,
                warmup=warmup,
                thin=int(mcmc_settings["thin"]),
                chains=chains,
                seed=int(mcmc_settings["seed"]),
                progress=bool(runtime["progress"]),
            ),
            laplace=bx.Laplace(
                max_iterations=int(inference["laplace"]["max_iterations"]),
                tolerance=float(inference["laplace"]["tolerance"]),
                curvature_floor=float(
                    inference["laplace"]["curvature_floor"]
                ),
                maximum_variance=float(
                    inference["laplace"]["maximum_variance"]
                ),
                draw_attempts=int(inference["laplace"]["draw_attempts"]),
                mh_steps=int(inference["laplace"]["mh_steps"]),
            ),
            state_kwargs={
                "phi_step_intercept": float(phi_options["step_intercept"]),
                "phi_step_linear": float(phi_options["step_linear"]),
                "phi_laplace_max_iterations": int(
                    phi_options["laplace_max_iterations"]
                ),
                "phi_laplace_tolerance": float(
                    phi_options["laplace_tolerance"]
                ),
                "phi_curvature_floor": float(phi_options["curvature_floor"]),
                "phi_maximum_variance": float(
                    phi_options["maximum_variance"]
                ),
                "phi_shift_limit": float(phi_options["shift_limit"]),
                "phi_laplace_mh_steps": int(phi_options["mh_steps"]),
                "phi_draw_attempts": int(phi_options["draw_attempts"]),
            },
            name=f"simulation: truth={truth_phi}, fit={fit_phi}",
        )
    fit.metadata.update(
        example="simulation_phi",
        settings_file=str(source_settings_path),
        truth=truth_record,
        truth_phi=truth_phi,
        fitted_phi=fit_phi,
        prior_settings=prior_settings,
    )
    fit_path.parent.mkdir(parents=True, exist_ok=True)
    fit.save(fit_path)
    print(f"Saved {fit_path}")

    if bool(runtime["chain_only"]):
        return

    # Standard tables: these intentionally match examples 03/04/08.
    table_dir = output_dir / "tables" / case_key
    figure_dir = output_dir / "figures" / case_key
    table_dir.mkdir(parents=True, exist_ok=True)
    interval_probability = float(figures["interval_probability"])
    tail_probability = 0.5 * (1.0 - interval_probability)
    quantiles = [tail_probability, 0.5, 1.0 - tail_probability]
    diagnostics = fit.diagnostics()
    engine_diagnostics = diagnostics["engine"]

    pd.DataFrame.from_dict(
        fit.static_summary(credible_interval=interval_probability),
        orient="index",
    ).rename_axis("parameter").to_csv(table_dir / "parameters.csv")
    diagnostics["parameters"].to_csv(table_dir / "diagnostics.csv")
    pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in engine_diagnostics.items()
        ]
    ).to_csv(table_dir / "algorithm.csv", index=False)

    eta_lower, eta_median, eta_upper = np.quantile(
        fit.eta_draws(original_scale=True), quantiles, axis=0
    )
    pd.DataFrame(
        {
            "time": simulation_table["time"],
            "phase": simulation_table["phase"],
            "observed": fit.observed,
            "lower": eta_lower,
            "median": eta_median,
            "upper": eta_upper,
            "truth": simulation_table["eta"],
        }
    ).to_csv(table_dir / "trajectory.csv", index=False)

    labels = {0: "zero", 1: "fixed", 2: "dynamic"}
    selection = fit.component_probabilities().reset_index()
    selection["truth_code"] = selection["process"].map(structural_truth)
    selection["truth_state"] = selection["truth_code"].map(labels)
    selection["probability_true_state"] = [
        row[labels[int(row["truth_code"])]] for _, row in selection.iterrows()
    ]
    selection.to_csv(table_dir / "selection.csv", index=False)
    fit.structural_model_probabilities().to_csv(
        table_dir / "models.csv", index=False
    )
    fit.component_transition_summary().reset_index().to_csv(
        table_dir / "switching.csv", index=False
    )

    phi_draws = fit.phi_draws()
    sigma_draws = fit.sigma_draws()
    phi_lower, phi_median, phi_upper = np.quantile(
        phi_draws, quantiles, axis=0
    )
    sigma_lower, sigma_median, sigma_upper = np.quantile(
        sigma_draws, quantiles, axis=0
    )
    pd.DataFrame(
        {
            "time": simulation_table["time"],
            "truth_phi": phi_truth,
            "phi_lower": phi_lower,
            "phi_median": phi_median,
            "phi_upper": phi_upper,
            "truth_sigma": np.exp(phi_truth),
            "sigma_lower": sigma_lower,
            "sigma_median": sigma_median,
            "sigma_upper": sigma_upper,
        }
    ).to_csv(table_dir / "phi.csv", index=False)
    phi_probabilities = (
        fit.phi_model_probabilities()
        if fit_phi == "ssvs"
        else {fit_phi: 1.0}
    )
    pd.DataFrame(
        [
            {"model": name, "posterior_probability": probability}
            for name, probability in phi_probabilities.items()
        ]
    ).to_csv(table_dir / "phi_models.csv", index=False)

    predictive_draws = int(figures["predictive_draws"])
    seed = int(mcmc_settings["seed"])
    predictive = fit.posterior_predictive(
        draws=predictive_draws,
        seed=seed + 20_000,
    )
    predictive.summary(level=interval_probability).to_csv(
        table_dir / "posterior_predictive.csv", index=False
    )
    forecast = fit.forecast(
        int(figures["forecast_horizon"]),
        draws=predictive_draws,
        seed=seed + 30_000,
    )
    forecast.summary(level=interval_probability).to_csv(
        table_dir / "forecast.csv", index=False
    )
    focus_phase = int(figures["focus_phase"])
    if not 1 <= focus_phase <= period:
        raise ValueError(f"figures.focus_phase must be between 1 and {period}.")
    forecast.summary(level=interval_probability, phase=focus_phase).to_csv(
        table_dir / f"forecast_phase_{focus_phase:02d}.csv", index=False
    )
    forecast.summary(level=interval_probability, target="level").to_csv(
        table_dir / "forecast_level.csv", index=False
    )

    summary = {
        "fit": str(fit_path),
        "series": fit.series_name,
        "n_time": fit.n_time,
        "n_chains": fit.n_chains,
        "draws_per_chain": fit.draws_per_chain,
        "truth_phi": truth_phi,
        "fitted_phi": fit_phi,
        "phi_model_probabilities": phi_probabilities,
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
        "plan": fit.plan.to_dict(),
        "engine_diagnostics": engine_diagnostics,
        "prior_settings": prior_settings,
    }
    (table_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([{"scenario": case_key, **engine_diagnostics}]).to_csv(
        output_dir / "tables" / "laplace_mh_diagnostics.csv", index=False
    )

    if not bool(figures["enabled"]):
        print(f"Simulation tables: {table_dir}")
        return

    # Standard figures, with phi added to the established result set.
    figure_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(map(str, figures["formats"]))
    dpi = int(figures["dpi"])
    phase_labels = tuple(f"phase {index + 1}" for index in range(period))
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )

    figure, axis = fit.plot(
        "predictor", credible_interval=interval_probability
    )
    axis.plot(
        np.arange(n_time),
        simulation_table["eta"],
        color="#123B4A",
        linestyle="--",
        linewidth=1.2,
        label=r"$\mu_t$ truth",
    )
    axis.legend()
    for extension in formats:
        figure.savefig(
            figure_dir / f"trajectory.{extension}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(figure)

    selected_phase = simulation_table["phase"].to_numpy(int) == focus_phase
    figure, axis = fit.plot(
        "predictor",
        credible_interval=interval_probability,
        phase=focus_phase,
    )
    axis.plot(
        np.arange(n_time)[selected_phase],
        simulation_table.loc[selected_phase, "eta"],
        color="#123B4A",
        linestyle="--",
        linewidth=1.2,
        label=r"$\mu_t$ truth",
    )
    axis.legend()
    for extension in formats:
        figure.savefig(
            figure_dir / f"trajectory_phase_{focus_phase:02d}.{extension}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(figure)

    axis = predictive.plot(
        level=interval_probability,
        observed=fit.observed,
        title=bx.config_title(config, "posterior_predictive"),
        ylabel="simulated observation",
    )
    for extension in formats:
        axis.figure.savefig(
            figure_dir / f"posterior_predictive.{extension}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(axis.figure)

    axis = forecast.plot(
        level=interval_probability,
        history=fit.observed,
        history_dates=np.arange(n_time),
        history_points=int(figures["forecast_history"]),
        title=bx.config_title(config, "forecast"),
        ylabel="simulated observation",
    )
    for extension in formats:
        axis.figure.savefig(
            figure_dir / f"forecast.{extension}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(axis.figure)

    axis = forecast.plot(
        level=interval_probability,
        phase=focus_phase,
        phase_label=phase_labels[focus_phase - 1],
        history=fit.observed,
        history_dates=np.arange(n_time),
        history_points=int(figures["forecast_history"]),
        title=bx.config_title(config, "phase_forecast"),
        ylabel="simulated observation",
    )
    for extension in formats:
        axis.figure.savefig(
            figure_dir / f"forecast_phase_{focus_phase:02d}.{extension}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(axis.figure)

    level_history = np.median(fit.state_original("level"), axis=0)
    axis = forecast.plot(
        level=interval_probability,
        target="level",
        history=level_history,
        history_dates=np.arange(n_time),
        history_points=int(figures["forecast_history"]),
        title=bx.config_title(config, "level_forecast"),
        ylabel="latent level",
    )
    for extension in formats:
        axis.figure.savefig(
            figure_dir / f"forecast_level.{extension}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(axis.figure)

    for show_observed, filename in (
        (True, "level"),
        (False, "level_no_observations"),
    ):
        figure, _ = fit.plot(
            "level",
            credible_interval=interval_probability,
            truth=simulation_table["level"].to_numpy(),
            show_observed=show_observed,
        )
        for extension in formats:
            figure.savefig(
                figure_dir / f"{filename}.{extension}",
                dpi=dpi,
                bbox_inches="tight",
            )
        plt.close(figure)

    figure, _ = fit.plot(
        "slope",
        credible_interval=interval_probability,
        truth=simulation_table["slope"].to_numpy(),
    )
    for extension in formats:
        figure.savefig(
            figure_dir / f"slope.{extension}", dpi=dpi, bbox_inches="tight"
        )
    plt.close(figure)

    for kind, filename in (
        ("component_probabilities", "selection"),
        ("process_sds", "process_sd"),
    ):
        plot_kwargs = (
            {"truths": parameter_truth} if kind == "process_sds" else {}
        )
        figure, _ = fit.plot(kind, **plot_kwargs)
        for extension in formats:
            figure.savefig(
                figure_dir / f"{filename}.{extension}",
                dpi=dpi,
                bbox_inches="tight",
            )
        plt.close(figure)

    gev_parameters = {
        "stationary": ("sigma", "xi"),
        "linear": ("phi_intercept", "phi_slope", "xi"),
        "rw": ("phi_intercept", "phi_rw_sd", "xi"),
        "ssvs": ("phi_intercept", "phi_slope", "phi_rw_sd", "xi"),
    }[fit_phi]
    figure, _ = fit.plot(
        "parameter_densities",
        parameters=gev_parameters,
        truths=parameter_truth,
        credible_interval=interval_probability,
    )
    for extension in formats:
        figure.savefig(
            figure_dir / f"gev.{extension}", dpi=dpi, bbox_inches="tight"
        )
    plt.close(figure)

    figure, _ = fit.plot(
        "season", labels=phase_labels, show_interval=False
    )
    for extension in formats:
        figure.savefig(
            figure_dir / f"season.{extension}", dpi=dpi, bbox_inches="tight"
        )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4))
    axis.fill_between(
        np.arange(n_time), phi_lower, phi_upper, color="C0", alpha=0.2
    )
    axis.plot(phi_truth, color="black", linestyle="--", label="truth")
    axis.plot(phi_median, color="C0", label="posterior median")
    axis.set(xlabel="time", ylabel=r"$\phi_t=\log(\sigma_t)$")
    title = bx.config_title(config, "phi")
    if title:
        axis.set_title(title)
    axis.legend()
    for extension in formats:
        figure.savefig(
            figure_dir / f"phi.{extension}", dpi=dpi, bbox_inches="tight"
        )
    plt.close(figure)

    if bool(figures["diagnostics"]):
        for kind, filename in (("traces", "sd_traces"), ("acf", "acf")):
            figure, _ = fit.plot(kind)
            for extension in formats:
                figure.savefig(
                    figure_dir / f"{filename}.{extension}",
                    dpi=dpi,
                    bbox_inches="tight",
                )
            plt.close(figure)

    print(f"Simulation tables and figures: {output_dir}")


if __name__ == "__main__":
    main()
