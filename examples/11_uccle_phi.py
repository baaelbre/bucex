"""Fit Uccle extremes with stationary, linear, RW, or SSVS log scale.

Pick the JSON below for an IDE run, or pass ``--config PATH``. Every model,
prior, MCMC, numerical, report, and output choice is read from that file. The
result directory matches the established Uccle report layout and adds phi
tables and figures.
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
    Path(__file__).parent
    / "config"
    / "phi"
    / "uccle"
    / "01_txx_stationary.json"
)
SCRIPT_NAME = Path(__file__).stem
MONTH_LABELS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


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
    location_settings = model_settings["location"]
    prior_settings = config["priors"]
    mcmc_settings = config["mcmc"]
    inference = config["inference"]
    figures = config["figures"]
    runtime = config["runtime"]
    output = config["output"]

    if (
        str(model_settings["observation_family"]).lower() != "gev"
        or str(location_settings["structure"]).lower() != "ssvs"
        or str(location_settings["trend_component"]).lower()
        != "local_linear_trend"
        or str(location_settings["seasonal_component"]).lower() != "dummy"
    ):
        raise ValueError(
            "This example supports a GEV model with structural location SSVS, "
            "a local-linear trend, and dummy seasonality."
        )

    data_dir = (
        None
        if data_settings["data_dir"] is None
        else Path(data_settings["data_dir"])
    )
    start = str(data_settings["start"])
    end = data_settings["end"] or None
    series_names = tuple(map(str, data_settings["series"]))
    period = int(data_settings["period"])
    fit_phi = str(model_settings["phi"])
    components = (
        bx.LocalLinearTrend(
            level_mode=str(location_settings["level_mode"]),
            trend_mode=str(location_settings["trend_mode"]),
        ),
        bx.DummySeasonal(
            period=period,
            mode=str(location_settings["seasonal_mode"]),
        ),
    )
    model = bx.Model(
        bx.GEV(xi_bounds=tuple(prior_settings["xi_bounds"]), phi=fit_phi),
        components,
        name=f"Uccle structural location SSVS with {fit_phi} log scale",
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

    draws = int(mcmc_settings["draws"])
    warmup = int(mcmc_settings["warmup"])
    chains = int(mcmc_settings["chains"])
    run_id = output.get("run_id") or datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    signature = (
        f"phi-{fit_phi}_{'-'.join(name.lower() for name in series_names)}"
        f"_y{start.removesuffix('-01-01')}-"
        f"{(end or 'latest').removesuffix('-12-31')}"
        f"_d{draws}w{warmup}c{chains}"
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

    interval_probability = float(figures["interval_probability"])
    seasonal_pattern_settings = figures.get("seasonal_patterns", {})
    seasonal_pattern_years = tuple(
        int(value) for value in seasonal_pattern_settings.get("years", ())
    )
    seasonal_pattern_show_interval = bool(
        seasonal_pattern_settings.get("show_interval", True)
    )
    if seasonal_pattern_settings.get("cycles"):
        raise ValueError(
            "Dated Uccle seasonal patterns use figures.seasonal_patterns.years."
        )
    tail_probability = 0.5 * (1.0 - interval_probability)
    quantiles = [tail_probability, 0.5, 1.0 - tail_probability]
    predictive_draws = int(figures["predictive_draws"])
    forecast_horizon = int(figures["forecast_horizon"])
    forecast_history = int(figures["forecast_history"])
    focus_month = int(figures["focus_month"])
    if not 1 <= focus_month <= 12:
        raise ValueError("figures.focus_month must be between 1 and 12.")
    focus_label = MONTH_LABELS[focus_month - 1]
    seed = int(mcmc_settings["seed"])
    selection_rows = []
    algorithm_rows = []
    phi_rows = []

    if bool(figures["enabled"]):
        plt.rcParams.update(
            {
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.titleweight": "bold",
                "legend.frameon": False,
            }
        )

    for series_number, series_name in enumerate(series_names):
        values = bx.load_uccle_series(
            series_name,
            data_dir,
            start=start,
            end=end,
        )
        focus_phase = (
            focus_month - int(values.index[0].month)
        ) % period + 1
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

        fit_path = output_dir / "fits" / series_name / "combined.bucex"
        if runtime.get("combine_runs"):
            source_paths = [
                Path(directory)
                / "fits"
                / series_name
                / "combined.bucex"
                for directory in runtime["combine_runs"]
            ]
            missing = [path for path in source_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Cannot combine chains; missing fit file(s):\n  "
                    + "\n  ".join(str(path) for path in missing)
                )
            fit = bx.combine_fits(
                [bx.FitResult.load(path) for path in source_paths]
            )
            fit.metadata["combined_chain_sources"] = [
                str(path) for path in source_paths
            ]
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
                    draws=draws,
                    warmup=warmup,
                    thin=int(mcmc_settings["thin"]),
                    chains=chains,
                    seed=seed + 100 * series_number,
                    progress=bool(runtime["progress"]),
                ),
                laplace=bx.Laplace(
                    max_iterations=int(
                        inference["laplace"]["max_iterations"]
                    ),
                    tolerance=float(inference["laplace"]["tolerance"]),
                    curvature_floor=float(
                        inference["laplace"]["curvature_floor"]
                    ),
                    maximum_variance=float(
                        inference["laplace"]["maximum_variance"]
                    ),
                    draw_attempts=int(
                        inference["laplace"]["draw_attempts"]
                    ),
                    mh_steps=int(inference["laplace"]["mh_steps"]),
                ),
                state_kwargs={
                    "phi_step_intercept": float(
                        phi_options["step_intercept"]
                    ),
                    "phi_step_linear": float(phi_options["step_linear"]),
                    "phi_laplace_max_iterations": int(
                        phi_options["laplace_max_iterations"]
                    ),
                    "phi_laplace_tolerance": float(
                        phi_options["laplace_tolerance"]
                    ),
                    "phi_curvature_floor": float(
                        phi_options["curvature_floor"]
                    ),
                    "phi_maximum_variance": float(
                        phi_options["maximum_variance"]
                    ),
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
            start=start,
            end=end,
            prior_settings=prior_settings,
        )
        fit_path.parent.mkdir(parents=True, exist_ok=True)
        fit.save(fit_path)
        print(f"Saved {fit_path}")

        if bool(runtime["chain_only"]):
            continue

        # Standard Uccle tables, with phi and scale-model tables added.
        table_dir = output_dir / "tables" / series_name
        figure_dir = output_dir / "figures" / series_name
        table_dir.mkdir(parents=True, exist_ok=True)
        diagnostics = fit.diagnostics()
        engine_diagnostics = diagnostics["engine"]
        algorithm_rows.append({"series": series_name, **engine_diagnostics})

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
                "date": values.index,
                "month": values.index.month,
                "observed": values.to_numpy(),
                "lower": eta_lower,
                "median": eta_median,
                "upper": eta_upper,
            }
        ).to_csv(table_dir / "trajectory.csv", index=False)

        selection = fit.component_probabilities().reset_index()
        selection.insert(0, "series", series_name)
        selection.insert(1, "engine", str(inference["engine"]))
        selection.to_csv(table_dir / "selection.csv", index=False)
        selection_rows.append(selection)
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
                "date": values.index,
                "phi_lower": phi_lower,
                "phi_median": phi_median,
                "phi_upper": phi_upper,
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
        phi_rows.append(
            {
                "series": series_name,
                "fitted_phi": fit_phi,
                "probability_stationary": phi_probabilities.get(
                    "stationary", 0.0
                ),
                "probability_linear": phi_probabilities.get("linear", 0.0),
                "probability_rw": phi_probabilities.get("rw", 0.0),
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

        predictive = fit.posterior_predictive(
            draws=predictive_draws,
            seed=seed + 20_000 + series_number,
        )
        predictive.summary(level=interval_probability).to_csv(
            table_dir / "posterior_predictive.csv", index=False
        )
        forecast = fit.forecast(
            forecast_horizon,
            draws=predictive_draws,
            seed=seed + 30_000 + series_number,
        )
        forecast.summary(level=interval_probability).to_csv(
            table_dir / "forecast.csv", index=False
        )
        focus_forecast = forecast.summary(
            level=interval_probability, phase=focus_phase
        )
        focus_forecast.insert(1, "calendar_month", focus_month)
        focus_forecast.insert(2, "calendar_month_label", focus_label)
        focus_forecast.to_csv(
            table_dir / f"forecast_{focus_label.lower()}.csv", index=False
        )
        forecast.summary(
            level=interval_probability, target="level"
        ).to_csv(table_dir / "forecast_level.csv", index=False)

        summary = {
            "fit": str(fit_path),
            "series": series_name,
            "description": bx.UCCLE_INFO[series_name]["description"],
            "tail": tail,
            "start": str(values.index.min().date()),
            "end": str(values.index.max().date()),
            "n_time": fit.n_time,
            "n_chains": fit.n_chains,
            "draws_per_chain": fit.draws_per_chain,
            "fitted_phi": fit_phi,
            "phi_model_probabilities": phi_probabilities,
            "plan": fit.plan.to_dict(),
            "engine_diagnostics": engine_diagnostics,
            "prior_settings": prior_settings,
        }
        (table_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if not bool(figures["enabled"]):
            continue

        # Standard Uccle figures, with a log-scale trajectory added.
        figure_dir.mkdir(parents=True, exist_ok=True)
        formats = tuple(map(str, figures["formats"]))
        dpi = int(figures["dpi"])

        figure, axis = fit.plot(
            "predictor", credible_interval=interval_probability
        )
        axis.set_ylabel("GEV location / °C")
        for extension in formats:
            figure.savefig(
                figure_dir / f"trajectory.{extension}",
                dpi=dpi,
                bbox_inches="tight",
            )
        plt.close(figure)

        figure, axis = fit.plot(
            "predictor",
            credible_interval=interval_probability,
            phase=focus_phase,
        )
        axis.set_ylabel("GEV location / °C")
        for extension in formats:
            figure.savefig(
                figure_dir
                / f"trajectory_{focus_label.lower()}.{extension}",
                dpi=dpi,
                bbox_inches="tight",
            )
        plt.close(figure)

        axis = predictive.plot(
            level=interval_probability,
            observed=fit.observed,
            title=bx.config_title(config, "posterior_predictive"),
            ylabel="temperature / °C",
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
            history_dates=values.index,
            history_points=forecast_history,
            title=bx.config_title(config, "forecast"),
            ylabel="temperature / °C",
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
            phase_label=focus_label,
            history=fit.observed,
            history_dates=values.index,
            history_points=forecast_history,
            title=bx.config_title(config, "phase_forecast"),
            ylabel="temperature / °C",
        )
        for extension in formats:
            axis.figure.savefig(
                figure_dir / f"forecast_{focus_label.lower()}.{extension}",
                dpi=dpi,
                bbox_inches="tight",
            )
        plt.close(axis.figure)

        level_history = np.median(fit.state_original("level"), axis=0)
        axis = forecast.plot(
            level=interval_probability,
            target="level",
            history=level_history,
            history_dates=values.index,
            history_points=forecast_history,
            title=bx.config_title(config, "level_forecast"),
            ylabel="latent GEV level / °C",
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
            figure, axis = fit.plot(
                "level",
                credible_interval=interval_probability,
                show_observed=show_observed,
            )
            axis.set_ylabel("GEV level / °C")
            for extension in formats:
                figure.savefig(
                    figure_dir / f"{filename}.{extension}",
                    dpi=dpi,
                    bbox_inches="tight",
                )
            plt.close(figure)

        trend_states = np.asarray(fit.parameter("state_trend"), dtype=int)
        slope_condition = "dynamic" if np.any(trend_states == 2) else None
        figure, _ = fit.plot(
            "slope",
            credible_interval=interval_probability,
            scale="decade",
            unit="slope / °C per decade",
            condition_on=slope_condition,
            show_fixed=bool(np.any(trend_states == 1)),
        )
        for extension in formats:
            figure.savefig(
                figure_dir / f"slope.{extension}",
                dpi=dpi,
                bbox_inches="tight",
            )
        plt.close(figure)

        for kind, filename in (
            ("component_probabilities", "selection"),
            ("process_sds", "process_sd"),
        ):
            figure, _ = fit.plot(kind)
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
            credible_interval=interval_probability,
        )
        for extension in formats:
            figure.savefig(
                figure_dir / f"gev.{extension}",
                dpi=dpi,
                bbox_inches="tight",
            )
        plt.close(figure)

        figure, _ = fit.plot("season", show_interval=False)
        for extension in formats:
            figure.savefig(
                figure_dir / f"season.{extension}",
                dpi=dpi,
                bbox_inches="tight",
            )
        plt.close(figure)

        if seasonal_pattern_years:
            figure, _ = fit.plot(
                "seasonal_patterns",
                years=seasonal_pattern_years,
                credible_interval=interval_probability,
                show_interval=seasonal_pattern_show_interval,
                title=bx.config_title(config, "seasonal_patterns"),
            )
            for extension in formats:
                figure.savefig(
                    figure_dir / f"seasonal_patterns.{extension}",
                    dpi=dpi,
                    bbox_inches="tight",
                )
            plt.close(figure)

        try:
            figure, _ = fit.plot(
                "endpoint", credible_interval=interval_probability
            )
        except ValueError:
            print(
                f"{series_name}: no finite endpoint in retained posterior draws."
            )
        else:
            for extension in formats:
                figure.savefig(
                    figure_dir / f"endpoint.{extension}",
                    dpi=dpi,
                    bbox_inches="tight",
                )
            plt.close(figure)

        figure, axis = plt.subplots(figsize=(10, 4))
        axis.fill_between(
            values.index, phi_lower, phi_upper, color="C0", alpha=0.2
        )
        axis.plot(values.index, phi_median, color="C0")
        axis.set(ylabel=r"$\phi_t=\log(\sigma_t)$", xlabel="date")
        title = bx.config_title(config, "phi") or bx.config_title(
            config, series_name
        )
        if title:
            axis.set_title(title)
        for extension in formats:
            figure.savefig(
                figure_dir / f"phi.{extension}",
                dpi=dpi,
                bbox_inches="tight",
            )
        plt.close(figure)

        if bool(figures["diagnostics"]):
            for kind, filename in (
                ("traces", "sd_traces"),
                ("acf", "acf"),
            ):
                figure, _ = fit.plot(kind)
                for extension in formats:
                    figure.savefig(
                        figure_dir / f"{filename}.{extension}",
                        dpi=dpi,
                        bbox_inches="tight",
                    )
                plt.close(figure)

    if bool(runtime["chain_only"]):
        print(f"One-chain Uccle outputs: {output_dir}")
        return

    aggregate_dir = output_dir / "tables"
    pd.concat(selection_rows, ignore_index=True).to_csv(
        aggregate_dir / "selection_all.csv", index=False
    )
    pd.DataFrame(algorithm_rows).to_csv(
        aggregate_dir / "laplace_mh_diagnostics.csv", index=False
    )
    pd.DataFrame(phi_rows).to_csv(
        aggregate_dir / "phi_summary_all.csv", index=False
    )
    (aggregate_dir / "phi_summary_all.json").write_text(
        json.dumps(phi_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Uccle tables and figures: {output_dir}")


if __name__ == "__main__":
    main()
