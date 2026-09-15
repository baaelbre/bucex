"""Small report writer; all scientific calculations use public bucex methods."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import bucex as bx


def new_run(root, name):
    """Create a fresh directory without overwriting an earlier analysis."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = Path(root) / f"{name}_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_band(fit, values, path, *, dates=None, ylabel="temperature / °C", level=0.90):
    """Export one scientific band with readable labels and no default title."""
    with plt.rc_context({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11}):
        return _save_band(fit, values, path, dates=dates, ylabel=ylabel, level=level)


def _save_band(fit, values, path, *, dates=None, ylabel="temperature / °C", level=0.90):
    """Export pointwise posterior summaries and a figure on the same scale."""
    dates = fit.time if dates is None else dates
    band = fit.posterior_summary(values, credible_interval=level)
    pd.DataFrame({"time": dates, **band}).to_csv(path.with_suffix(".csv"), index=False)
    figure, axis = plt.subplots(figsize=(8, 3))
    axis.fill_between(dates, band["lower"], band["upper"], alpha=0.2)
    axis.plot(dates, band["median"])
    axis.set(xlabel="time", ylabel=ylabel)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def scientific_targets(fit):
    """First-to-last observed changes, retaining chain and draw dimensions."""
    targets = {}
    for name in fit.channel_names if fit.is_multiseries_model else (None,):
        values = fit.component_draws("level", channel=name, combine_chains=False)
        targets[f"{name or 'series'}_level_change"] = values[..., -1] - values[..., 0]
    if fit.is_multiseries_model:
        for left, right in (("TXx", "TXm"), ("TNn", "TNm"), ("TXm", "TNm"), ("TXx", "TNx")):
            if {left, right} <= set(fit.channel_names):
                targets[f"{left}_minus_{right}_change"] = targets[f"{left}_level_change"] - targets[f"{right}_level_change"]
    for group in getattr(fit.model, "shared", ()):
        if not isinstance(group.component, (bx.LocalLevel, bx.LocalLinearTrend)):
            continue
        if isinstance(group, bx.Shared):
            values = fit.shared_draws(group.name, combine_chains=False)
            targets[f"{group.name}_change"] = values[..., -1] - values[..., 0]
        else:
            for name in fit.channel_names:
                values = fit.departure_draws(name, name=group.name, combine_chains=False)
                targets[f"{name}_{group.name}_change"] = values[..., -1] - values[..., 0]
    return targets


def write_report(fit, directory, *, config, risks=None, horizon=12, level=0.90):
    """Use readable paper-sized text without changing global matplotlib defaults."""
    sizes = {"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11}
    if isinstance(config.get("figures"), dict):
        sizes.update(config["figures"])
    with plt.rc_context(sizes):
        return _write_report(fit, directory, config=config, risks=risks, horizon=horizon, level=level)


def _write_report(fit, directory, *, config, risks=None, horizon=12, level=0.90):
    """Save a fit, parameters, diagnostics, fitted paths, risks and forecasts.

    In-sample PIT is labelled as a descriptive posterior check. This function
    does not judge convergence or turn pointwise bands into simultaneous bands.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fit.save(directory / "fit.bucex")
    bx.save_config(config, directory / "config.json")
    pd.DataFrame.from_dict(fit.static_summary(level), orient="index").to_csv(
        directory / "parameters.csv"
    )
    diagnostics = fit.diagnostics()
    diagnostics["parameters"].to_csv(directory / "mcmc.csv")
    targets = scientific_targets(fit)
    if targets:
        fit.contrast_diagnostics(targets, credible_interval=level).to_csv(
            directory / "scientific_targets.csv")
    engine = {key: float(value) if np.isfinite(value) else None
              for key, value in diagnostics["engine"].items()}
    (directory / "engine.json").write_text(json.dumps(engine, indent=2) + "\n")
    pd.DataFrame(diagnostics["pit"]).to_csv(directory / "in_sample_pit.csv", index=False)
    (directory / "run.json").write_text(json.dumps({
        "bucex_version": bx.__version__, "engine": fit.plan.engine,
        "model": fit.model.to_dict(), "warnings": diagnostics["warnings"],
        "interval": "pointwise posterior credible interval",
        "validation": "Exploratory fit; inspect mixing and held-out predictions separately.",
    }, indent=2) + "\n")
    forecast = fit.forecast(horizon, draws=config.get("forecast_draws"), seed=config.get("seed", 42))
    forecast.summary(level=level).to_csv(directory / "forecast.csv", index=False)
    for group in getattr(fit.model, "shared", ()):
        if isinstance(group, bx.Shared):
            save_band(fit, fit.shared_draws(group.name), directory / f"shared_{group.name}", level=level)
        else:
            for name in fit.channel_names:
                save_band(fit, fit.departure_draws(name, name=group.name),
                          directory / f"{name}_{group.name}", level=level)
    if getattr(fit.model, "copula", None) is not None:
        fit.copula_summary().reset_index().to_csv(directory / "copula_correlations.csv", index=False)
        figure, _ = fit.plot("copula")
        figure.savefig(directory / "copula_correlations.pdf", bbox_inches="tight")
        plt.close(figure)
    if fit.is_multiseries_model:
        constraints = [pair for pair in bx.UCCLE_ORDER_CONSTRAINTS if set(pair) <= set(fit.channel_names)]
        if constraints:
            check = fit.ordering_diagnostics(constraints, draws=config.get("predictive_check_draws", 400), seed=config.get("seed", 42))
            check.summary.to_csv(directory / "ordering_in_sample.csv", index=False)
            check.by_time.to_csv(directory / "ordering_in_sample_by_time.csv", index=False)
            check = forecast.ordering_diagnostics(constraints)
            check.summary.to_csv(directory / "ordering_forecast.csv", index=False)
            check.by_time.to_csv(directory / "ordering_forecast_by_time.csv", index=False)
    if fit.is_multiseries_model:
        bx.residual_dependence_check(fit, draws=config.get("predictive_check_draws", 200),
                                    seed=config.get("seed", 42)).to_csv(directory / "residual_dependence.csv", index=False)
        if {"TXx", "TNx"} <= set(fit.channel_names):
            probability = forecast.compound_probability({"TXx": (">", 35), "TNx": (">", 25)})
            pd.DataFrame({"time": forecast.dates, "probability": probability}).to_csv(directory / "compound_heat_forecast.csv", index=False)
    channels = fit.channel_names if fit.is_multiseries_model else (None,)
    for channel in channels:
        save_band(fit, fit.sigma_draws(channel=channel), directory / f"{channel or 'series'}_observation_scale",
                  ylabel="observation scale / °C", level=level)
        name = channel or "series"
        values = (fit.channel_eta_draws(channel, original_scale=True)
                  if channel else fit.eta_draws(original_scale=True))
        save_band(fit, values, directory / f"{name}_location", level=level)
        kind = "channel" if channel else "predictor"
        kwargs = {"channel": channel} if channel else {}
        figure, _ = fit.plot(kind, credible_interval=level, **kwargs)
        figure.savefig(directory / f"{name}_fit.pdf", bbox_inches="tight")
        plt.close(figure)
        if not isinstance(fit.priors, bx.JointPriors):
            fit.component_probabilities(channel=channel).to_csv(
                directory / f"{name}_structure.csv")
            fit.component_transition_summary(channel=channel).to_csv(directory / f"{name}_structure_mixing.csv")
            fit.structural_model_probabilities(channel=channel).to_csv(directory / f"{name}_models.csv", index=False)
        for component in ("level", "season"):
            if component == "season" and fit.model.period is None:
                continue
            figure, _ = fit.plot(component, credible_interval=level, **kwargs)
            figure.savefig(directory / f"{name}_{component}.pdf", bbox_inches="tight")
            plt.close(figure)
        axis = forecast.plot(channel=channel) if channel else forecast.plot()
        axis.figure.savefig(directory / f"{name}_forecast.pdf", bbox_inches="tight")
        plt.close(axis.figure)
        family = (next(item.family for item in fit.model.channels if item.name == channel)
                  if channel else fit.family)
        if family == "gev" and channel is None:
            save_band(fit, fit.sigma_draws(), directory / "scale", ylabel="scale / °C", level=level)
            scale_mode = fit.model.observation.phi
            scale_models = {"declared_mode": scale_mode}
            if scale_mode == "ssvs":
                scale_models["posterior_probabilities"] = fit.phi_model_probabilities()
            (directory / "scale_models.json").write_text(json.dumps(scale_models, indent=2) + "\n")
        if family == "gev":
            save_band(fit, fit.return_level_draws(100, channel=channel),
                      directory / f"{name}_return_level_100_blocks", level=level)
        if risks and name in risks:
            probability = fit.exceedance_probability_draws(
                risks[name], channel=channel, return_labels=False
            )
            save_band(fit, probability, directory / f"{name}_risk", level=level,
                      ylabel=fit.event_label(risks[name], channel=channel))
    return directory
