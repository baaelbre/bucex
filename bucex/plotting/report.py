"""Compact report exports assembled from the public fit/forecast/plot API."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from .predictive import (plot_predictive_diagnostics, plot_chain_traces,
                        plot_forecast_months, plot_scale_calendar, plot_calendar_risk_curves)


def save_prediction_report(fit, directory, *, channel=None, forecast=None, predictive=None,
                           threshold=None, level=.95, draws=400, seed=None,
                           months=tuple(range(1, 13)), image_format="png", dpi=150,
                           figures=True, prefix=None):
    """Save slope, scale, trace, PIT/Q-Q, calendar forecast and risk summaries.

    PNG is the default. Tables accompany figures. Annual/seasonal forecasts
    retain whole posterior paths and omit incomplete windows. Risk curves
    integrate monthly observation noise, conditional on future state paths.
    This function never refits, changes the data window, or certifies convergence.
    """
    import matplotlib.pyplot as plt
    if image_format not in {"png", "pdf", "svg"}:
        raise ValueError("image_format must be png, pdf, or svg.")
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    label = prefix or channel or fit.series_name or "series"
    forecast = forecast if forecast is not None else fit.forecast(120, draws=draws, seed=seed)
    predictive = predictive if predictive is not None else fit.posterior_predictive(draws=draws, seed=seed)
    observed = fit.observed[:, fit.channel_names.index(channel)] if fit.is_multiseries_model else fit.observed
    family = fit.model.channel(channel).family if fit.is_multiseries_model else fit.family
    written = []

    def table(value, name):
        file = path/f"{label}_{name}.csv"
        value.to_csv(file, index=False)
        written.append(file.name)

    def save(figure, name):
        file = path/f"{label}_{name}.{image_format}"
        figure.savefig(file, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        written.append(file.name)

    def band(values, dates, name, ylabel, *, probability=False):
        low, median, high = np.quantile(values, [(1-level)/2, .5, (1+level)/2], axis=0)
        mean = values.mean(axis=0)
        table(pd.DataFrame(dict(time=dates, lower=low, median=median, upper=high, mean=mean)), name)
        if figures:
            figure, ax = plt.subplots(figsize=(9, 3.5))
            ax.fill_between(dates, low, high, alpha=.2)
            ax.plot(dates, mean if probability else median)
            if probability:
                ax.set_ylim(-.01, 1.01)
            ax.set(xlabel="time", ylabel=ylabel)
            figure.tight_layout()
            save(figure, name)

    slope = fit.component_draws("slope", channel=channel, combine_chains=False)*120
    band(slope.reshape((-1, fit.n_time)), fit.time, "slope", "latent slope / °C per decade")
    targets = {"end slope / °C per decade": slope[..., -1]}
    level_draws = fit.component_draws("level", channel=channel, combine_chains=False)
    targets["level change / °C"] = level_draws[..., -1]-level_draws[..., 0]
    table(fit.contrast_diagnostics(targets, credible_interval=level).reset_index(), "target_mcmc")
    if figures:
        save(plot_chain_traces(targets)[0], "target_traces")
        suffix = f".{channel}" if fit.is_multiseries_model else ""
        names = [name for name, values in fit.parameter_draws.items()
                 if np.asarray(values).ndim == 2 and
                 (not suffix or name.endswith(suffix) or name.startswith(f"sd.channel.{channel}.")) and
                 (name.startswith("sd.") or name in ["sigma"+suffix, "xi"+suffix,
                     "scale_slope"+suffix, "scale_rw_sd"+suffix])]
        if names:
            save(plot_chain_traces({name:fit.parameter(name, combine_chains=False) for name in names})[0], "parameter_traces")
        save(plot_predictive_diagnostics(predictive, observed, channel=channel, in_sample=True)[0], "pit_qq_residuals")
    table(pd.DataFrame(dict(time=fit.time, pit=predictive.pit(observed, channel=channel))), "smoothed_pit")
    figure, _, scales = plot_scale_calendar(fit, channel=channel, level=level)
    table(scales, "scale_by_month")
    if figures:
        save(figure, "scale_by_month")
    else:
        plt.close(figure)
    dates = pd.DatetimeIndex(fit.time)
    sigma_chains = fit.sigma_draws(channel=channel, combine_chains=False)
    scale_targets = {f"month {month:02d} scale / °C": sigma_chains[..., np.flatnonzero(dates.month == month)[-1]]
                     for month in range(1, 13) if np.any(dates.month == month)}
    table(fit.contrast_diagnostics(scale_targets, credible_interval=level).reset_index(), "scale_mcmc")
    if figures:
        selected = {key:value for key,value in scale_targets.items() if int(key[6:8]) in (1,4,7,10)}
        if selected:
            save(plot_chain_traces(selected)[0], "scale_traces")
    band(forecast.sigma_draws(channel=channel), forecast.dates, "forecast_observation_scale",
         ("residual SD" if family == "gaussian" else "GEV scale")+" / °C")
    band(forecast.component_draws("slope", channel=channel)*120, forecast.dates,
         "forecast_slope", "latent slope / °C per decade")

    # Every month remains individually available in the data table and panels.
    table(forecast.summary(level=level, channel=channel), "forecast_monthly")
    if figures:
        save(plot_forecast_months(forecast, channel=channel, months=months, level=level,
            history=observed, history_dates=fit.time)[0], "forecast_by_month")
    if threshold is not None:
        risk = forecast.probability_draws(threshold, channel=channel)
        band(risk, forecast.dates, "forecast_monthly_risk", fit.event_label(threshold,channel=channel), probability=True)
        if figures:
            save(plot_forecast_months(forecast, channel=channel, months=months, level=level,
                                     threshold=threshold)[0], "forecast_risk_by_month")
            # In-sample monthly curves, explicitly based on the smoothed posterior.
            save(plot_forecast_months(predictive, channel=channel, months=months, level=level,
                                     threshold=threshold)[0], "smoothed_risk_by_month")

    for frequency in ("year", "season"):
        aggregate = forecast.aggregate(frequency=frequency, channel=channel)
        table(aggregate.summary(level), f"forecast_{frequency}")
        all_periods = forecast.aggregate(frequency=frequency, channel=channel, include_partial=True).periods
        table(all_periods[~all_periods.complete], f"omitted_partial_{frequency}")
        if not aggregate.n_periods:
            continue
        if aggregate.reduction == "mean":
            mean, variance = aggregate.conditional_moments()
            table(aggregate.periods.assign(latent_mean_variance=np.var(mean, axis=0),
                expected_observation_variance=variance.mean(axis=0),
                total_predictive_variance=np.var(mean, axis=0)+variance.mean(axis=0)),
                f"forecast_{frequency}_variance")
        if figures:
            windows = list(dict.fromkeys(aggregate.periods.window))
            figure, axes = plt.subplots(len(windows), 1, squeeze=False, figsize=(9, 3*len(windows)))
            for window, ax in zip(windows, axes[:, 0]):
                aggregate.plot(level=level, window=window, ax=ax)
                ax.set_title(window, fontsize=11)
            figure.tight_layout()
            save(figure, f"forecast_{frequency}")
        # Curves are tabulated even when plotting is disabled. A small table of
        # grid thresholds avoids choosing a new scientific event on the user's behalf.
        figure, _, curves = plot_calendar_risk_curves(aggregate, level=level, points=40)
        table(curves, f"forecast_{frequency}_risk_curves")
        if figures:
            save(figure, f"forecast_{frequency}_risk_curves")
        else:
            plt.close(figure)

    notes = dict(channel=channel or fit.series_name, fitted_start=str(fit.time[0]),
        fitted_end=str(fit.time[-1]), forecast_start=str(forecast.dates[0]), forecast_end=str(forecast.dates[-1]),
        posterior_chains=fit.n_chains, posterior_draws_per_chain=fit.draws_per_chain,
        figures=written, slope_unit="degrees C per decade (120 monthly slope units)",
        aggregation="Gaussian monthly means: day-weighted; upper/lower GEV blocks: max/min. Complete periods only; DJF labelled by ending year.",
        intervals=f"{level:.0%} pointwise; aggregate observation bands are predictive intervals.",
        risk_bands="Variation across posterior parameters and future state paths, with observation noise integrated; lines use the mean probability.",
        residual_assumption="Conditional independence across months. Cross-series copula dependence is contemporaneous, not serial.",
        calibration="Smoothed PIT/Q-Q are in-sample descriptive checks. Use held-out predictions for forecast calibration.")
    (path/f"{label}_prediction_notes.json").write_text(json.dumps(notes, indent=2)+"\n")
    return notes
