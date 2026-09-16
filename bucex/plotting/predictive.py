"""Reusable calendar, scale, chain, and predictive diagnostic figures."""
from __future__ import annotations

import calendar
import numpy as np
import pandas as pd
from scipy.special import ndtri


def plot_predictive_diagnostics(prediction, observed, *, channel=None, in_sample=False,
                                max_lag=24, bins=10):
    """PIT, normal-score Q-Q, residual time plot and ACF.

    In-sample PITs use smoothed states and are descriptive checks, not forecast
    calibration tests. No IID reference envelope or uniformity p-value is
    implied. The ACF is for normal scores of the mixture PIT, not a sampler ACF.
    """
    import matplotlib.pyplot as plt
    pit = prediction.pit(observed, channel=channel)
    if np.any(~np.isfinite(pit)):
        raise ValueError("Predictive diagnostic PITs must be finite.")
    z = ndtri(np.clip(pit, 1e-10, 1-1e-10))
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    axis = axes[0, 0]
    axis.hist(pit, bins=np.linspace(0, 1, bins+1), color="C0", alpha=.7)
    axis.axhline(len(pit)/bins, color="black", ls="--", lw=1)
    axis.set(xlabel="PIT", ylabel="count", xlim=(0, 1))
    theoretical = ndtri((np.arange(len(z))+.5)/len(z))
    axes[0, 1].plot(theoretical, np.sort(z), ".", ms=3)
    limits = [min(theoretical.min(), z.min()), max(theoretical.max(), z.max())]
    axes[0, 1].plot(limits, limits, color="black", lw=1)
    axes[0, 1].set(xlabel="standard normal quantile", ylabel="normal-score residual quantile")
    axes[1, 0].plot(prediction.dates, z, lw=.6)
    axes[1, 0].axhline(0, color="black", lw=.6)
    axes[1, 0].set(xlabel="time", ylabel="normal-score residual")
    centered = z-z.mean()
    denominator = centered @ centered
    lags = np.arange(1, min(int(max_lag), len(z)-1)+1)
    acf = [centered[:-lag] @ centered[lag:]/denominator if denominator > 0 else np.nan for lag in lags]
    axes[1, 1].bar(lags, acf, color="C0", alpha=.7)
    axes[1, 1].axhline(0, color="black", lw=.6)
    axes[1, 1].set(xlabel="lag / monthly blocks", ylabel="residual autocorrelation")
    figure.suptitle("In-sample smoothed predictive checks" if in_sample else "Held-out predictive checks", fontsize=12)
    figure.tight_layout()
    return figure, axes


def plot_chain_traces(draws, *, max_lag=50):
    """Traces and ACFs for labelled arrays of shape (chain, retained draw).

    Accepts scientific targets and physical innovation magnitudes. Signed FS
    coefficient sign flips alone are not evidence of effective magnitude mixing.
    """
    import matplotlib.pyplot as plt
    if not draws:
        raise ValueError("Provide at least one chain array.")
    figure, axes = plt.subplots(len(draws), 2, squeeze=False, figsize=(10, 2.05*len(draws)),
                                gridspec_kw={"width_ratios": [3, 1]})
    for (label, values), row in zip(draws.items(), axes):
        values = np.asarray(values)
        if values.ndim != 2:
            raise ValueError("Each trace must have shape (chain, retained draw).")
        for chain, x in enumerate(values):
            row[0].plot(x, lw=.7, label=f"chain {chain+1}")
            centered = x-x.mean()
            variance = centered @ centered
            lags = np.arange(min(max_lag, len(x)-1)+1)
            acf = [1 if lag == 0 else centered[:-lag] @ centered[lag:]/variance if variance > 0
                   else np.nan for lag in lags]
            row[1].plot(lags, acf, lw=.9)
        row[0].set_ylabel(label)
        row[1].axhline(0, color="black", lw=.5)
        row[1].set(ylim=(-1, 1.05), ylabel="ACF")
    axes[0, 0].legend(fontsize=8, ncol=4)
    axes[-1, 0].set_xlabel("retained draw")
    axes[-1, 1].set_xlabel("lag / draws")
    figure.tight_layout()
    return figure, axes


def plot_forecast_months(forecast, *, channel=None, months=tuple(range(1, 13)),
                        level=.95, history=None, history_dates=None,
                        observed=None, history_points=120, threshold=None, direction=None):
    """One panel per calendar month, for predictions or conditional risks.

    Risk curves show the posterior MEAN probability; shading spans conditional
    risk across parameter and future-state draws, not a credible interval for
    the fully integrated probability.
    """
    import matplotlib.pyplot as plt
    from dataclasses import replace
    from ..api.aggregate import monthly_groups
    monthly_groups(forecast.dates)  # Reject daily or gapped inputs to a monthly view.
    # Calendar-month views do not require a seasonal latent state or period=12.
    forecast = replace(forecast, period=12, phases=pd.DatetimeIndex(forecast.dates).month.to_numpy())
    months = tuple(months)
    if not months or any(int(m) != m or not 1 <= m <= 12 for m in months):
        raise ValueError("months must be integers from 1 to 12.")
    cols = min(3, len(months))
    rows = (len(months)+cols-1)//cols
    figure, axes = plt.subplots(rows, cols, figsize=(4*cols, 2.65*rows), squeeze=False)
    present = set(pd.DatetimeIndex(forecast.dates).month)
    for month, ax in zip(months, axes.flat):
        if month not in present:
            ax.text(.5, .5, "Outside forecast window", ha="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        if threshold is None:
            forecast.plot(channel=channel, phase=int(month), ax=ax, level=level,
                          history=history, history_dates=history_dates, history_points=history_points,
                          observed=observed)
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
        else:
            table = forecast.risk_summary(threshold, channel=channel, direction=direction, phase=int(month), level=level)
            ax.fill_between(table.time, table.lower, table.upper, alpha=.2)
            ax.plot(table.time, table['mean'])
            ax.set(xlabel="year", ylabel="event probability", ylim=(-.01, 1.01))
        ax.set_title(calendar.month_name[int(month)], fontsize=11)
        ax.tick_params(axis="x", labelrotation=30)
    for ax in list(axes.flat)[len(months):]:
        ax.set_axis_off()
    if threshold is None:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        handles = [Line2D([], [], color="C0", label="predictive median"),
                   Patch(color="C0", alpha=.2, label=f"{level:.0%} pointwise predictive interval")]
        if history is not None or observed is not None:
            handles.append(Line2D([], [], ls="", marker="o", color="0.5", label="observed"))
        figure.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=9)
        figure.supylabel("temperature / °C", fontsize=11)
        figure.tight_layout(rect=(.015, 0, 1, .97))
    else:
        index = forecast._channel_index(channel)
        tail = forecast.tail[index] if forecast.is_multiseries_forecast else forecast.tail
        event = direction or ("<" if tail == "lower" else ">")
        figure.suptitle(f"Monthly event: Y {event} {threshold:g} °C; line = predictive probability", fontsize=11)
        figure.tight_layout()
    return figure, axes


def plot_scale_calendar(fit, *, channel=None, level=.95):
    """Scale by calendar month, using the last available occurrence of each.

    With RW/linear scale the 12 points need not describe one stationary cycle;
    the selected dates are returned alongside the figure for auditability.
    """
    import matplotlib.pyplot as plt
    dates = pd.DatetimeIndex(fit.time)
    sigma = fit.sigma_draws(channel=channel)
    rows = []
    for month in range(1, 13):
        selected = np.flatnonzero(dates.month == month)
        if not len(selected):
            continue
        j = selected[-1]
        low, median, high = np.quantile(sigma[:, j], [(1-level)/2, .5, (1+level)/2])
        rows.append(dict(month=month, date=dates[j], lower=low, median=median, upper=high))
    table = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(8, 3.5))
    axis.errorbar(table.month, table['median'], yerr=[table['median']-table.lower, table.upper-table['median']],
                  fmt="o-", capsize=3)
    axis.set_xticks(range(1, 13), calendar.month_abbr[1:])
    family = fit.model.channel(channel).family if fit.is_multiseries_model else fit.family
    axis.set(ylabel=("residual SD" if family == "gaussian" else "GEV scale")+" / °C", xlabel="last observed occurrence of each month")
    figure.tight_layout()
    return figure, axis, table


def plot_calendar_risk_curves(aggregate, *, level=.95, thresholds=None, points=60):
    """Calendar event probabilities against thresholds, for up to three periods per season.

    This avoids silently reusing a monthly-mean threshold for an annual mean.
    Plotted lines are predictive probabilities; bands span conditional risks.
    """
    import matplotlib.pyplot as plt
    if not aggregate.n_periods:
        raise ValueError("No complete calendar period to plot.")
    windows = list(dict.fromkeys(aggregate.periods.window))
    figure, axes = plt.subplots(len(windows), 1, figsize=(8, 3*len(windows)), squeeze=False)
    tables = []
    for window, ax in zip(windows, axes[:, 0]):
        indices = np.flatnonzero(aggregate.periods.window.to_numpy() == window)
        selected = np.unique(indices[np.linspace(0, len(indices)-1, min(3, len(indices))).astype(int)])
        grid = np.asarray(thresholds) if thresholds is not None else np.linspace(
            *np.quantile(aggregate.observations[:, indices], [.005, .995]), points)
        table = aggregate.risk_curve(grid, level=level)
        table = table[table.window == window]
        tables.append(table)
        for j in selected:
            label = aggregate.periods.iloc[j].label
            rows = table[table.label == label]
            line, = ax.plot(rows.threshold, rows['mean'], label=label)
            ax.fill_between(rows.threshold, rows.lower, rows.upper, color=line.get_color(), alpha=.1)
        direction = "below" if aggregate.tail == "lower" else "above"
        ax.set(xlabel=f"threshold for {window} {aggregate.reduction} / °C", ylabel=f"probability {direction} threshold", ylim=(-.01, 1.01))
        ax.legend(fontsize=9)
    figure.tight_layout()
    return figure, axes[:, 0], pd.concat(tables, ignore_index=True)
