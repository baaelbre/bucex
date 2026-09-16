"""Draw-wise calendar aggregation of monthly posterior predictions.

Monthly means use calendar-day weights. Monthly block maxima/minima use the
maximum/minimum of the simulated observations, never of fitted locations.
Residuals are conditionally independent across time in the current models;
common posterior parameters and latent trajectories remain shared in a draw.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.special import ndtr


def monthly_groups(dates, *, frequency="year", months=None, include_partial=False):
    """Return calendar windows; DJF is labelled by its January/February year.

    ``months=(12, 1, 2)`` is a custom winter window with the same convention.
    A partial window is excluded unless explicitly requested and always carries
    ``complete=False``. Dates must be consecutive, unique monthly blocks.
    """
    raw = np.asarray(dates)
    if np.issubdtype(raw.dtype, np.number):
        raise ValueError("Calendar aggregation requires dated monthly observations.")
    dates = pd.DatetimeIndex(raw)
    if dates.hasnans or not len(dates):
        raise ValueError("Calendar dates must be nonempty and finite.")
    if np.any(np.diff(dates.to_period("M").asi8) != 1):
        raise ValueError("Calendar aggregation requires consecutive, unique monthly dates.")
    if frequency not in {"year", "season"}:
        raise ValueError("frequency must be 'year' or 'season'.")
    if months is not None and frequency != "year":
        raise ValueError("Custom months use frequency='year'.")
    if months is not None:
        months = tuple(months)
        if (not months or any(int(m) != m or not 1 <= m <= 12 for m in months)
                or len(set(months)) != len(months)):
            raise ValueError("months must contain distinct integers from 1 to 12.")
        months = tuple(map(int, months))
        if any((b-a) % 12 != 1 for a, b in zip(months, months[1:])):
            raise ValueError("Custom months must be in consecutive calendar order.")
    windows = ({"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
               if frequency == "season" else {"year" if months is None else "custom": months or tuple(range(1, 13))})
    groups, rows = [], []
    for window, selected in windows.items():
        wrap = next((i for i in range(1, len(selected)) if selected[i] < selected[i-1]), None)
        previous_year = set(selected[:wrap]) if wrap is not None else set()
        years = dates.year.to_numpy() + np.isin(dates.month, list(previous_year))
        valid = np.isin(dates.month, selected)
        for year in np.unique(years[valid]):
            indices = np.flatnonzero(valid & (years == year))
            complete = len(indices) == len(selected)
            if not complete and not include_partial:
                continue
            groups.append(indices)
            rows.append(dict(label=f"{year}" if window == "year" else f"{window} {year}",
                             year=int(year), window=window, start=dates[indices[0]], end=dates[indices[-1]],
                             months=len(indices), expected_months=len(selected), complete=complete))
    order = sorted(range(len(rows)), key=lambda j: rows[j]["start"])
    columns = ["label", "year", "window", "start", "end", "months", "expected_months", "complete"]
    return [groups[j] for j in order], pd.DataFrame([rows[j] for j in order], columns=columns)


def _band(values, level):
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1).")
    low, median, high = np.quantile(values, [(1-level)/2, .5, (1+level)/2], axis=0)
    return dict(mean=np.mean(values, axis=0), lower=low, median=median, upper=high)


@dataclass
class AggregateForecast:
    """Calendar predictive draws and conditional aggregate event probabilities.

    Probability bands vary over posterior parameters AND future state paths.
    They are not Monte Carlo error bars or parameter-only risk intervals.
    """
    forecast: object
    observations: np.ndarray
    periods: pd.DataFrame
    groups: list
    weights: list
    reduction: str
    channel: str | None
    weighting: str

    @property
    def n_periods(self):
        return len(self.groups)

    @property
    def tail(self):
        f = self.forecast
        return f.tail[f._channel_index(self.channel)] if f.is_multiseries_forecast else f.tail

    def aggregate_values(self, values):
        """Apply the same calendar operation to a time vector or aligned draws."""
        values = np.asarray(values, dtype=float)
        if values.shape[-1] != self.forecast.horizon:
            raise ValueError("The final dimension must match the monthly forecast horizon.")
        result = []
        for indices, weights in zip(self.groups, self.weights):
            selected = values[..., indices]
            result.append(np.sum(selected * weights, axis=-1) if self.reduction == "mean"
                          else getattr(np, self.reduction)(selected, axis=-1))
        return np.stack(result, axis=-1) if result else np.empty(values.shape[:-1] + (0,))

    def summary(self, level=.95):
        """Predictive intervals for realised calendar averages/maxima/minima."""
        return self.periods.assign(reduction=self.reduction, weighting=self.weighting,
                                   **_band(self.observations, level))

    def conditional_moments(self):
        """Mean and variance for Gaussian calendar averages, per posterior path.

        Observation variance is sum(w_t**2 * sigma_t**2), not mean(sigma_t)**2.
        This does not remove posterior covariance between the latent means.
        """
        f = self.forecast
        model = f.observation_model[self.channel] if f.is_multiseries_forecast else f.observation_model
        if self.reduction != "mean" or model.name != "gaussian":
            raise NotImplementedError("Analytic aggregate moments require a Gaussian mean.")
        eta = f._target_draws("predictor", channel=self.channel)
        sigma = f.sigma_draws(channel=self.channel)
        means = self.aggregate_values(eta)
        variances = [np.sum(sigma[:, idx]**2 * weights**2, axis=1)
                     for idx, weights in zip(self.groups, self.weights)]
        return means, np.stack(variances, axis=-1) if variances else np.empty_like(means)

    def probability_draws(self, threshold, *, direction=None):
        """Integrate observation noise within each posterior/state draw.

        Analytic for Gaussian means and for maxima/minima of all supported
        continuous marginals. Temporal conditional independence is assumed.
        A cross-channel contemporaneous copula does not add serial dependence.
        """
        direction = direction or ("<" if self.tail == "lower" else ">")
        if direction not in {"<", ">"} or not np.isfinite(threshold):
            raise ValueError("Use a finite threshold and direction '<' or '>'.")
        if not self.n_periods:
            return np.empty((self.forecast.n_draws, 0))
        if self.reduction == "mean":
            means, variances = self.conditional_moments()
            z = (float(threshold)-means)/np.sqrt(variances)
            return ndtr(z if direction == "<" else -z)
        cdf = self.forecast.conditional_cdf(np.full(self.forecast.horizon, threshold), channel=self.channel)
        probabilities = []
        for indices in self.groups:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_product = np.sum(np.log(cdf[:, indices]) if self.reduction == "max"
                                     else np.log1p(-cdf[:, indices]), axis=1)
            same_side = (direction == "<") == (self.reduction == "max")
            probabilities.append(np.exp(log_product) if same_side else -np.expm1(log_product))
        return np.stack(probabilities, axis=1)

    def risk_summary(self, threshold, *, direction=None, level=.95):
        values = self.probability_draws(threshold, direction=direction)
        return self.periods.assign(threshold=float(threshold),
            direction=direction or ("<" if self.tail == "lower" else ">"),
            reduction=self.reduction, **_band(values, level))

    def risk_curve(self, thresholds, *, direction=None, level=.95):
        """Probability against threshold, retaining labels for every period."""
        tables = [self.risk_summary(value, direction=direction, level=level) for value in thresholds]
        if not tables:
            raise ValueError("Supply at least one risk threshold.")
        return pd.concat(tables, ignore_index=True)

    def plot(self, *, level=.95, ax=None, ylabel=None, window=None):
        """Plot predictive intervals; facet seasons by passing window='DJF', etc."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 3.5))
        table = self.summary(level)
        if window is not None:
            table = table[table.window == window]
        single_window = table.window.nunique() <= 1
        x = table.year if single_window else table.end
        if len(table) == 1:
            ax.errorbar(x, table['median'], yerr=[table['median']-table.lower, table.upper-table['median']],
                        fmt="o", capsize=4, label=f"median and {level:.0%} predictive interval")
        else:
            ax.fill_between(x, table.lower, table.upper, alpha=.2, label=f"{level:.0%} predictive interval")
            ax.plot(x, table['median'], label="predictive median")
        ax.set(xlabel="year (DJF uses ending year)" if single_window else "period ending",
               ylabel=ylabel or f"calendar {self.reduction} / °C")
        if single_window:
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        return ax


def aggregate_forecast(forecast, *, frequency="year", reduction=None, channel=None,
                       months=None, weighting="days", include_partial=False):
    """Aggregate aligned MONTHLY predictive draws without destroying dependence.

    Defaults: Gaussian monthly summaries -> mean; upper/lower GEV -> max/min.
    Override ``reduction`` if the Gaussian observations represent another
    quantity. ``weighting='days'`` assumes a mean over all days in each month;
    use ``'equal'`` only for a deliberately equal-month estimand.
    """
    index = forecast._channel_index(channel)
    model = forecast.observation_model[channel] if forecast.is_multiseries_forecast else forecast.observation_model
    tail = forecast.tail[index] if forecast.is_multiseries_forecast else forecast.tail
    reduction = reduction or ("mean" if model.name == "gaussian" else "min" if tail == "lower" else "max")
    if reduction not in {"mean", "max", "min"}:
        raise ValueError("reduction must be 'mean', 'max', or 'min'.")
    if weighting not in {"days", "equal"}:
        raise ValueError("weighting must be 'days' or 'equal'.")
    groups, periods = monthly_groups(forecast.dates, frequency=frequency, months=months,
                                     include_partial=include_partial)
    day_counts = pd.DatetimeIndex(forecast.dates).days_in_month.to_numpy(dtype=float)
    weights = [day_counts[idx]/sum(day_counts[idx]) if weighting == "days"
               else np.ones(len(idx))/len(idx) for idx in groups]
    result = AggregateForecast(forecast, np.empty((forecast.n_draws, 0)), periods, groups,
                               weights, reduction, channel, weighting if reduction == "mean" else "none")
    result.observations = result.aggregate_values(forecast._target_draws("observations", channel=channel))
    return result
