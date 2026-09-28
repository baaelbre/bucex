"""Descriptive exploration of observed monthly series, without fitting a model."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from collections.abc import Mapping

import numpy as np
import pandas as pd


@dataclass
class MonthlyExploration:
    """Observed monthly cycles and detrended spreads for declared time windows.

    ``seasonal_cycles`` contains empirical means and quartiles; its bands
    describe observation variability, not uncertainty about a mean.
    ``monthly_spread`` contains residual IQRs after a separate intercept and
    linear year trend for each series, calendar month and era. Detrending is
    descriptive only: ``observations`` retains the original values.
    """

    observations: pd.DataFrame
    seasonal_cycles: pd.DataFrame
    monthly_spread: pd.DataFrame
    metadata: dict

    def plot_cycles(self, **kwargs):
        """Return the seasonal-cycle figure; Matplotlib is imported on demand."""
        from .plotting.exploration import plot_exploratory_cycles
        return plot_exploratory_cycles(self, **kwargs)

    def plot_spread(self, **kwargs):
        """Return the within-month detrended-IQR figure."""
        from .plotting.exploration import plot_exploratory_spread
        return plot_exploratory_spread(self, **kwargs)

    def save(self, directory, **kwargs):
        """Save the source snapshot, tables, metadata and optional figures.

        See ``bucex.reporting.exploration.save_exploration_report`` for plotting
        options. ``figures=False`` writes numerical results without Matplotlib.
        """
        from .reporting.exploration import save_exploration_report
        return save_exploration_report(self, directory, **kwargs)


def _monthly_frame(data):
    if isinstance(data, pd.Series):
        data = data.to_frame(name=data.name if data.name is not None else "series")
    if not isinstance(data, pd.DataFrame) or data.empty or not len(data.columns):
        raise ValueError("data must be a nonempty monthly Series or DataFrame.")
    if not data.columns.is_unique or any(not isinstance(c, str) or not c for c in data.columns):
        raise ValueError("Series names must be unique nonempty strings.")
    if isinstance(data.index, pd.PeriodIndex):
        if data.index.freqstr != "M":
            raise ValueError("The PeriodIndex must have monthly frequency M.")
        dates = data.index.to_timestamp()
    elif isinstance(data.index, pd.DatetimeIndex):
        dates = data.index
    else:
        raise ValueError("Use a DatetimeIndex or monthly PeriodIndex for observation dates.")
    if dates.hasnans or dates.tz is not None:
        raise ValueError("Observation dates must be finite and timezone-naive.")
    months = dates.to_period("M")
    if months.duplicated().any():
        raise ValueError("Provide at most one observation per calendar month and series.")
    frame = data.apply(pd.to_numeric, errors="raise").copy()
    if np.iscomplexobj(frame.to_numpy()):
        raise ValueError("Observations must be real numbers.")
    frame = frame.astype(float)
    if np.isinf(frame.to_numpy()).any():
        raise ValueError("Observations cannot contain infinity; missing values may use NaN.")
    frame.index = months.to_timestamp().rename("date")
    return frame.sort_index()


def _windows(specification, frame, name):
    if not isinstance(specification, Mapping) or not specification:
        raise ValueError(f"{name} must map nonempty labels to [start_month, end_month].")
    result = {}
    first, last = frame.index.to_period("M")[[0, -1]]
    for label, limits in specification.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"{name} labels must be nonempty strings.")
        if isinstance(limits, (str, bytes)) or not hasattr(limits, "__len__") or len(limits) != 2:
            raise ValueError(f"{name}[{label!r}] needs [start_month, end_month].")
        start, end = (pd.Period(value, freq="M") for value in limits)
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError(f"Invalid {name} window {label!r}.")
        if start < first or end > last:
            raise ValueError(f"{name} window {label!r} lies outside available months {first} to {last}.")
        result[label] = (start, end)
    return result


def _cells(frame, windows, min_count):
    for label, (start, end) in windows.items():
        block = frame.loc[start.start_time:end.end_time]
        expected = pd.period_range(start, end, freq="M")
        for name in frame.columns:
            for month in range(1, 13):
                values = block.loc[block.index.month == month, name].dropna()
                n_expected = int(np.sum(expected.month == month))
                if len(values) < min_count:
                    raise ValueError(f"{name}, {label}, month {month}: {len(values)} finite observations; "
                                     f"at least {min_count} required.")
                yield name, label, month, values, n_expected


def explore_monthly(data, *, periods, eras, min_count=3):
    """Summarize observed monthly distributions for any named numeric series.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        One value per calendar month, indexed by dates. Values remain on their
        supplied scale (including the original sign of observed minima).
        Missing months/values are excluded and counted in the exported tables.
    periods, eras : mapping
        Ordered mappings ``label: [inclusive_start_month, inclusive_end_month]``.
        Periods select the raw cycle summaries; eras select the separate
        within-calendar-month OLS detrending windows. Both must lie inside the
        observed date range and contain enough data for every calendar month.
    min_count : int, default 3
        Minimum finite observations per series/month/window. Insufficient
        groups raise an error rather than produce misleading spread estimates.

    Returns
    -------
    MonthlyExploration
        Tables, original observations and provenance. No posterior computation,
        smoothing, tail reflection, or preprocessing for subsequent fits occurs.
    """
    if isinstance(min_count, bool) or not isinstance(min_count, (int, np.integer)) or min_count < 3:
        raise ValueError("min_count must be an integer of at least 3.")
    frame = _monthly_frame(data)
    cycle_windows = _windows(periods, frame, "periods")
    spread_windows = _windows(eras, frame, "eras")
    cycles, spreads = [], []
    for name, label, month, values, expected in _cells(frame, cycle_windows, min_count):
        q25, q75 = values.quantile([.25, .75])
        cycles.append(dict(series=name, period=label, month=month, n=len(values),
                           mean=float(values.mean()), q25=float(q25), q75=float(q75),
                           n_expected=expected, n_missing=expected-len(values)))
    for name, label, month, values, expected in _cells(frame, spread_windows, min_count):
        years = values.index.year.to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(values)), years-years.mean()])
        observed = values.to_numpy()
        intercept, slope = np.linalg.lstsq(design, observed, rcond=None)[0]
        residuals = observed-design @ np.array([intercept, slope])
        q25, q75 = np.quantile(residuals, [.25, .75])
        spreads.append(dict(series=name, era=label, month=month, n=len(values),
                            residual_iqr=float(q75-q25), n_expected=expected,
                            n_missing=expected-len(values), trend_per_year=float(slope)))
    snapshot = frame.to_csv(date_format="%Y-%m-%d", float_format="%.17g")
    metadata = dict(
        series=list(frame.columns), observed_start=str(frame.index[0].date()),
        observed_end=str(frame.index[-1].date()), n_months=len(frame),
        n_finite={name: int(frame[name].count()) for name in frame},
        input_sha256=hashlib.sha256(snapshot.encode("utf-8")).hexdigest(),
        periods={label: [str(a), str(b)] for label, (a, b) in cycle_windows.items()},
        eras={label: [str(a), str(b)] for label, (a, b) in spread_windows.items()},
        min_count=int(min_count),
        cycle_method="Observed calendar-month mean and empirical 25th/75th percentiles.",
        spread_method="OLS intercept and linear year trend separately by series, calendar month and era; residual IQR.",
        inferential_status="Descriptive exploration. Empirical bands are not confidence or posterior intervals; no tests are performed.",
        missing_values="Excluded within groups; n_expected and n_missing include absent months.",
        input_scale="Unmodified observation values; detrending is used only to compute exploratory spread.",
    )
    return MonthlyExploration(frame, pd.DataFrame(cycles), pd.DataFrame(spreads), metadata)


__all__ = ["MonthlyExploration", "explore_monthly"]
