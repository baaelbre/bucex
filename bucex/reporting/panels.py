"""Data-driven publication panels. Scientific estimates are never refitted."""
from __future__ import annotations

import calendar
import numpy as np
import pandas as pd
from scipy.special import ndtri

from ..diagnostics.calendar import pit_by_month
from ..plotting.calendar import plot_monthly_score_sd
from ..plotting.predictive import plot_chain_traces, plot_pit_diagnostics
from ..plotting.traces import traces_from_frame
from ..plotting.style import PUBLICATION_COLORS


def _color(name, colors, index=0):
    return colors.get(name, PUBLICATION_COLORS[index % len(PUBLICATION_COLORS)])


def _grid(names, *, columns=3, sharex=False, sharey=False, height=3.1):
    import matplotlib.pyplot as plt
    columns = min(columns, len(names))
    rows = (len(names)+columns-1)//columns
    figure, axes = plt.subplots(rows, columns, figsize=(10.6, rows*height),
        squeeze=False, sharex=sharex, sharey=sharey, layout="constrained")
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    for name, ax in zip(names, axes.flat):
        ax.set_title(name, loc="left", weight="bold")
    return figure, axes


def _band(ax, data, color, *, x="time", center="median", multiplier=1., label=None):
    required = {x, "lower", center, "upper"}
    if not required <= set(data):
        raise ValueError(f"Band table requires {sorted(required)}.")
    if data.empty:
        raise ValueError("Requested panel has no observations.")
    low, high = data.lower.to_numpy(), data.upper.to_numpy()
    if np.any(low > high):
        raise ValueError("Invalid interval: lower exceeds upper.")
    ax.fill_between(data[x], multiplier*low, multiplier*high, color=color, alpha=.20, lw=0)
    ax.plot(data[x], multiplier*data[center], color=color, label=label)


def _year_ticks(ax, *, dates=True):
    if dates:
        import matplotlib.dates as mdates
        span = (ax.get_xlim()[1]-ax.get_xlim()[0])/365.25
        step = next((n for n in (1,2,5,10,20,25,50,100,200,500) if n >= span/5), 1000)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=step))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))


def bands(reports, spec, names, colors):
    figure, axes = _grid(names, columns=spec.get("columns", 3), sharex=True)
    for i, (name, ax) in enumerate(zip(names, axes.flat)):
        data = reports.read(name, spec["table"])
        if "month" in spec:
            data = data[data.time.dt.month.eq(spec["month"])]
        if "window" in spec:
            data = data[data.window.eq(spec["window"])]
        _band(ax, data, _color(name, colors, i), x=spec.get("x", "time"))
        _year_ticks(ax, dates=spec.get('x','time') == 'time')
        if spec.get("zero", False):
            ax.axhline(0, color=".45", lw=.8, ls="--")
        if i % axes.shape[1] == 0:
            ax.set_ylabel(spec.get("ylabel", "Value"))
        if i >= len(names)-axes.shape[1]:
            ax.set_xlabel("Year")
    return figure, {}


def monthly_pit(reports, spec, names, colors):
    figure, axes = _grid(names, sharex=True, sharey=True, height=2.8)
    tables = []
    for i, (name, ax) in enumerate(zip(names, axes.flat)):
        pit = reports.read(name, "smoothed_pit")
        data = pit_by_month(pit.pit, pit.time)
        tables.append(data.assign(series=name))
        plot_monthly_score_sd(data, ax=ax, color=_color(name, colors, i))
        if i % axes.shape[1]:
            ax.set_ylabel("")
    return figure, {"monthly_diagnostics": pd.concat(tables, ignore_index=True)}


def qq(reports, spec, names, colors):
    figure, axes = _grid(names)
    scores = reports.normal_scores(series=names)
    for i, (name, ax) in enumerate(zip(names, axes.flat)):
        values = np.sort(scores[name].to_numpy())
        theoretical = ndtri((np.arange(len(values))+.5)/len(values))
        ax.scatter(theoretical, values, s=5, alpha=.65, color=_color(name, colors, i), rasterized=True)
        bounds = [min(theoretical.min(), values.min()), max(theoretical.max(), values.max())]
        ax.plot(bounds, bounds, color=".4", ls="--", lw=1)
        ax.set(xlabel="Standard-normal quantile", ylabel="Smoothed PIT normal score")
    return figure, {}


def dependence(reports, spec, names, colors):
    import matplotlib.pyplot as plt
    scores = reports.normal_scores(series=names)
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), layout="constrained",
                               gridspec_kw={"width_ratios": [1, 1.1]})
    corr = scores.corr()
    axes[0].grid(False)
    artist = axes[0].imshow(corr, cmap="RdBu", vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(names)), names, rotation=45, ha="right")
    axes[0].set_yticks(range(len(names)), names)
    for i, j in np.ndindex(corr.shape):
        value = corr.iloc[i, j]
        axes[0].text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9,
                     color="white" if abs(value)>.65 else "#143349")
    figure.colorbar(artist, ax=axes[0], shrink=.8, label="Score correlation")
    axes[0].set_title("(a) Pooled residual scores", loc="left")
    rows = []
    for i, (left, right) in enumerate(spec.get("pairs", [])):
        if left not in scores or right not in scores:
            raise ValueError(f"Dependence pair {left}/{right} is absent from selected reports.")
        values = []
        for month in range(1, 13):
            data = scores.loc[scores.index.month == month, [left, right]]
            value = data.corr().iloc[0, 1] if len(data)>1 else np.nan
            values.append(value)
            rows.append(dict(left=left, right=right, month=month, n=len(data), correlation=value))
        axes[1].plot(range(1, 13), values, "o-", label=f"{left} / {right}",
                     color=PUBLICATION_COLORS[i % len(PUBLICATION_COLORS)])
    axes[1].set_xticks([1, 3, 5, 7, 9, 11], ["Jan", "Mar", "May", "Jul", "Sep", "Nov"])
    axes[1].set_ylim(-1.02 if (corr.to_numpy()<0).any() or any(r["correlation"]<0 for r in rows) else 0, 1.02)
    axes[1].set_ylabel("Score correlation")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].set_title("(b) Within calendar months", loc="left")
    return figure, {"residual_correlations": corr.rename_axis("series").reset_index(),
                     "residual_correlations_by_month": pd.DataFrame(rows),
                     "smoothed_normal_scores": scores.rename_axis("time").reset_index()}


def risks(reports, spec, names, colors):
    events = spec["events"]
    figure, axes = _grid([event["series"] for event in events], columns=2, sharex=True)
    for i, (event, ax) in enumerate(zip(events, axes.flat)):
        name, month, threshold = event["series"], event["month"], event["threshold"]
        reports.check_event(name, threshold)
        data = reports.read(name, "risk")
        data = data[data.time.dt.month.eq(month)]
        _band(ax, data, event.get('color',_color(name, colors, i)), multiplier=100., center=spec.get("center", "median"))
        _year_ticks(ax)
        ax.set_title(f"{calendar.month_name[month]}: {name} {event['direction']} {threshold:g}°C",
                     loc="left", weight="normal", fontsize=12)
        ax.set_ylabel("Conditional probability / %")
        if i >= len(events)-2:
            ax.set_xlabel("Year")
    return figure, {}


def forecast(reports, spec, names, colors):
    import matplotlib.pyplot as plt
    month = spec.get("month", 7)
    level = reports.interval_level()
    figure, axes = plt.subplots(2, len(names), figsize=(10.6, 6.8), squeeze=False,
                               layout="constrained")
    widths = []
    for col, name in enumerate(names):
        data = reports.read(name, "forecast_monthly")
        data = data[data.time.dt.month.eq(month)]
        if data.empty:
            raise FileNotFoundError(f"No {calendar.month_name[month]} forecast in {name}'s saved horizon.")
        if not {"eta_lower", "eta_median", "eta_upper"} <= set(data):
            raise ValueError("Forecast figure requires saved location and observation intervals.")
        ax = axes[0, col]
        ax.fill_between(data.time, data.lower, data.upper, color=PUBLICATION_COLORS[0], alpha=.14,
                        label=f"Observation: {level:.0%} PI")
        ax.fill_between(data.time, data.eta_lower, data.eta_upper, color=PUBLICATION_COLORS[1], alpha=.28,
                        label=f"Location: {level:.0%} interval")
        ax.plot(data.time, data.eta_median, color=PUBLICATION_COLORS[1])
        ax.set_title(f"{name}: {calendar.month_name[month]} forecasts", loc="left", fontsize=12)
        ax.set_ylabel("Temperature / °C")
        _year_ticks(ax)
        ax.legend(fontsize=9)
        ax = axes[1, col]
        for prefix, label, color in [("", "Observation", PUBLICATION_COLORS[0]),
                                     ("eta_", "Location", PUBLICATION_COLORS[1])]:
            width = data[prefix+"upper"]-data[prefix+"lower"]
            ax.plot(data.time, width, "o-", color=color, label=label)
            widths.append(pd.DataFrame(dict(series=name, time=data.time, component=label,
                                             interval_level=level, width=width)))
        ax.set(xlabel="Year", ylabel=f"{level:.0%} interval width / °C")
        _year_ticks(ax)
    return figure, {"forecast_widths": pd.concat(widths, ignore_index=True)}


def scales(reports, spec, names, colors):
    figure, axes = _grid(names, sharex=True)
    for i, (name, ax) in enumerate(zip(names, axes.flat)):
        data = reports.read(name, "scale_by_month")
        ax.errorbar(data.month, data["median"], yerr=[data["median"]-data.lower, data.upper-data["median"]],
                     fmt="o-", capsize=3, color=_color(name, colors, i))
        ax.set_xticks([1, 3, 5, 7, 9, 11], ["Jan", "Mar", "May", "Jul", "Sep", "Nov"])
        ax.set_ylabel("Observation scale / °C")
    return figure, {}


def traces(reports, spec, names, colors):
    if len(names) != 1:
        raise ValueError("A trace recipe selects one series.")
    data = reports.read(names[0], spec.get("table", "parameter_traces"))
    draws = traces_from_frame(data)
    selected = spec.get("parameters")
    if selected is not None:
        draws = {key: draws[key] for key in selected}
    figure, _ = plot_chain_traces(draws)
    return figure, {}


def pit_diagnostics(reports, spec, names, colors):
    if len(names) != 1:
        raise ValueError("A PIT diagnostic recipe selects one series.")
    data = reports.read(names[0], "smoothed_pit")
    figure, _ = plot_pit_diagnostics(data.pit, data.time, in_sample=True)
    return figure, {}


PANEL_BUILDERS = {"bands": bands, "monthly_pit": monthly_pit, "qq": qq,
    "dependence": dependence, "risks": risks, "forecast": forecast,
    "scales": scales, "traces": traces, "pit_diagnostics": pit_diagnostics}
