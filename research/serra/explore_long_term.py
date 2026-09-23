"""Plot descriptive long-term movement in the six monthly Uccle summaries.

This script is deliberately exploratory.  It removes the calendar-month
climatology of a declared reference period and applies the same fixed,
local-linear smoother to every series.  It does not fit a BUCEX model, select
changepoints, estimate uncertainty bands, or perform significance tests.

Run from the project root with

    python -m research.serra.explore_long_term
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import bucex as bx
from bucex.plotting.style import publication_style, save_figure
from .report import new_run


SERIES_LABELS = {
    "TXm": "Mean daily maximum",
    "TXx": "Highest daily maximum",
    "TXn": "Lowest daily maximum",
    "TNm": "Mean daily minimum",
    "TNx": "Highest daily minimum",
    "TNn": "Lowest daily minimum",
}


def calendar_month_anomalies(data, reference):
    """Subtract the reference-period mean separately for each calendar month."""
    start, end = (pd.Period(value, freq="M") for value in reference)
    if start > end:
        raise ValueError("reference must be [start_month, end_month].")
    first, last = data.index.to_period("M")[[0, -1]]
    if start < first or end > last:
        raise ValueError(f"Reference period {start} to {end} lies outside {first} to {last}.")

    reference_data = data.loc[start.start_time:end.end_time]
    counts = reference_data.groupby(reference_data.index.month).count()
    if counts.empty or (counts < 3).any().any():
        raise ValueError("The reference period needs at least three observations per month and series.")
    climatology = reference_data.groupby(reference_data.index.month).mean()

    anomalies = data.copy()
    for month in range(1, 13):
        rows = anomalies.index.month == month
        anomalies.loc[rows] = anomalies.loc[rows] - climatology.loc[month]
    return anomalies, climatology


def local_linear_smooth(data, bandwidth_years=10.0):
    """Apply a fixed tricube local-linear smoother, including at the endpoints.

    ``bandwidth_years`` is the half-width of the moving window.  The same
    bandwidth is used for every series, so differences in apparent smoothness
    are not caused by separate tuning choices.
    """
    bandwidth_years = float(bandwidth_years)
    if not np.isfinite(bandwidth_years) or bandwidth_years <= 0:
        raise ValueError("bandwidth_years must be positive and finite.")

    dates = pd.DatetimeIndex(data.index)
    time = dates.year.to_numpy(float) + (dates.month.to_numpy(float) - 0.5) / 12.0
    result = pd.DataFrame(index=dates, columns=data.columns, dtype=float)

    for name in data:
        observed = data[name].to_numpy(float)
        finite = np.isfinite(observed)
        if finite.sum() < 3:
            raise ValueError(f"{name} has fewer than three finite observations.")
        x = time[finite]
        y = observed[finite]
        fitted = np.empty(len(time), dtype=float)

        for i, target in enumerate(time):
            distance = x - target
            scaled = np.abs(distance) / bandwidth_years
            inside = scaled < 1.0
            if inside.sum() < 3:
                raise ValueError(
                    f"The bandwidth leaves fewer than three observations near {dates[i]:%Y-%m}."
                )
            local_distance = distance[inside]
            weights = (1.0 - scaled[inside] ** 3) ** 3
            design = np.column_stack((np.ones(inside.sum()), local_distance))
            root_weights = np.sqrt(weights)
            coefficients = np.linalg.lstsq(
                design * root_weights[:, None], y[inside] * root_weights, rcond=None
            )[0]
            fitted[i] = coefficients[0]
        result[name] = fitted

    result.index.name = data.index.name or "date"
    return result


def plot_panels(anomalies, smooths, *, colors, bandwidth_years, figsize, style, dpi):
    """Six panels retaining the noisy monthly observations behind each smooth."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    names = list(anomalies.columns)
    with publication_style(style=style, dpi=dpi):
        figure, axes = plt.subplots(
            2, 3, sharex=True, figsize=figsize, layout="constrained"
        )
        for position, (axis, name) in enumerate(zip(axes.flat, names)):
            axis.scatter(
                anomalies.index,
                anomalies[name],
                color="#8b9198",
                s=5,
                alpha=0.27,
                linewidths=0,
                rasterized=True,
            )
            axis.plot(smooths.index, smooths[name], color=colors[name], lw=2.1)
            axis.axhline(0, color="#555555", lw=0.7, alpha=0.45)
            axis.set_title(f"{name}: {SERIES_LABELS[name]}", loc="left", weight="bold")
            if position % 3 == 0:
                axis.set_ylabel("Calendar-month anomaly / Â°C")
            if position >= 3:
                axis.set_xlabel("Year")
        axes[0, 0].legend(
            handles=[
                Line2D([], [], marker="o", ls="", color="#8b9198", alpha=0.55,
                       markersize=4, label="Monthly anomaly"),
                Line2D([], [], color="#333333", lw=2.1,
                       label=f"Local-linear smooth ({bandwidth_years:g}-year half-width)"),
            ],
            loc="upper left",
            fontsize=8.5,
        )
    return figure


def plot_overlay(smooths, *, colors, linestyles, figsize, style, dpi):
    """Overlay the six smooths to display shared and summary-specific movement."""
    import matplotlib.pyplot as plt

    with publication_style(style=style, dpi=dpi):
        figure, axis = plt.subplots(figsize=figsize, layout="constrained")
        for name in smooths:
            axis.plot(
                smooths.index,
                smooths[name],
                color=colors[name],
                ls=linestyles[name],
                lw=2.0,
                label=name,
            )
        axis.axhline(0, color="#555555", lw=0.8, alpha=0.5)
        axis.set(xlabel="Year", ylabel="Smoothed calendar-month anomaly / Â°C")
        axis.legend(ncol=2, loc="upper left")
    return figure


def run(config, *, output=None):
    """Create both figures, their plotted values, and an explicit method record."""
    import matplotlib.pyplot as plt

    data = bx.load_uccle_multiseries(**config["data"])
    order = config.get("series_order", list(data.columns))
    if set(order) != set(data.columns) or len(order) != len(data.columns):
        raise ValueError("series_order must contain every loaded series exactly once.")
    data = data.loc[:, order]

    reference = config["reference"]
    bandwidth = float(config.get("bandwidth_years", 10.0))
    anomalies, climatology = calendar_month_anomalies(data, reference)
    smooths = local_linear_smooth(anomalies, bandwidth)

    settings = config.get("figures", {})
    colors = settings.get("colors", {})
    linestyles = settings.get("linestyles", {})
    if set(colors) != set(order) or set(linestyles) != set(order):
        raise ValueError("figures.colors and figures.linestyles must define every series.")

    directory = new_run(output or config["output"], "long_term_exploration")
    (directory / "data").mkdir()
    bx.save_config(config, directory / "config.json")
    data.to_csv(directory / "data" / "monthly_observations.csv", float_format="%.17g")
    anomalies.to_csv(directory / "data" / "calendar_month_anomalies.csv", float_format="%.17g")
    smooths.to_csv(directory / "data" / "local_linear_smooths.csv", float_format="%.17g")
    climatology.rename_axis("month").to_csv(
        directory / "data" / "reference_climatology.csv", float_format="%.17g"
    )

    formats = tuple(settings.get("formats", ("png", "pdf")))
    dpi = int(settings.get("dpi", 200))
    style = settings.get("style", "manuscript")
    figures = [
        (
            "exploratory_long_term_anomalies",
            plot_panels(
                anomalies,
                smooths,
                colors=colors,
                bandwidth_years=bandwidth,
                figsize=tuple(settings.get("panel_figsize", (10.6, 6.4))),
                style=style,
                dpi=dpi,
            ),
        ),
        (
            "exploratory_shared_smooths",
            plot_overlay(
                smooths,
                colors=colors,
                linestyles=linestyles,
                figsize=tuple(settings.get("overlay_figsize", (8.4, 4.2))),
                style=style,
                dpi=dpi,
            ),
        ),
    ]
    try:
        for name, figure in figures:
            save_figure(figure, directory / "figures" / name, formats=formats, dpi=dpi)
    finally:
        for _, figure in figures:
            plt.close(figure)

    bx.save_config(
        {
            "bucex_version": bx.__version__,
            "observed_start": str(data.index[0].date()),
            "observed_end": str(data.index[-1].date()),
            "n_months": len(data),
            "reference": reference,
            "bandwidth_years": bandwidth,
            "anomaly_definition": (
                "Observation minus the corresponding calendar-month mean in the reference period."
            ),
            "smoother": (
                "Local-linear regression with tricube weights and a fixed half-width; "
                "the same bandwidth is used for all six series. Endpoint values use "
                "asymmetric windows and should be interpreted descriptively."
            ),
            "inferential_status": (
                "Descriptive only: no model selection, confidence bands, hypothesis tests, "
                "or changepoint analysis."
            ),
        },
        directory / "method.json",
    )
    (directory / "README.md").write_text(
        "# Long-term exploratory figures\n\n"
        "`exploratory_long_term_anomalies` retains the monthly observations and "
        "adds one fixed descriptive smooth per series.\n\n"
        "`exploratory_shared_smooths` overlays those same smooths; it contains no "
        "additional estimation. TX summaries are blue and TN summaries red; line "
        "type distinguishes the mean, upper extreme and lower extreme.\n\n"
        "The figures are descriptive model motivation, not evidence for a specific "
        "changepoint or a substitute for the state-space analysis. Smooths near the "
        "start and end of the record use asymmetric windows.\n",
        encoding="utf-8",
    )
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("research/serra/config/revision/long_term_exploration.json"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bandwidth-years", type=float)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"))
    args = parser.parse_args()

    config = bx.load_config(args.config)
    if args.bandwidth_years is not None:
        config["bandwidth_years"] = args.bandwidth_years
    if args.formats:
        config.setdefault("figures", {})["formats"] = args.formats
    directory = run(config, output=args.output)
    print(f"Long-term exploration directory: {directory.resolve()}")


if __name__ == "__main__":
    main()