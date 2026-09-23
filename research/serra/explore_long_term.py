"""Plot descriptive long-term movement in the six monthly Uccle summaries.

This script is deliberately exploratory. It expresses each observation as an
anomaly from the corresponding calendar-month mean in a declared reference
period and applies the same LOESS specification to all six series. It does not
fit a BUCEX model, select changepoints, estimate uncertainty bands, or perform
significance tests.

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


def loess_smooth(data, span=0.15):
    """Apply degree-one LOESS with tricube weights and no robust reweighting.

    ``span`` is the fraction of finite observations used in each local fit.
    The same span is used for every series. With the Uccle record, a span of
    0.15 corresponds to a neighbourhood of approximately 20 years.
    """
    span = float(span)
    if not np.isfinite(span) or not 0 < span <= 1:
        raise ValueError("span must be finite and lie in (0, 1].")

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
        neighbours = max(3, int(np.ceil(span * len(x))))
        fitted = np.empty(len(time), dtype=float)

        for i, target in enumerate(time):
            distance = np.abs(x - target)
            bandwidth = np.partition(distance, neighbours - 1)[neighbours - 1]
            if bandwidth <= 0:
                raise ValueError(f"LOESS neighbourhood is degenerate near {dates[i]:%Y-%m}.")

            inside = distance <= bandwidth
            scaled = np.minimum(distance[inside] / bandwidth, 1.0)
            weights = (1.0 - scaled**3) ** 3
            local_time = x[inside] - target
            design = np.column_stack((np.ones(inside.sum()), local_time))
            root_weights = np.sqrt(weights)
            coefficients = np.linalg.lstsq(
                design * root_weights[:, None], y[inside] * root_weights, rcond=None
            )[0]
            fitted[i] = coefficients[0]
        result[name] = fitted

    result.index.name = data.index.name or "date"
    return result


def plot_panels(anomalies, smooths, *, colors, loess_span, figsize, style, dpi):
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
                axis.set_ylabel("Calendar-month anomaly / °C")
            if position >= 3:
                axis.set_xlabel("Year")
        axes[0, 0].legend(
            handles=[
                Line2D([], [], marker="o", ls="", color="#8b9198", alpha=0.55,
                       markersize=4, label="Monthly anomaly"),
                Line2D([], [], color="#333333", lw=2.1,
                       label=f"LOESS (span = {loess_span:g})"),
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
        axis.set(xlabel="Year", ylabel="Calendar-month anomaly / °C")
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
    span = float(config.get("loess_span", 0.15))
    anomalies, climatology = calendar_month_anomalies(data, reference)
    smooths = loess_smooth(anomalies, span)

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
    smooths.to_csv(directory / "data" / "loess_smooths.csv", float_format="%.17g")
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
                loess_span=span,
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
            "loess_span": span,
            "loess_degree": 1,
            "robust_reweighting": False,
            "anomaly_definition": (
                "Observation minus the corresponding calendar-month mean in the reference period."
            ),
            "smoother": (
                "Degree-one LOESS with tricube distance weights and no robust "
                "reweighting; the same span is used for all six series. Neighbourhoods "
                "near the endpoints are one-sided and should be interpreted descriptively."
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
        "adds one descriptive degree-one LOESS smooth per series.\n\n"
        "`exploratory_shared_smooths` overlays those same smooths; it contains no "
        "additional estimation. TX summaries are blue and TN summaries red; line "
        "type distinguishes the mean, upper extreme and lower extreme.\n\n"
        "The figures are descriptive model motivation, not evidence for a specific "
        "changepoint or a substitute for the state-space analysis. The same LOESS "
        "span is used for all series; endpoint neighbourhoods are one-sided.\n",
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
    parser.add_argument("--loess-span", type=float)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"))
    args = parser.parse_args()

    config = bx.load_config(args.config)
    if args.loess_span is not None:
        config["loess_span"] = args.loess_span
    if args.formats:
        config.setdefault("figures", {})["formats"] = args.formats
    directory = run(config, output=args.output)
    print(f"Long-term exploration directory: {directory.resolve()}")


if __name__ == "__main__":
    main()
