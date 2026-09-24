"""Build the seasonal manuscript figures from auditable report tables.

The final-report figures are required.  Sensitivity, validation, block-frequency
and pre-2019 panels are generated only when their completed run directories are
supplied.  No model is fitted by this command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import bucex as bx
from research.seasonal.prepare import exploratory_structure


SERIES = ("TXm", "TNm", "TXx", "TXn", "TNx", "TNn")
SEASONS = ("DJF", "MAM", "JJA", "SON")
TX = "#24658a"
TN = "#a44839"
INK = "#30363d"
GRID = "#d9e0e5"
SEASON_COLORS = ("#5b8fb0", "#d07a55", "#7a9f72", "#8b6fa8")
RISK_SPECS = (
    ("TXx", 6, "JJA", r"$P(\mathrm{TXx}>35^\circ\mathrm{C})$", TX),
    ("TNx", 6, "JJA", r"$P(\mathrm{TNx}>25^\circ\mathrm{C})$", TN),
    ("TXn", 12, "DJF", r"$P(\mathrm{TXn}<0^\circ\mathrm{C})$", TX),
    ("TNn", 12, "DJF", r"$P(\mathrm{TNn}<-10^\circ\mathrm{C})$", TN),
)


class Inputs:
    def __init__(self):
        self.sources = {}

    def _record(self, path: Path):
        path = path.resolve()
        self.sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def csv(self, path: Path, **kwargs):
        return pd.read_csv(self._record(path), **kwargs)

    def config(self, path: Path):
        self._record(path)
        return bx.load_config(path)


def _series_color(name):
    return TX if name.startswith("TX") else TN


def _save(figure, output, name, formats, dpi, figures):
    paths = bx.save_figure(figure, output / name, formats=formats, dpi=dpi, close=True)
    figures.append({"name": name, "files": [path.name for path in paths]})


def _report_directory(path: Path, marker="period_contrasts.csv"):
    path = path.resolve()
    if (path / marker).is_file():
        return path
    matches = sorted(path.rglob(marker))
    if len(matches) != 1:
        raise ValueError(f"Expected one {marker} below {path}; found {len(matches)}.")
    return matches[0].parent


def _band_panels(inputs, run, table, ylabel, *, zero=False):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(10.8, 6.3), sharex=True, layout="constrained")
    for index, (name, axis) in enumerate(zip(SERIES, axes.flat)):
        frame = inputs.csv(run / f"{name}_{table}.csv", parse_dates=["time"])
        x = frame.time.to_numpy()
        color = _series_color(name)
        axis.fill_between(x, frame.lower.to_numpy(), frame.upper.to_numpy(), color=color, alpha=.18, lw=0)
        axis.plot(x, frame["median"].to_numpy(), color=color, lw=1.55)
        if zero:
            axis.axhline(0, color=INK, lw=.75, ls="--")
        axis.set_title(name, loc="left", weight="bold")
        if index % 3 == 0:
            axis.set_ylabel(ylabel)
        if index >= 3:
            axis.set_xlabel("time")
    return figure


def _scale_panels(inputs, run):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(10.8, 6.5), sharex=True, layout="constrained")
    for index, (name, axis) in enumerate(zip(SERIES, axes.flat)):
        frame = inputs.csv(run / f"{name}_scale_by_season.csv").set_index("season").loc[list(SEASONS)]
        x = np.arange(4)
        color = _series_color(name)
        axis.errorbar(x, frame["median"],
                      yerr=[frame["median"] - frame.lower, frame.upper - frame["median"]],
                      fmt="o-", color=color, capsize=3, lw=1.35)
        axis.set_title(name, loc="left", weight="bold")
        axis.set_xticks(x, SEASONS)
        if index % 3 == 0:
            axis.set_ylabel("observation scale / °C")
    return figure


def _acf(values, lags=8):
    values = np.asarray(values, dtype=float)
    values = values - values.mean()
    denominator = np.dot(values, values)
    if denominator <= 0:
        return np.zeros(lags + 1)
    return np.asarray([1.] + [np.dot(values[:-lag], values[lag:]) / denominator
                              for lag in range(1, lags + 1)])


def _pit_panels(inputs, run):
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(12.4, 8.8), layout="constrained")
    outer = figure.add_gridspec(2, 3)
    for position, name in enumerate(SERIES):
        frame = inputs.csv(run / f"{name}_smoothed_pit.csv", parse_dates=["time"])
        pit = frame.pit.to_numpy(float)
        scores = norm.ppf(np.clip(pit, 1e-10, 1 - 1e-10))
        inner = outer[position // 3, position % 3].subgridspec(2, 2, wspace=.28, hspace=.34)
        axes = [figure.add_subplot(inner[i, j]) for i in range(2) for j in range(2)]
        axes[0].hist(pit, bins=np.linspace(0, 1, 11), color=_series_color(name), alpha=.72)
        axes[0].axhline(len(pit) / 10, color=INK, lw=.7, ls="--")
        axes[0].set(title=name, xlabel="PIT", ylabel="count")
        theoretical = norm.ppf((np.arange(len(scores)) + .5) / len(scores))
        ordered = np.sort(scores)
        limits = [min(theoretical[0], ordered[0]), max(theoretical[-1], ordered[-1])]
        axes[1].scatter(theoretical, ordered, s=5, color=_series_color(name), alpha=.6)
        axes[1].plot(limits, limits, color=INK, lw=.7, ls="--")
        axes[1].set(xlabel="normal quantile", ylabel="score quantile")
        axes[2].plot(frame.time, scores, color=_series_color(name), lw=.55)
        axes[2].axhline(0, color=INK, lw=.65)
        axes[2].set(xlabel="time", ylabel="normal score")
        correlations = _acf(scores)
        axes[3].vlines(np.arange(1, len(correlations)), 0, correlations[1:], color=_series_color(name), lw=1.2)
        bound = 1.96 / np.sqrt(len(scores))
        axes[3].axhline(bound, color=INK, lw=.6, ls="--")
        axes[3].axhline(-bound, color=INK, lw=.6, ls="--")
        axes[3].set(xlabel="lag / seasons", ylabel="ACF", xlim=(.5, len(correlations) - .5))
        for axis in axes:
            axis.tick_params(labelsize=7.5)
            axis.xaxis.label.set_size(8)
            axis.yaxis.label.set_size(8)
            axis.title.set_fontsize(10)
            axis.title.set_weight("bold")
    return figure


def _shared_shrinkage(inputs, run):
    import matplotlib.pyplot as plt

    frame = inputs.csv(run / "shared_shrinkage.csv")
    components = ("level", "slope", "seasonal", "initial_slope")
    figure, axes = plt.subplots(1, 4, figsize=(12.3, 2.9), layout="constrained")
    for axis, component in zip(axes, components):
        rows = frame[frame.component.eq(component)].set_index("distribution")
        for y, distribution, marker, alpha in ((0, "prior", "s", .55), (1, "posterior", "o", 1.)):
            row = rows.loc[distribution]
            axis.errorbar(row["median"], y,
                          xerr=[[row["median"] - row.lower], [row.upper - row["median"]]],
                          fmt=marker, color="#78618c", alpha=alpha, capsize=3)
        axis.set_yticks([0, 1], ["hyperprior", "posterior"])
        axis.set_title(component.replace("_", " "), loc="left", weight="bold")
        axis.set_xlabel("shared scale")
        axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3), useMathText=True)
    return figure


def _copula(inputs, run):
    import matplotlib.pyplot as plt

    frame = inputs.csv(run / "copula_correlations.csv")
    matrices = {phase: np.eye(len(SERIES)) for phase in range(1, 5)}
    for row in frame.itertuples():
        parts = str(row.quantity).split(".")
        if len(parts) != 5 or parts[0] != "phase" or parts[2] != "rho":
            continue
        phase, left, right = int(parts[1]), parts[3], parts[4]
        if left in SERIES and right in SERIES:
            i, j = SERIES.index(left), SERIES.index(right)
            matrices[phase][i, j] = matrices[phase][j, i] = float(row.median)
    figure, axes = plt.subplots(2, 2, figsize=(8.7, 7.4), layout="constrained")
    image = None
    for phase, season, axis in zip(range(1, 5), SEASONS, axes.flat):
        image = axis.imshow(matrices[phase], vmin=-1, vmax=1, cmap="RdBu_r")
        axis.set_title(season, loc="left", weight="bold")
        axis.set_xticks(range(6), SERIES, rotation=45, ha="right")
        axis.set_yticks(range(6), SERIES)
        for i in range(6):
            for j in range(6):
                value = matrices[phase][i, j]
                axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.8,
                          color="white" if abs(value) > .55 else INK)
    figure.colorbar(image, ax=axes, shrink=.76, label="posterior median correlation")
    return figure


def _contrast_figures(inputs, run):
    import matplotlib.pyplot as plt

    frame = inputs.csv(run / "period_contrasts.csv").set_index("quantity")
    seasons = (("month_03", "MAM"), ("month_06", "JJA"),
               ("month_09", "SON"), ("month_12", "DJF"))
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.1, 4.8),
                                        gridspec_kw={"width_ratios": [1.18, 1]}, layout="constrained")
    ybase = np.arange(len(SERIES))[::-1]
    for offset, (code, season), color in zip(np.linspace(-.24, .24, 4), seasons, SEASON_COLORS):
        rows = frame.loc[[f"{name}.location.{code}.change" for name in SERIES]]
        median = rows["median"].to_numpy()
        left.errorbar(median, ybase + offset,
                      xerr=[median - rows.lower.to_numpy(), rows.upper.to_numpy() - median],
                      fmt="o", ms=4, lw=1.1, capsize=2, color=color, label=season)
    left.axvline(0, color=INK, lw=.8)
    left.set(yticks=ybase, yticklabels=SERIES, xlabel="late–early location change / °C")
    left.set_title("A  Change by response and season", loc="left", weight="bold")
    left.legend(frameon=False, ncol=4, loc="lower right", fontsize=8)

    pairs = (("TXx_minus_TXm.level.change", "TXx – TXm"),
             ("TXn_minus_TXm.level.change", "TXn – TXm"),
             ("TNx_minus_TNm.level.change", "TNx – TNm"),
             ("TNn_minus_TNm.level.change", "TNn – TNm"),
             ("TXm_minus_TNm.level.change", "TXm – TNm"),
             ("TXx_minus_TNx.level.change", "TXx – TNx"),
             ("TXn_minus_TNn.level.change", "TXn – TNn"))
    y = np.arange(len(pairs))[::-1]
    for yi, (quantity, _) in zip(y, pairs):
        row = frame.loc[quantity]
        right.errorbar(row["median"], yi,
                       xerr=[[row["median"] - row.lower], [row.upper - row["median"]]],
                       fmt="o", color=TX if row["median"] >= 0 else TN, capsize=2)
    right.axvline(0, color=INK, lw=.8, ls="--")
    right.set(yticks=y, yticklabels=[label for _, label in pairs],
              xlabel="difference in late–early level change / °C")
    right.set_title("B  Departure from an additive shift", loc="left", weight="bold")

    rate, axis = plt.subplots(figsize=(7.3, 4.5), layout="constrained")
    y = np.arange(len(SERIES))[::-1]
    for yi, name in zip(y, SERIES):
        recent, change = frame.loc[f"{name}.slope.comparison"], frame.loc[f"{name}.slope.change"]
        axis.errorbar(recent["median"], yi + .12,
                      xerr=[[recent["median"] - recent.lower], [recent.upper - recent["median"]]],
                      fmt="o", color=TX, capsize=2, label="recent-period rate" if yi == y[0] else None)
        axis.errorbar(change["median"], yi - .12,
                      xerr=[[change["median"] - change.lower], [change.upper - change["median"]]],
                      fmt="s", color=TN, capsize=2, label="recent minus early" if yi == y[0] else None)
    axis.axvline(0, color=INK, lw=.8, ls="--")
    axis.set(yticks=y, yticklabels=SERIES, xlabel="rate / °C per decade")
    axis.legend(loc="lower right")
    return figure, rate


def _risk_panels(inputs, run, forecast=False):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10.4, 6.1), sharex=True, layout="constrained")
    for axis, (name, month, season, ylabel, color) in zip(axes.flat, RISK_SPECS):
        if forecast:
            frame = inputs.csv(run / f"{name}_forecast_season_risk.csv")
            frame = frame[frame.window.eq(season)]
            x = frame.year.to_numpy()
        else:
            frame = inputs.csv(run / f"{name}_risk.csv", parse_dates=["time"])
            frame = frame[frame.time.dt.month.eq(month)]
            x = frame.time.dt.year.to_numpy()
        axis.fill_between(x, frame.lower.clip(lower=0).to_numpy(), frame.upper.clip(lower=0).to_numpy(),
                          color=color, alpha=.18, lw=0)
        axis.plot(x, frame["median"].clip(lower=0).to_numpy(), color=color)
        axis.set_title(f"{name}, {season}", loc="left", weight="bold")
        axis.set_ylabel(ylabel)
    for axis in axes[-1]:
        axis.set_xlabel("season year")
    return figure


def _compound(inputs, run):
    import matplotlib.pyplot as plt

    frame = inputs.csv(run / "compound_heat_conditional_risk.csv", parse_dates=["time"])
    figure, axis = plt.subplots(figsize=(8.8, 3.4), layout="constrained")
    axis.fill_between(frame.time.to_numpy(), frame.lower.clip(lower=0).to_numpy(),
                      frame.upper.clip(lower=0).to_numpy(), color="#ba861f", alpha=.2)
    axis.plot(frame.time.to_numpy(), frame["median"].clip(lower=0).to_numpy(), color="#ba861f")
    axis.set(xlabel="forecast season", ylabel="conditional compound probability")
    return figure


def _variant_reports(roots):
    reports = {}
    for root in roots or ():
        root = root.resolve()
        candidates = ([root] if (root / "period_and_endpoint_targets.csv").is_file()
                      else [path.parent for path in root.rglob("period_and_endpoint_targets.csv")])
        for directory in candidates:
            config = bx.load_config(directory / "config.json")
            name = config.get("variant", {}).get("name") or directory.parent.name
            if name in reports and reports[name] != directory:
                if name == "reference":
                    # Separate sensitivity studies intentionally repeat the
                    # same reference fit. One is sufficient for the overlay.
                    continue
                raise ValueError(f"Duplicate sensitivity variant {name!r}.")
            reports[name] = directory
    return reports


def _target_table(inputs, directory):
    frame = inputs.csv(directory / "period_and_endpoint_targets.csv")
    if "quantity" not in frame:
        frame = frame.rename(columns={frame.columns[0]: "quantity"})
    return frame.set_index("quantity")


def _sensitivity(inputs, roots):
    import matplotlib.pyplot as plt

    reports = _variant_reports(roots)
    if len(reports) < 2:
        raise ValueError("Sensitivity figure requires at least two completed variants.")
    variants = list(reports)
    palette = dict(zip(variants, bx.PUBLICATION_COLORS * 3))
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 7.4), layout="constrained")
    for axis, suffix, label in ((axes[0, 0], ".level.change", "late–early level change / °C"),
                                (axes[0, 1], ".slope.change", "recent–early rate / °C per decade")):
        x = np.arange(len(SERIES))
        offsets = np.linspace(-.28, .28, len(variants))
        for offset, variant in zip(offsets, variants):
            table = _target_table(inputs, reports[variant])
            rows = table.loc[[name + suffix for name in SERIES]]
            median = rows["median"].to_numpy()
            axis.errorbar(x + offset, median,
                          yerr=[median - rows.lower.to_numpy(), rows.upper.to_numpy() - median],
                          fmt="o", ms=3.5, capsize=1.5, color=palette[variant], label=variant)
        axis.axhline(0, color=INK, ls="--", lw=.7)
        axis.set(xticks=x, xticklabels=SERIES, ylabel=label)
    risk_names = ("TXx", "TNx", "TXn", "TNn")
    x = np.arange(len(risk_names))
    offsets = np.linspace(-.28, .28, len(variants))
    for offset, variant in zip(offsets, variants):
        medians, lower, upper = [], [], []
        for name in risk_names:
            month = 6 if name.endswith("x") else 12
            path = inputs.csv(reports[variant] / f"{name}_risk.csv", parse_dates=["time"])
            row = path[path.time.dt.month.eq(month)].iloc[-1]
            medians.append(row["median"]); lower.append(row.lower); upper.append(row.upper)
        medians, lower, upper = map(np.asarray, (medians, lower, upper))
        axes[1, 0].errorbar(x + offset, medians, yerr=[medians - lower, upper - medians],
                            fmt="o", ms=3.5, capsize=1.5, color=palette[variant])
    axes[1, 0].set(xticks=x, xticklabels=risk_names, ylabel="end-of-record event probability")

    components = ("level", "slope", "seasonal", "initial_slope")
    x = np.arange(len(components))
    for offset, variant in zip(offsets, variants):
        frame = inputs.csv(reports[variant] / "shared_shrinkage.csv")
        rows = frame[frame.distribution.eq("posterior")].set_index("component").loc[list(components)]
        axes[1, 1].scatter(x + offset, rows["median"], color=palette[variant], s=20, label=variant)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(xticks=x, xticklabels=["level", "slope", "seasonal", "initial\nslope"],
                   ylabel="posterior shared scale (log axis)")
    axes[0, 0].legend(loc="best", fontsize=7.5)
    for label, axis in zip(("A", "B", "C", "D"), axes.flat):
        axis.set_title(label, loc="left", weight="bold")
    return figure


def _find_validation(path):
    path = path.resolve()
    if (path / "central_coverage.csv").is_file():
        return path
    matches = sorted(path.rglob("central_coverage.csv"))
    if len(matches) != 1:
        raise ValueError(f"Expected one validation result below {path}; found {len(matches)}.")
    return matches[0].parent


def _validation(inputs, path):
    import matplotlib.pyplot as plt

    path = _find_validation(path)
    coverage = inputs.csv(path / "central_coverage.csv")
    pits = inputs.csv(path / "held_out_pit.csv")
    tails = inputs.csv(path / "directional_tails.csv")
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), layout="constrained")
    rows = coverage[(coverage.stratum == "overall") & np.isclose(coverage.nominal, .95)].set_index("channel").loc[list(SERIES)]
    axes[0].scatter(np.arange(6), rows.empirical_coverage, color=[_series_color(name) for name in SERIES])
    axes[0].axhline(.95, color=INK, ls="--", lw=.8)
    axes[0].set(xticks=np.arange(6), xticklabels=SERIES, ylim=(0, 1.03), ylabel="held-out 95% coverage")
    for name, group in pits.groupby("channel", sort=False):
        values = np.sort(group.pit.to_numpy())
        axes[1].step(values, np.arange(1, len(values) + 1) / len(values), where="post",
                     color=_series_color(name), alpha=.85, label=name)
    axes[1].plot([0, 1], [0, 1], color=INK, ls="--", lw=.8)
    axes[1].set(xlabel="held-out PIT", ylabel="empirical CDF", xlim=(0, 1), ylim=(0, 1))
    axes[1].legend(ncol=2, fontsize=7)
    overall = tails[(tails.stratum == "overall") &
                    (((tails.side == "upper") & np.isclose(tails.nominal, .99)) |
                     ((tails.side == "lower") & np.isclose(tails.nominal, .01)))]
    for side, marker in (("lower", "v"), ("upper", "^")):
        rows = overall[overall.side.eq(side)].set_index("channel").reindex(SERIES)
        axes[2].scatter(np.arange(6), rows.empirical_event_rate, marker=marker,
                        color=[_series_color(name) for name in SERIES], label=f"{side} tail")
    axes[2].axhline(.01, color=INK, ls="--", lw=.8, label="nominal 1%")
    axes[2].set(xticks=np.arange(6), xticklabels=SERIES, ylabel="held-out directional event rate")
    axes[2].legend(fontsize=7)
    for label, axis in zip(("A", "B", "C"), axes):
        axis.set_title(label, loc="left", weight="bold")
    return figure


def _block_comparison(inputs, path):
    import matplotlib.pyplot as plt

    path = path.resolve()
    comparison = path / "comparison" if (path / "comparison").is_dir() else path
    paired = inputs.csv(comparison / "paired_summary.csv")
    calibration = inputs.csv(comparison / "calibration_by_case.csv")
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), layout="constrained")
    crps = paired[paired.score.eq("crps")].set_index("channel").reindex(SERIES)
    axes[0].bar(np.arange(6), crps.seasonal_minus_monthly, color=[_series_color(name) for name in SERIES])
    axes[0].axhline(0, color=INK, lw=.8)
    axes[0].set(xticks=np.arange(6), xticklabels=SERIES,
                ylabel="seasonal minus monthly CRPS / °C")
    for model, group in calibration.groupby("model", sort=False):
        values = np.sort(group.pit.to_numpy())
        axes[1].step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=model)
    axes[1].plot([0, 1], [0, 1], color=INK, ls="--", lw=.8)
    axes[1].set(xlabel="held-out PIT", ylabel="empirical CDF", xlim=(0, 1), ylim=(0, 1))
    axes[1].legend()
    axes[0].set_title("A  Matched forecast score", loc="left", weight="bold")
    axes[1].set_title("B  Matched calibration", loc="left", weight="bold")
    return figure


def _pre2019(inputs, roots):
    import matplotlib.pyplot as plt

    reports = _variant_reports(roots)
    # A direct final-style pre-2019 report has no sensitivity target file.
    # Prefer it for the reference when both it and shorter sensitivity fits
    # are supplied.
    for root in roots:
        try:
            directory = _report_directory(root)
        except ValueError:
            continue
        if (directory / "TXx_forecast_risk_39p7.csv").is_file():
            config = inputs.config(directory / "config.json")
            reports[config.get("variant", {}).get("name", "reference")] = directory
    if "reference" not in reports:
        reference = next(iter(reports))
    else:
        reference = "reference"
    figure, axes = plt.subplots(1, 2, figsize=(9.8, 3.9), layout="constrained")
    forecast = inputs.csv(reports[reference] / "forecast.csv", parse_dates=["time"])
    row = forecast[(forecast.channel == "TXx") & (forecast.time.dt.month == 6)].iloc[0]
    axes[0].errorbar(0, row["median"], yerr=[[row["median"] - row.lower], [row.upper - row["median"]]],
                     fmt="o", color=TX, capsize=4, label="pre-event prediction")
    axes[0].scatter(0, 39.7, marker="*", s=90, color="#ba861f", label="observed 39.7°C")
    axes[0].set(xticks=[0], xticklabels=["JJA 2019 TXx"], ylabel="seasonal maximum / °C")
    axes[0].legend(fontsize=8)
    variants = list(reports)
    for y, variant in enumerate(variants):
        row = inputs.csv(reports[variant] / "TXx_forecast_risk_39p7.csv").iloc[0]
        axes[1].errorbar(row["median"], y,
                         xerr=[[row["median"] - row.lower], [row.upper - row["median"]]],
                         fmt="o", color=bx.PUBLICATION_COLORS[y % len(bx.PUBLICATION_COLORS)], capsize=3)
    axes[1].set(yticks=np.arange(len(variants)), yticklabels=variants,
                xlabel=r"prospective $P(\mathrm{TXx}>39.7^\circ\mathrm{C})$")
    axes[0].set_title("A  Predictive temperature", loc="left", weight="bold")
    axes[1].set_title("B  Record-event probability", loc="left", weight="bold")
    return figure


def build(run, output, *, formats=("png", "pdf"), dpi=220,
          allow_unconverged=False, sensitivity=(), validation=None,
          block_comparison=None, pre2019=()):
    import matplotlib.pyplot as plt

    run = _report_directory(Path(run))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs, figures = Inputs(), []
    config = inputs.config(run / "config.json")
    convergence = inputs.config(run / "convergence.json")
    if convergence.get("status") != "passed_numerical_checks" and not allow_unconverged:
        raise RuntimeError("The final report has not passed convergence.json; use --allow-unconverged only for figure development.")
    with bx.publication_style(style="manuscript", dpi=dpi,
                              overrides={"font.size": 9, "axes.labelsize": 9,
                                         "axes.titlesize": 10, "xtick.labelsize": 8,
                                         "ytick.labelsize": 8, "legend.fontsize": 8}):
        data = bx.load_uccle_multiseries(**config["data"])
        figure, cycle, smooth = exploratory_structure(
            data, reference=config["contrasts"]["reference"], comparison=config["contrasts"]["comparison"])
        cycle.to_csv(output / "exploratory_seasonal_cycles.csv", index=False)
        smooth.to_csv(output / "exploratory_seasonal_smooths.csv", index=False)
        _save(figure, output, "exploratory_seasonal_blocks", formats, dpi, figures)
        _save(_band_panels(inputs, run, "level", "latent level / °C"), output,
              "seasonal_levels", formats, dpi, figures)
        _save(_band_panels(inputs, run, "slope_C_per_decade", "slope / °C per decade", zero=True), output,
              "seasonal_slopes", formats, dpi, figures)
        _save(_scale_panels(inputs, run), output, "seasonal_scales", formats, dpi, figures)
        _save(_pit_panels(inputs, run), output, "seasonal_pit_qq", formats, dpi, figures)
        change, rates = _contrast_figures(inputs, run)
        _save(change, output, "seasonal_change_contrasts", formats, dpi, figures)
        _save(rates, output, "seasonal_rate_summary", formats, dpi, figures)
        _save(_shared_shrinkage(inputs, run), output, "shared_shrinkage", formats, dpi, figures)
        _save(_copula(inputs, run), output, "copula_correlations", formats, dpi, figures)
        _save(_risk_panels(inputs, run), output, "seasonal_risks", formats, dpi, figures)
        _save(_risk_panels(inputs, run, forecast=True), output, "seasonal_risk_forecasts", formats, dpi, figures)
        _save(_compound(inputs, run), output, "compound_heat_conditional_risk", formats, dpi, figures)
        if sensitivity:
            _save(_sensitivity(inputs, sensitivity), output, "prior_sensitivity", formats, dpi, figures)
        if validation is not None:
            _save(_validation(inputs, validation), output, "heldout_predictive_validation", formats, dpi, figures)
        if block_comparison is not None:
            _save(_block_comparison(inputs, block_comparison), output, "monthly_block_sensitivity", formats, dpi, figures)
        if pre2019:
            _save(_pre2019(inputs, pre2019), output, "pre2019_record_event", formats, dpi, figures)
    manifest = {
        "bucex_version": bx.__version__, "run": str(run),
        "convergence_status": convergence.get("status"),
        "development_override": bool(allow_unconverged),
        "formats": list(formats), "dpi": dpi, "figures": figures,
        "sources": [{"path": path, "sha256": digest} for path, digest in sorted(inputs.sources.items())],
        "interpretation": [
            "All intervals are read from the saved report tables; this command performs no posterior fitting.",
            "The exploratory smooth is descriptive and is not a state estimate.",
            "In-sample PIT panels are descriptive; held-out calibration is shown only when a validation run is supplied.",
            "The pre-2019 panel is generated only from a fit ending before JJA 2019.",
        ],
    }
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="Timestamped final joint report from research.seasonal.fit.")
    parser.add_argument("--output", type=Path, default=Path("results/serra_187_manuscript_figures"))
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=("png", "pdf"))
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--allow-unconverged", action="store_true",
                        help="Development only: render while retaining the failed status in the manifest.")
    parser.add_argument("--sensitivity", type=Path, nargs="+", default=(),
                        help="One or more completed sensitivity/adequacy run roots.")
    parser.add_argument("--validation", type=Path, help="Completed expanding-window validation root or joint directory.")
    parser.add_argument("--block-comparison", type=Path, help="Completed monthly-versus-seasonal comparison root.")
    parser.add_argument("--pre2019", type=Path, nargs="+", default=(),
                        help="Completed pre-2019 reference and optional sensitivity run roots.")
    args = parser.parse_args()
    print(build(args.run, args.output, formats=args.formats, dpi=args.dpi,
                allow_unconverged=args.allow_unconverged,
                sensitivity=args.sensitivity, validation=args.validation,
                block_comparison=args.block_comparison, pre2019=args.pre2019))


if __name__ == "__main__":
    main()
