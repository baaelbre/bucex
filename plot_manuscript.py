"""Matplotlib figures from BUCEX 1.9.8.3/1.9.8.4 report tables.

Run this file from your BUCEX project root. It never fits a model.
See plotting_config.example.json and README.txt for paths and commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm, halfnorm, invgamma
from scipy.optimize import brentq
from scipy.special import gammaln

CH = ["TXm", "TXx", "TXn", "TNm", "TNx", "TNn"]
BLUE, RED, TEAL, GOLD, GRAY = "#24658a", "#a44839", "#3e7c78", "#bf8c32", "#707b85"
SEASONS = {12: "DJF", 3: "MAM", 6: "JJA", 9: "SON"}
GRID_VARIANTS = ["reference", "half_level", "double_level", "half_slope",
                 "double_slope", "half_level_half_slope", "half_level_double_slope",
                 "double_level_half_slope", "double_level_double_slope"]
FIGURES = {}


class MissingInput(Exception):
    """An optional figure cannot be made from the available files."""


def figure(name):
    def register(func):
        FIGURES[name] = func
        return func
    return register


def required(frame, columns, source):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")


def report_dir(path, marker="config.json"):
    """Accept the report itself or the task directory immediately above it."""
    path = Path(path).expanduser().resolve()
    for candidate in (path / "report", path):
        if (candidate / marker).is_file():
            return candidate
    raise MissingInput(f"No {marker} in {path} or its report subdirectory")


class Inputs:
    def __init__(self, settings):
        self.settings = settings
        self.sources, self.outputs, self.notes = {}, [], []
        self.prior_comparisons = []
        self._cases, self._seasonal, self._config = None, None, None
        self.root = Path(settings.get("results_root", "results/serra_1983")).expanduser().resolve()
        tier = settings.get("tier", "paper")
        if (self.root / tier).is_dir():
            self.root = self.root / tier
        self.output = Path(settings.get("output", "manuscript/figures")).expanduser().resolve()
        self.formats = settings.get("formats", ["pdf", "png"])
        self.dpi = int(settings.get("dpi", 220))

    def csv(self, path):
        path = Path(path).resolve()
        if not path.is_file():
            raise MissingInput(f"Missing {path}")
        self.record(path)
        frame = pd.read_csv(path)
        if "credible_interval" in frame and not np.allclose(frame.credible_interval.dropna(), .95):
            raise ValueError(f"{path}: expected central 95% intervals")
        return frame

    def record(self, path):
        path = Path(path).resolve()
        if str(path) not in self.sources:
            self.sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    def read(self, name, root=None):
        if root is None:
            _ = self.config
        return self.csv((self.reference if root is None else root) / name)

    @property
    def reference(self):
        return report_dir(self.settings.get("reference_report", self.root / "final_posterior_reference"))

    @property
    def config(self):
        if self._config is None:
            p = self.reference / "config.json"
            self.record(p)
            self._config = json.loads(p.read_text(encoding="utf-8"))
            if not np.isclose(self._config.get("credible_interval", .95), .95):
                raise ValueError("Reference report does not contain 95% intervals")
            if self._config.get("model", {}).get("period") != 4:
                raise ValueError("These manuscript figures require the seasonal (p=4) analysis")
        return self._config

    @property
    def pre2019(self):
        return report_dir(self.settings.get("pre2019_report", self.root / "final_pre2019_reference_2019-05"))

    def sensitivity(self, variant):
        if variant == "reference":
            return self.reference
        override = self.settings.get("sensitivity_reports", {}).get(variant)
        return report_dir(override or self.root / f"final_posterior_{variant}")

    def anchors(self, directory=None):
        if directory is None:
            config = self.config
        else:
            p = directory / "config.json"
            self.record(p)
            config = json.loads(p.read_text(encoding="utf-8"))
        prior = config["priors"]
        if prior.get("shared_shrinkage", {}).get("hyperprior") != "half_normal":
            raise ValueError("This prior figure expects pooled half-normal scales")
        a = prior["innovation_sd"]
        return {"level": a["level"], "slope": a["trend"], "seasonal": a["season"]}

    def save(self, fig, name):
        self.output.mkdir(parents=True, exist_ok=True)
        paths = []
        for ext in self.formats:
            p = self.output / f"{name}.{ext}"
            fig.savefig(p, dpi=self.dpi, bbox_inches="tight")
            paths.append(str(p))
        plt.close(fig)
        self.outputs.append({"name": name, "files": paths})

    def seasonal_data(self):
        if self._seasonal is not None:
            return self._seasonal
        if self.settings.get("seasonal_csv"):
            data = self.csv(self.settings["seasonal_csv"])
            time = "time" if "time" in data else data.columns[0]
            data = data.set_index(pd.to_datetime(data[time]))[CH]
        else:
            # The plotting file normally sits in the BUCEX checkout root.
            sys.path.insert(0, str(Path.cwd()))
            try:
                import bucex as bx
            except ImportError as exc:
                raise MissingInput("Run in BUCEX's environment, or supply seasonal_csv") from exc
            options = dict(self.config["data"])
            daily = Path(self.settings.get("daily", options["daily_source"])).expanduser()
            if not daily.is_file():
                raise MissingInput(f"Daily observations not found: {daily}; set --daily or seasonal_csv")
            self.record(daily)
            options["daily_source"] = str(daily)
            data = bx.load_uccle_multiseries(**options)[CH]
        if data.isna().any().any() or not data.index.is_unique:
            raise ValueError("Seasonal observations contain missing or duplicate dates")
        if not data.index.month.isin(list(SEASONS)).all():
            raise ValueError("Seasonal CSV must use seasonal block start dates")
        # Verify that the observations cover exactly the reference-fit time axis.
        expected = pd.DatetimeIndex(pd.to_datetime(self.read("TXm_level.csv").time))
        if not data.index.equals(expected):
            raise ValueError("Seasonal observations do not match the reference report dates")
        self._seasonal = data
        return data

    def validation_report(self, path, origin):
        """Normalize native 1.9.8.3 and 1.9.8.4 exports."""
        path = Path(path).expanduser().resolve()
        candidates = [path / "report/joint", path / "joint", path / "report", path]
        path = next((p for p in candidates if (p / "predictions.csv").exists()
                     or (p / "forecast_paths.csv").exists()), None)
        if path is None:
            raise MissingInput(f"No forecast report under {candidates[-1]}")
        keys = ["channel", "time", "horizon"]
        if (path / "forecast_paths.csv").exists():
            f = self.csv(path / "forecast_paths.csv")
            if "pit" not in f:
                pit = self.csv(path / "held_out_pit.csv")
                f = f.merge(pit[keys + ["pit"]], on=keys, how="left", validate="one_to_one")
        else:
            f = self.csv(path / "predictions.csv").rename(columns={"mean": "predictive_mean"})
            f = f.merge(self.csv(path / "held_out_pit.csv")[keys + ["pit"]],
                        on=keys, how="left", validate="one_to_one")
            f = f.rename(columns={"lower": "q025", "upper": "q975"})
        f["origin_date"] = origin
        return f

    def validation(self):
        if self._cases is not None:
            return self._cases
        historical = self.settings.get("historical_origins", ["1956-11", "1976-11", "1996-11"])
        recent = self.settings.get("recent_origins", ["2015-11", "2020-11"])
        aggregated = None
        if self.settings.get("validation_cases"):
            aggregated = self.csv(self.settings["validation_cases"])
        elif self.settings.get("validation_root"):
            base = Path(self.settings["validation_root"]).expanduser().resolve()
            if base.is_file():
                aggregated = self.csv(base)
            else:
                tier = self.settings.get("validation_tier", self.settings.get("tier", "paper"))
                paths = [base / "validation_cases.csv",
                         base / "validation_report_horizon_reference/validation_cases.csv",
                         base / tier / "validation_report_horizon_reference/validation_cases.csv"]
                existing = [p for p in paths if p.exists()]
                if existing:
                    aggregated = self.csv(existing[0])
        if aggregated is not None:
            required(aggregated, ["origin_date", "channel", "time"], "validation cases")
            if "variant" in aggregated:
                aggregated = aggregated[aggregated.variant.eq("reference")].copy()
        parts = []
        explicit = self.settings.get("validation_reports", {})
        for origin in dict.fromkeys(historical + recent):
            # Explicit mapping wins. Recent 1.9.8.3 refits match the revised manuscript.
            path = explicit.get(origin)
            final = self.root / f"final_forecast_reference_{origin}"
            if path is None and origin in recent and final.is_dir():
                path = final
            if path is not None:
                f = self.validation_report(path, origin)
            elif aggregated is not None and origin in set(aggregated.origin_date):
                f = aggregated[aggregated.origin_date.eq(origin)].copy()
            else:
                base = Path(self.settings.get("validation_root", self.root)).expanduser()
                tier = self.settings.get("validation_tier", self.settings.get("tier", "paper"))
                candidates = [base / f"validation1984_reference_{origin}",
                              base / tier / f"validation1984_reference_{origin}", final]
                path = next((p for p in candidates if p.is_dir()), None)
                if path is None:
                    continue
                f = self.validation_report(path, origin)
            required(f, ["channel", "time", "horizon", "observed", "predictive_mean",
                         "pit", "q025", "q975"], f"validation {origin}")
            # Only held-out observations are validation cases; unobserved futures are excluded.
            f = f[f.observed.notna()].copy()
            if f[["predictive_mean", "pit", "q025", "q975"]].isna().any().any():
                raise ValueError(f"{origin}: incomplete predictions or PITs for observed cases")
            t = pd.to_datetime(f.time)
            f["time"] = t.dt.strftime("%Y-%m-%d")
            f["season"] = t.dt.month.map(SEASONS)
            f["meteorological_year"] = t.dt.year + t.dt.month.eq(12)
            f["lead_year"] = (f.horizon.astype(int) - 1) // 4 + 1
            if f.season.isna().any() or not f.pit.between(0, 1).all():
                raise ValueError(f"{origin}: invalid seasons or PIT values")
            if (f.q025 > f.q975).any() or f.duplicated(["channel", "time"]).any():
                raise ValueError(f"{origin}: reversed intervals or duplicate cases")
            parts.append(f)
        if not parts:
            raise MissingInput("No validation cases; set validation_root or validation_cases")
        self._cases = pd.concat(parts, ignore_index=True)
        return self._cases


def yr(d):
    t = pd.to_datetime(d.time)
    return t.dt.year + (t.dt.month - 1) / 12


def panel(size=(7, 4.8), samey=False):
    return plt.subplots(2, 3, figsize=size, layout="constrained", sharex=True, sharey=samey)


def historyaxis(ax):
    ax.set_xlim(1892, 2027)
    ax.set_xticks([1900, 1940, 1980, 2020])
    ax.tick_params(axis="x", labelrotation=45)
    ax.yaxis.set_major_locator(MaxNLocator(4))


def bands(ax, d, col, scale=1, label=None):
    required(d, ["time", "mean", "lower", "upper"], "trajectory table")
    x = yr(d)
    ax.fill_between(x, scale*d.lower, scale*d.upper, color=col, alpha=.18, lw=0)
    ax.plot(x, scale*d["mean"], color=col, lw=1.4, label=label)


def legend(fig, ax, ncol=2, fontsize=8):
    # 'outside lower center' requires Matplotlib >=3.7. Use a 3.6-compatible layout.
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(.5, -.015), ncol=ncol, fontsize=fontsize)


def differences(ctx, c, suffix):
    ref = ctx.read(f"{c}_{suffix}.csv").set_index("time")
    values = []
    variants = ctx.settings.get("grid_variants", GRID_VARIANTS)
    if len(variants) < 2 or variants[0] != "reference":
        raise ValueError("grid_variants must begin with reference and include at least one comparison")
    for v in variants:
        report = ctx.sensitivity(v)
        other = ctx.read(f"{c}_{suffix}.csv", report).set_index("time")
        if not ref.index.is_unique or not other.index.is_unique or set(ref.index) != set(other.index):
            raise ValueError(f"{v}, {c}: sensitivity dates differ from the reference")
        values.append(other.reindex(ref.index)["mean"].to_numpy() - ref["mean"].to_numpy())
    return ref.reset_index(), np.asarray(values)


def check_origins(cases, origins):
    for o in origins:
        for c in CH:
            if cases[(cases.origin_date == o) & (cases.channel == c)].empty:
                raise MissingInput(f"Missing validation cases for {o}, {c}")


@figure('trajectories')
def plot_trajectories(ctx):
    read, save = (ctx.read, ctx.save)
    for suffix, name, ylabel in [('level', 'levels', 'Latent level / °C'), ('slope_C_per_decade', 'rates', 'Rate / °C per decade')]:
        fig, axes = panel()
        for c, ax in zip(CH, axes.flat):
            bands(ax, read(c + '_' + suffix + '.csv'), BLUE if c.startswith('TX') else RED)
            ax.set_title(c, loc='left', weight='bold')
            historyaxis(ax)
            if suffix != 'level':
                ax.axhline(0, color=GRAY, ls='--', lw=0.7)
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
        for ax in axes[1]:
            ax.set_xlabel('Year')
        save(fig, name)

@figure('marginal')
def plot_marginal(ctx):
    read, save = (ctx.read, ctx.save)
    fig, axes = panel((7, 4.6), True)
    for c, ax in zip(CH, axes.flat):
        u = read(c + '_smoothed_pit.csv').pit.values
        n = len(u)
        z = np.sort(norm.ppf(np.clip(u, 1e-10, 1 - 1e-10)))
        q = norm.ppf((np.arange(n) + 0.5) / n)
        ax.scatter(q, z, s=6, color=BLUE if c.startswith('TX') else RED, alpha=0.7, linewidths=0)
        ax.plot([-3.6, 3.6], [-3.6, 3.6], ls='--', color=GRAY, lw=0.8)
        ax.set_title(c, loc='left', weight='bold')
        ax.set(xlim=(-3.5, 3.5), ylim=(-3.8, 3.8))
    for ax in axes[:, 0]:
        ax.set_ylabel('Smoothed normal score')
    for ax in axes[1]:
        ax.set_xlabel('Standard-normal quantile')
    save(fig, 'normal_score_qq')

@figure('risk')
def plot_risk(ctx):
    read, save = (ctx.read, ctx.save)
    riskperiod = read('risk_period_summaries.csv')
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.0), layout='constrained')
    for ax, c, phase, title in zip(axes.flat, ['TXx', 'TNx', 'TXn', 'TNn'], [6, 6, 12, 12], ['JJA TXx > 35 °C', 'JJA TNx > 25 °C', 'DJF TXn < 0 °C', 'DJF TNn < −10 °C']):
        d = read(c + '_risk.csv')
        d = d[pd.to_datetime(d.time).dt.month == phase]
        col = BLUE if c.startswith('TX') else RED
        bands(ax, d, col, 100)
        p = riskperiod[(riskperiod.channel == c) & (riskperiod.phase == phase)]
        for period, x in [('reference', 1907), ('comparison', 2011)]:
            row = p[p.period == period].iloc[0]
            ax.errorbar(x, 100 * row['mean'], yerr=[[100 * (row['mean'] - row.lower)], [100 * (row.upper - row['mean'])]], fmt='D', ms=4, color=col, ecolor='#253441', capsize=3, zorder=5)
        ax.set_title(title, loc='left')
        ax.set_ylabel('Seasonal probability / %')
        historyaxis(ax)
        ax.set_xlabel('Year')
    save(fig, 'historical_risks')

@figure('record2019')
def plot_record2019(ctx):
    read, save = (ctx.read, ctx.save)
    pre = ctx.pre2019
    p = read('pre2019_observed_predictions.csv', pre).set_index('channel').loc[CH]
    q = read('pre2019_TXx_risk_curve.csv', pre)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7, 3.2), layout='constrained', gridspec_kw={'width_ratios': [1, 1.15]})
    for i, (c, row) in enumerate(p.iterrows()):
        col = BLUE if c.startswith('TX') else RED
        ax.errorbar(i, row['mean'], yerr=[[row['mean'] - row.lower], [row.upper - row['mean']]], fmt='o', ms=4, capsize=3, color=col)
        ax.scatter(i, row.observed, marker='D', s=26, color='#252c32', zorder=4)
    ax.set_xticks(range(6), CH, rotation=45)
    ax.set_ylabel('Temperature / °C')
    ax.set_title('A  JJA 2019 predictions', loc='left')
    ax.plot([], [], 'o', color=BLUE, label='Predictive mean + 95% PI')
    ax.plot([], [], 'D', color='#252c32', label='Observed')
    legend(fig, ax, ncol=2, fontsize=7)
    bx.fill_between(q.threshold, 100 * q.lower, 100 * q.upper, color=BLUE, alpha=0.18, lw=0)
    bx.plot(q.threshold, 100 * q['mean'], color=BLUE, lw=1.3)
    for threshold, col in [(35, GOLD), (36.6, TEAL), (39.7, RED)]:
        row = q[np.isclose(q.threshold, threshold)].iloc[0]
        bx.scatter(threshold, 100 * row['mean'], color=col, s=26, zorder=3, label=f"{threshold:g} °C: {100 * row['mean']:.3g}%")
        bx.axvline(threshold, color=col, lw=0.8, ls=':')
    bx.set(yscale='log', ylim=(0.0001, 100), xlim=(30, 41), xlabel='TXx threshold / °C', ylabel='Exceedance probability / %')
    bx.set_title('B  Predicted upper tail', loc='left')
    bx.legend(fontsize=7, loc='lower left')
    save(fig, 'summer_2019')

@figure('seasonal')
def plot_seasonal(ctx):
    read, save = (ctx.read, ctx.save)
    fig, axes = panel((7, 4.6), True)
    sc = {12: BLUE, 3: RED, 6: GOLD, 9: TEAL}
    for c, ax in zip(CH, axes.flat):
        d = read(c + '_seasonal_change.csv')
        months = pd.to_datetime(d.time).dt.month
        for m, s in [(12, 'DJF'), (3, 'MAM'), (6, 'JJA'), (9, 'SON')]:
            b = d[months == m]
            bands(ax, b, sc[m], label=s)
        ax.set_title(c, loc='left', weight='bold')
        historyaxis(ax)
        ax.axhline(0, color=GRAY, ls='--', lw=0.7)
    for ax in axes[:, 0]:
        ax.set_ylabel('Seasonal change / °C')
    for ax in axes[1]:
        ax.set_xlabel('Year')
    legend(fig, axes[0, 0], ncol=4)
    save(fig, 'seasonal_evolution')

@figure('learning')
def plot_learning(ctx):
    read, save = (ctx.read, ctx.save)
    tr = read('shared_shrinkage_traces.csv.gz')
    fig, axes = plt.subplots(2, 3, figsize=(7, 4.9), layout='constrained')
    gains = {'level': np.sqrt(120), 'slope': np.sqrt(120 * 119 * 239 / 6), 'seasonal': np.sqrt(60)}
    anchors = ctx.anchors()
    for j, (comp, title) in enumerate([('level', 'Level'), ('slope', 'Slope'), ('seasonal', 'Seasonal')]):
        A = anchors[comp]
        x = np.linspace(0, 3.4, 200)
        axes[0, j].plot(x, halfnorm.pdf(x), color=GRAY, label='Half-normal prior')
        axes[0, j].hist(tr['shrinkage.shared.' + comp] / A, bins=45, density=True, color=BLUE, alpha=0.6, label='Posterior')
        axes[0, j].set(xlim=(0, 3.2), xlabel='Shared scale $\\tau/A$', title=title)
        ax = axes[1, j]
        for i, c in enumerate(CH):
            d = read(c + '_prior_posterior.csv')
            r = d[(d.component == comp) & (d.distribution == 'posterior') & (d.scale == 'SD')].iloc[0]
            g = gains[comp]
            ax.errorbar(g * r['mean'], 5 - i, xerr=[[g * (r['mean'] - r.lower)], [g * (r.upper - r['mean'])]], fmt='o', ms=3, capsize=2, color=BLUE if c.startswith('TX') else RED)
        ax.set_yticks(range(6), CH[::-1])
        ax.set_xlabel('30-year SD / °C')
        ax.set_xlim(left=0)
    axes[0, 0].set_ylabel('Density')
    legend(fig, axes[0, 0], ncol=2)
    save(fig, 'structural_evolution')

@figure('sensitivity')
def plot_sensitivity(ctx):
    read, save = (ctx.read, ctx.save)
    for suffix, name, ylabel in [('level', 'sensitivity_levels', 'Level difference / °C'), ('slope_C_per_decade', 'sensitivity_rates', 'Rate difference / °C per decade')]:
        fig, axes = panel((7, 4.8), True)
        for c, ax in zip(CH, axes.flat):
            ref, a = differences(ctx, c, suffix)
            x = yr(ref)
            col = BLUE if c.startswith('TX') else RED
            ax.fill_between(x, a.min(axis=0), a.max(axis=0), color=col, alpha=0.24, lw=0)
            for curve in a[1:]:
                ax.plot(x, curve, color=col, alpha=0.34, lw=0.55)
            ax.axhline(0, color=GRAY, lw=0.7, ls='--')
            ax.set_title(c, loc='left', weight='bold')
            historyaxis(ax)
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
        for ax in axes[1]:
            ax.set_xlabel('Year')
        save(fig, name)

@figure('seasonal_sensitivity')
def plot_seasonal_sensitivity(ctx):
    read, save = (ctx.read, ctx.save)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7, 3.5), layout='constrained', gridspec_kw={'width_ratios': [1, 1.15]})
    for i, (v, A, col) in enumerate([('seasonal_5e2', 0.05, TEAL), ('reference', 0.1, BLUE), ('double_seasonal', 0.2, RED)]):
        d = read('shared_shrinkage.csv', ctx.sensitivity(v))
        actual = ctx.anchors(ctx.sensitivity(v))['seasonal']
        if not np.isclose(actual, A):
            raise ValueError(f'{v}: expected seasonal calibration {A}, found {actual}')
        for dist, y, clr in [('prior', i - 0.1, GRAY), ('posterior', i + 0.1, col)]:
            r = d[(d.component == 'seasonal') & (d.distribution == dist)].iloc[0]
            ax.errorbar(r['mean'], y, xerr=[[r['mean'] - r.lower], [r.upper - r['mean']]], fmt='o', ms=4, capsize=2, color=clr)
        for j, c in enumerate(CH):
            d = read(c + '_prior_posterior.csv', ctx.sensitivity(v))
            r = d[(d.component == 'seasonal') & (d.distribution == 'posterior') & (d.scale == 'SD')].iloc[0]
            bx.errorbar(r['mean'], 5 - j + (i - 1) * 0.18, xerr=[[r['mean'] - r.lower], [r.upper - r['mean']]], fmt='o', ms=3, capsize=2, color=col, label=f'$A_\\gamma={A:g}$' if j == 0 else None)
    ax.set_yticks([0, 1, 2], ['0.05', '0.10', '0.20'])
    ax.set_ylabel('Calibration $A_\\gamma$')
    ax.set_xlabel('Shared seasonal scale $\\tau_\\gamma$')
    ax.set_title('A  Prior and posterior', loc='left')
    ax.set_xlim(left=0)
    bx.set_yticks(range(6), CH[::-1])
    bx.set_xlabel('Seasonal innovation SD / °C')
    bx.set_title('B  Response-specific posteriors', loc='left')
    bx.set_xlim(left=0)
    bx.legend(fontsize=7, loc='lower right')
    ax.plot([], [], 'o', color=GRAY, label='Prior')
    ax.plot([], [], 'o', color=BLUE, label='Posterior')
    ax.legend(fontsize=8)
    save(fig, 'seasonal_sensitivity')

# Additional prior checks use exported posterior means and quantiles. The short
# refits do not contain parameter traces, so no posterior density is fabricated.
PRIOR_COLORS = [TEAL, BLUE, RED]
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]


def prior_cases(ctx, variants):
    cases = []
    for variant, color in zip(variants, PRIOR_COLORS):
        directory = ctx.sensitivity(variant)
        path = directory / "config.json"
        ctx.record(path)
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if cfg["model"]["period"] != 4 or not np.isclose(cfg["credible_interval"], .95):
            raise ValueError(f"{variant}: seasonal fits with 95% intervals are required")
        for key in ("start", "end", "frequency", "exclude_months"):
            if cfg["data"].get(key) != ctx.config["data"].get(key):
                raise ValueError(f"{variant}: data window differs from the reference")
        parameters = ctx.read("parameters.csv", directory)
        parameters = parameters.set_index(parameters.columns[0])
        required(parameters, ["mean", "lower", "upper"], directory / "parameters.csv")
        diagnostics = ctx.read("mcmc.csv", directory).set_index("parameter")
        cases.append(dict(variant=variant, color=color, directory=directory,
                          config=cfg, parameters=parameters, diagnostics=diagnostics))
    return cases


def transformed_summary(row, multiplier=1.):
    bounds = sorted([multiplier * float(row["lower"]), multiplier * float(row["upper"])])
    result = dict(mean=multiplier * float(row["mean"]), lower=bounds[0], upper=bounds[1])
    if not np.isfinite(list(result.values())).all():
        raise ValueError("A prior-sensitivity summary is not finite")
    return result


def interval_mark(ax, summary, y, color):
    # hlines also works if the posterior mean lies outside a central interval.
    ax.hlines(y, summary["lower"], summary["upper"], color=color, lw=1.6)
    ax.plot(summary["mean"], y, "o", color=color, ms=3.8)


def comparison_row(ctx, case, quantity, channel, summary, prior, unit,
                   parameter=None, season=None):
    row = dict(variant=case["variant"], quantity=quantity, channel=channel,
               season=season, unit=unit, prior_mean=prior["mean"],
               prior_lower=prior["lower"], prior_upper=prior["upper"],
               posterior_mean=summary["mean"], posterior_lower=summary["lower"],
               posterior_upper=summary["upper"], credible_interval=.95)
    if parameter in case["diagnostics"].index:
        d = case["diagnostics"].loc[parameter]
        for name in ("rhat", "ess_bulk", "ess_tail"):
            row[name] = float(d[name])
    ctx.prior_comparisons.append(row)


def normal_prior(sd, mean=0.):
    return dict(mean=mean, lower=norm.ppf(.025, loc=mean, scale=sd),
                upper=norm.ppf(.975, loc=mean, scale=sd))


def scalar_posterior_panel(ctx, ax, cases, channels, parameter, priors,
                           quantity, unit, multiplier=None):
    for i, case in enumerate(cases):
        for j, channel in enumerate(channels):
            key = parameter(channel)
            scale = 1. if multiplier is None else multiplier(channel, case)
            s = transformed_summary(case["parameters"].loc[key], scale)
            interval_mark(ax, s, len(channels)-1-j + (.19 * (1-i)), case["color"])
            comparison_row(ctx, case, quantity, channel, s, priors[i], unit, key)
    ax.set_yticks(range(len(channels)), channels[::-1])
    ax.set_ylim(-.55, len(channels)-.45)
    ax.set_xlabel(unit)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="y", visible=False)
    ax.set_title("Posterior means and 95% intervals", loc="left", fontsize=10)


def initial_seasonal_prior(cfg, step):
    """Marginal density at an observed step, including new seasonal innovations.

    gamma_t = gamma_initial[k(t)] + s_gamma * W_t. For the HN--normal
    hierarchy, s_gamma = A * |Z1| * Z2. Conditional on Z1,Z2, gamma_t is
    normal with variance a^2(1-1/p) + Var(W_t)*A^2*Z1^2*Z2^2.
    Gaussian quadrature integrates the two independent standard normals.
    """
    p, pr = cfg["model"]["period"], cfg["priors"]
    if pr["seasonal_initial_basis"] != "orthonormal":
        raise ValueError("Starting-cycle prior requires orthonormal seasonal contrasts")
    if pr["shared_shrinkage"]["hyperprior"] != "half_normal":
        raise ValueError("Starting-cycle prior requires half-normal shared scales")
    impulse = [1.]
    for k in range(1, step):
        impulse.append(-sum(impulse[max(0, k-p+1):k]))
    gain = np.sum(np.square(impulse))
    nodes, weights = np.polynomial.hermite.hermgauss(24)
    z = np.sqrt(2.) * nodes
    w = (weights[:, None] * weights[None, :] / np.pi).ravel()
    s = (pr["innovation_sd"]["season"] * z[:, None] * z[None, :]).ravel()
    sd = np.sqrt(pr["seasonal_initial_sd"]**2 * (1-1/p) + gain*s*s)
    def pdf(x):
        return norm.pdf(np.asarray(x)[..., None], scale=sd) @ w
    def cdf(x):
        return float(norm.cdf(x, scale=sd) @ w)
    hi = brentq(lambda x: cdf(x)-.975, 0., 10*max(sd))
    return pdf, dict(mean=0., lower=-hi, upper=hi)


def seasonal_scale_prior(cfg):
    """Prior on sigma_k = sqrt(V)*exp(d_k), V~IG(a,b).

    The marginal d_k SD is coordinate_sd*sqrt(1-1/p), not coordinate_sd.
    Integrate this normal log-scale contrast by Gaussian quadrature.
    """
    a, b = cfg["priors"]["observation_variance"]
    p = cfg["model"]["period"]
    sd = cfg["model"]["scale_prior_sd"] * np.sqrt(1-1/p)
    nodes, weights = np.polynomial.hermite.hermgauss(48)
    d, w = np.sqrt(2.) * sd * nodes, weights / np.sqrt(np.pi)
    def pdf(x):
        x = np.asarray(x)[..., None]
        return (2*x*np.exp(-2*d) * invgamma.pdf(x*x*np.exp(-2*d), a=a, scale=b)) @ w
    def cdf(x):
        return float(invgamma.cdf(x*x*np.exp(-2*d), a=a, scale=b) @ w)
    hi = max(1., np.sqrt(invgamma.ppf(.999, a=a, scale=b))*np.exp(3*sd))
    while cdf(hi) < .975:
        hi *= 2
    limits = [brentq(lambda x: cdf(x)-u, 1e-10, hi) for u in (.025, .975)]
    mean = np.sqrt(b)*np.exp(gammaln(a-.5)-gammaln(a)+.5*sd*sd)
    return pdf, dict(mean=mean, lower=limits[0], upper=limits[1])


def seasonal_comparison(ctx, cases, name, initial_cycle):
    fig = plt.figure(figsize=(8.3, 6.6), layout="constrained")
    grid = fig.add_gridspec(3, 3, height_ratios=[.9, 1., 1.])
    prior_ax = fig.add_subplot(grid[0, :])
    axes = [fig.add_subplot(grid[1+i//3, i%3]) for i in range(6)]
    all_priors = []
    for case in cases:
        if initial_cycle:
            distributions = [initial_seasonal_prior(case["config"], t) for t in range(1, 5)]
        else:
            distributions = [seasonal_scale_prior(case["config"])] * 4
        all_priors.append(distributions)
    if initial_cycle:
        bound = max(pr["upper"] for distributions in all_priors for _, pr in distributions)*1.15
        x = np.linspace(-bound, bound, 500)
        xlabel, quantity = "Starting seasonal contribution / °C", "starting_seasonal_component"
    else:
        bound = max(pr["upper"] for distributions in all_priors for _, pr in distributions)*1.15
        x = np.linspace(.001, bound, 500)
        xlabel, quantity = "Seasonal observation scale / °C", "seasonal_observation_scale"
    for case, distributions in zip(cases, all_priors):
        for t, (density, _) in enumerate(distributions if initial_cycle else distributions[:1]):
            prior_ax.plot(x, density(x), color=case["color"], lw=1.5,
                          label=case["label"] if t == 0 else None, alpha=1 if t == 0 else .5)
    prior_ax.set(xlabel=xlabel, ylabel="Prior density", xlim=(x[0], x[-1]), ylim=(0, None))
    prior_ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(.5, 1.02))
    for channel, ax in zip(CH, axes):
        for i, case in enumerate(cases):
            if initial_cycle:
                d = ctx.read(f"{channel}_seasonal.csv", case["directory"]).head(4).copy()
                dates = pd.to_datetime(d.time)
                if len(d) != 4 or not np.all(np.diff(dates.dt.year*12+dates.dt.month) == 3):
                    raise ValueError("First four seasonal states must be consecutive")
                d["season"] = dates.dt.month.map(SEASONS)
                d["step"] = np.arange(4)
            else:
                d = ctx.read(f"{channel}_scale_by_season.csv", case["directory"])
            if set(d.season) != set(SEASON_ORDER):
                raise ValueError(f"{channel}: seasonal report does not contain all four seasons")
            for j, season in enumerate(SEASON_ORDER):
                r = d[d.season == season].iloc[0]
                s = transformed_summary(r)
                interval_mark(ax, s, 3-j+.17*(1-i), case["color"])
                step = int(r["step"]) if initial_cycle else 0
                comparison_row(ctx, case, quantity, channel, s, all_priors[i][step][1], "°C", season=season)
        ax.set_yticks(range(4), SEASON_ORDER[::-1])
        ax.set_ylim(-.55, 3.55)
        ax.set_title(channel + " posterior", loc="left", weight="bold", fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel(xlabel)
        if initial_cycle:
            ax.axvline(0, color=GRAY, lw=.65, ls="--")
    ctx.save(fig, name)


@figure("structural_priors")
def plot_structural_priors(ctx):
    states = prior_cases(ctx, ["initial_state_narrow", "reference", "initial_state_wide"])
    slopes = prior_cases(ctx, ["half_initial_slope", "reference", "double_initial_slope"])
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 6.7), layout="constrained",
                             gridspec_kw={"width_ratios": [1., 1.4]})
    for row, (cases, key, quantity, title, unit) in enumerate([
        (states, "baseline_sd", "initial_level", "A  Initial level", "Initial level / °C"),
        (slopes, "initial_slope_sd", "initial_rate", "B  Initial rate", "Initial rate / °C per decade")]):
        priors, sds = [], []
        for case in cases:
            cfg = case["config"]
            gain = 10*cfg["model"]["steps_per_year"] if row else 1.
            sd = cfg["priors"][key]*gain
            mean = 0. if row else cfg["priors"].get("initial_level_mean", 0.)
            priors.append(normal_prior(sd, mean))
            sds.append(sd)
            case["label"] = f'Prior SD = {sd:g}'
        x = np.linspace(min(pr["lower"] for pr in priors)*1.25,
                        max(pr["upper"] for pr in priors)*1.25, 500)
        for case, sd, pr in zip(cases, sds, priors):
            axes[row, 0].plot(x, norm.pdf(x, loc=pr["mean"], scale=sd), color=case["color"], label=case["label"])
        axes[row, 0].set(xlabel=unit, ylabel="Prior density", ylim=(0, None))
        axes[row, 0].set_title(title, loc="left", weight="bold", fontsize=10)
        axes[row, 0].legend(fontsize=8)
        # Raw initial coefficients for reflected GEV minima are internal;
        # negate the mean and swap/negate interval endpoints to report °C.
        def multiplier(c, case):
            sign = -1. if c in ("TXn", "TNn") else 1.
            return sign * (10*case["config"]["model"]["steps_per_year"] if row else 1.)
        field = "slope" if row else "level"
        scalar_posterior_panel(ctx, axes[row, 1], cases, CH,
                              lambda c: f"initial.channel.{c}.{field}", priors,
                              quantity, unit, multiplier)
    ctx.save(fig, "structural_coefficients_prior_posterior")
    for case in states:
        pr = case["config"]["priors"]
        case["label"] = f'Initial contrast SD = {pr["seasonal_initial_sd"]:g} °C'
    seasonal_comparison(ctx, states, "initial_seasonal_prior_posterior", initial_cycle=True)


@figure("observation_priors")
def plot_observation_priors(ctx):
    shapes = prior_cases(ctx, ["xi_narrow", "reference", "xi_wide"])
    scales = prior_cases(ctx, ["observation_scale_half", "reference", "observation_scale_double"])
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 6.7), layout="constrained",
                             gridspec_kw={"width_ratios": [1., 1.4]})
    priors = []
    x = np.linspace(-1.5, 1.5, 500)
    for case in shapes:
        pr = case["config"]["priors"]
        if pr["xi_prior"] != "normal" or any(v is not None for v in pr.get("xi_bounds", [])):
            raise ValueError("The shape prior panel requires the declared unbounded normal prior")
        sd = pr["xi_sd"]
        priors.append(normal_prior(sd))
        axes[0, 0].plot(x, norm.pdf(x, scale=sd), color=case["color"], label=f'Prior SD = {sd:g}')
    axes[0, 0].set(xlabel="GEV shape $\\xi$", ylabel="Prior density", ylim=(0, None))
    axes[0, 0].set_title("A  GEV shape", loc="left", weight="bold", fontsize=10)
    axes[0, 0].legend(fontsize=8)
    scalar_posterior_panel(ctx, axes[0, 1], shapes, ["TXx", "TXn", "TNx", "TNn"],
                          lambda c: f"xi.{c}", priors, "GEV_shape", "GEV shape $\\xi$")
    priors = []
    for case in scales:
        a, b = case["config"]["priors"]["observation_variance"]
        priors.append(dict(mean=np.sqrt(b)*np.exp(gammaln(a-.5)-gammaln(a)),
                           lower=np.sqrt(invgamma.ppf(.025, a=a, scale=b)),
                           upper=np.sqrt(invgamma.ppf(.975, a=a, scale=b))))
    x = np.linspace(.001, 1.05*max(pr["upper"] for pr in priors), 500)
    for case in scales:
        a, b = case["config"]["priors"]["observation_variance"]
        axes[1, 0].plot(x, 2*x*invgamma.pdf(x*x, a=a, scale=b), color=case["color"],
                        label=f'Variance prior IG({a:g}, {b:g})')
    unit = "Geometric-mean scale / °C"
    axes[1, 0].set(xlabel=unit, ylabel="Prior density", ylim=(0, None))
    axes[1, 0].set_title("B  Baseline observation scale", loc="left", weight="bold", fontsize=10)
    axes[1, 0].legend(fontsize=8)
    scalar_posterior_panel(ctx, axes[1, 1], scales, CH, lambda c: f"sigma.{c}",
                          priors, "geometric_mean_observation_scale", unit)
    ctx.save(fig, "observation_coefficients_prior_posterior")
    seasonal = prior_cases(ctx, ["narrow_seasonal_log_scale", "reference", "wide_seasonal_log_scale"])
    for case in seasonal:
        sd = case["config"]["model"]["scale_prior_sd"]
        case["label"] = f'Log-contrast prior SD = {sd:g}'
    seasonal_comparison(ctx, seasonal, "observation_seasonal_prior_posterior", initial_cycle=False)


@figure('validation_pits')
def plot_validation_pits(ctx):
    read, save = (ctx.read, ctx.save)
    cases = ctx.validation()

    def pitfigure(origins, cols, name):
        if len(origins) > len(cols):
            raise ValueError("Choose at most three origins per compact PIT figure")
        check_origins(cases, origins)
        fig, axes = panel((7, 4.8), True)
        for c, ax in zip(CH, axes.flat):
            for origin, col in zip(origins, cols):
                f = cases[(cases.origin_date == origin) & (cases.channel == c)]
                n = len(f)
                u = np.sort(f.pit)
                assert n and len(set(f.time)) == n
                ax.step(np.r_[0, u, 1], np.r_[0, np.arange(1, n + 1) / n, 1], where='post', color=col, lw=1.3, label=f'{origin[:4]} ({n} seasons)')
            ax.plot([0, 1], [0, 1], color=GRAY, ls='--', lw=0.7)
            ax.set(xlim=(0, 1), ylim=(0, 1))
            ax.set_aspect('equal')
            ax.set_title(c, loc='left', weight='bold')
        for ax in axes[:, 0]:
            ax.set_ylabel('Empirical CDF')
        for ax in axes[1]:
            ax.set_xlabel('Predictive PIT')
        legend(fig, axes[0, 0], ncol=len(origins))
        save(fig, name)
    for origins, colors, filename in [(ctx.settings.get('historical_origins', ['1956-11', '1976-11', '1996-11']), [BLUE, TEAL, RED], 'validation_historical_pit'), (ctx.settings.get('recent_origins', ['2015-11', '2020-11']), [BLUE, RED, TEAL], 'validation_recent_pit')]:
        try:
            pitfigure(origins, colors, filename)
        except MissingInput as exc:
            ctx.notes.append(str(exc))
            print(f'SKIP {filename}: {exc}')

@figure('validation_paths')
def plot_validation_paths(ctx):
    read, save = (ctx.read, ctx.save)
    check_origins(ctx.validation(), ['2015-11', '2020-11'])
    cases = ctx.validation()
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.7), layout='constrained', sharex='col', sharey='row')
    for j, o in enumerate(['2015-11', '2020-11']):
        for i, (c, s, thr) in enumerate([('TXx', 'JJA', 35), ('TNn', 'DJF', -10)]):
            ax = axes[i, j]
            d = cases[(cases.origin_date == o) & (cases.channel == c) & (cases.season == s)]
            x = d.meteorological_year
            ax.fill_between(x, d.q025, d.q975, color=BLUE, alpha=0.18)
            ax.plot(x, d.predictive_mean, color=BLUE, lw=1.4, label='Predictive mean')
            ax.scatter(x, d.observed, color='#253441', s=15, label='Held-out observation', zorder=3)
            ax.axhline(thr, color=RED, ls=':', lw=1)
            ax.set_title(f'{o[:4]} origin: {s} {c}', loc='left')
            ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
            ax.set_ylabel('Temperature / °C')
            ax.tick_params(axis='x', labelrotation=45)
    for ax in axes[1]:
        ax.set_xlabel('Meteorological year')
    legend(fig, axes[0, 0], ncol=2)
    save(fig, 'validation_recent_paths')

@figure('forecast')
def plot_forecast(ctx):
    read, save = (ctx.read, ctx.save)
    try:
        cases = ctx.validation()
    except MissingInput:
        cases = pd.DataFrame(columns=['origin_date', 'channel', 'season', 'time', 'observed'])
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.6), layout='constrained')
    for j, c in enumerate(['TXm', 'TXx']):
        ax = axes[0, j]
        d = read(c + '_forecast_uncertainty.csv')
        d = d[pd.to_datetime(d.time).dt.month == 6]
        obs = d[(d.target == 'observation') & np.isclose(d.nominal, 0.95)]
        loc = d[(d.target == 'location') & np.isclose(d.nominal, 0.95)]
        x = yr(obs)
        ax.fill_between(x, obs.lower, obs.upper, color=BLUE, alpha=0.16, label='Observation: 95% PI')
        ax.plot(x, obs['mean'], color=BLUE, lw=1.5, label='Predictive mean')
        ax.fill_between(x, loc.lower, loc.upper, color=RED, alpha=0.2, label='Location: 95% interval')
        ax.plot(x, loc['mean'], color=RED, lw=1.1, label='Location mean')
        for nominal, ls in [(0.95, '--'), (0.99, ':')]:
            u = d[(d.target == 'observation') & np.isclose(d.nominal, nominal)]
            ax.plot(yr(u), u.upper_quantile, color=BLUE, lw=0.8, ls=ls, label=f'{100 * nominal:.0f}th percentile')
        h = cases[(cases.origin_date == '2015-11') & (cases.channel == c) & (cases.season == 'JJA')]
        ax.scatter(pd.to_datetime(h.time).dt.year, h.observed, color='#253441', s=10, zorder=3)
        ax.set_title(f'{c}: summer forecasts', loc='left')
        ax.set_ylabel('Temperature / °C')
        ax.set_xticks([2020, 2030, 2040, 2050])
        ax.tick_params(axis='x', labelrotation=45)
    for j, s in enumerate(['35', '39p7']):
        ax = axes[1, j]
        d = read('TXx_forecast_risk.csv' if s == '35' else 'TXx_forecast_risk_39p7.csv')
        d = d[pd.to_datetime(d.time).dt.month == 6]
        bands(ax, d, BLUE, 100)
        ax.set_title('TXx > ' + ('35' if s == '35' else '39.7') + ' °C', loc='left')
        ax.set(xlabel='Year', ylabel='Summer probability / %', ylim=(0, 100))
        ax.tick_params(axis='x', labelrotation=45)
    legend(fig, axes[0, 0], ncol=3, fontsize=7)
    save(fig, 'forecast_30y')


def loess_linear(x, y, fraction=.18):
    """Local linear LOESS, tricube weights, no robustness reweighting."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not 0 < fraction <= 1 or len(x) < 3:
        raise ValueError("LOESS needs at least three observations and a span in (0,1]")
    neighbours = max(3, int(np.ceil(fraction * len(x))))
    result = np.empty_like(y)
    for i, xi in enumerate(x):
        distance = np.abs(x - xi)
        radius = np.partition(distance, neighbours-1)[neighbours-1]
        w = (1 - np.minimum(distance / radius, 1)**3)**3
        design = np.column_stack([np.ones(len(x)), x - xi])
        result[i] = np.linalg.lstsq(design * np.sqrt(w[:, None]),
                                    y * np.sqrt(w), rcond=None)[0][0]
    return result


@figure("exploration")
def plot_exploration(ctx):
    data = ctx.seasonal_data()
    fig = plt.figure(figsize=(10, 6.4), layout="constrained")
    grid = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1.15])
    axes = [fig.add_subplot(grid[i, j]) for i in range(3) for j in range(2)]
    overlay = fig.add_subplot(grid[:, 2])
    order = ["TXm", "TNm", "TXx", "TNx", "TXn", "TNn"]
    months = data.index.month
    # A winter's label is the year in which it ends.
    season_year = data.index.year + (months == 12)
    colors = [BLUE, RED, GOLD, TEAL, "#78618c", GRAY]
    smooths = []
    for c, ax, color in zip(order, axes, colors):
        for month, col in [(12, BLUE), (3, TEAL), (6, RED), (9, GOLD)]:
            selected = months == month
            ax.plot(season_year[selected], data.loc[selected, c],
                    color=col, lw=.55, label=SEASONS[month])
        ax.set_title(c, loc="left", weight="bold")
        ax.set_ylabel("Temperature / °C")
        historyaxis(ax)
        anomaly = data[c] - data[c].groupby(months).transform("mean")
        x = data.index.year + (months-1)/12
        smooth = loess_linear(x, anomaly, ctx.settings.get("loess_span", .18))
        style = "-" if c.endswith("m") else "--" if c.endswith("x") else ":"
        overlay.plot(x, smooth, color=color, ls=style, lw=1.4, label=c)
        smooths.append(pd.DataFrame({"time": data.index, "channel": c,
                                    "anomaly": anomaly.to_numpy(), "smooth": smooth}))
    for ax in axes[-2:]:
        ax.set_xlabel("Meteorological year")
    historyaxis(overlay)
    overlay.axhline(0, color=GRAY, lw=.7)
    overlay.set(xlabel="Year", ylabel="Within-season anomaly / °C")
    overlay.legend(fontsize=8)
    legend(fig, axes[0], ncol=4)
    ctx.save(fig, "data_exploration")
    pd.concat(smooths, ignore_index=True).to_csv(ctx.output / "exploratory_smoothers.csv", index=False)


@figure("prior")
def plot_prior(ctx):
    """Exact Gaussian horizon contributions mixed over the declared priors."""
    anchors = ctx.anchors()
    sd0 = float(ctx.config["priors"]["initial_slope_sd"])
    count = int(ctx.settings.get("prior_draws", 50000))
    rng = np.random.default_rng(ctx.settings.get("prior_seed", 19830))
    horizons = np.array([40, 120])
    # Shared scales are drawn ONCE per replication, then six signed amplitudes.
    amplitudes = {}
    for c, A in anchors.items():
        tau = A * np.abs(rng.normal(size=(count, 1)))
        amplitudes[c] = tau * rng.normal(size=(count, 6))
    initial = sd0 * rng.normal(size=(count, 6))
    components = {}
    # Covariances preserve the joint 10/30-year construction, not only its marginals.
    for c in anchors:
        covariance = np.empty((2, 2))
        for i, h in enumerate(horizons):
            for j, k in enumerate(horizons):
                if c == "level":
                    covariance[i, j] = min(h, k)
                elif c == "slope":
                    r = np.arange(1, min(h, k))
                    covariance[i, j] = np.sum((h-r)*(k-r))
                else:
                    covariance[i, j] = 2 * min(h, k) / 4
        z = rng.normal(size=(count, 6, 2)) @ np.linalg.cholesky(covariance).T
        components[c] = amplitudes[c][:, :, None] * z
    components["initial_rate"] = initial[:, :, None] * horizons
    components["combined"] = sum(components.values())
    labels = ["Level", "Integrated slope", "Seasonal", "Initial rate", "Combined"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), layout="constrained")
    rows = []
    for i, (h, ax) in enumerate(zip(horizons, axes)):
        for j, (component, label) in enumerate(zip(components, labels)):
            a = components[component][:, :, i]
            low, high = np.quantile(a, [.025, .975])
            mean = a.mean()
            ax.hlines(4-j, low, high, color=BLUE, lw=1.3)
            ax.scatter(mean, 4-j, color=BLUE, s=20)
            rows.append(dict(years=int(h/4), component=component, mean=mean,
                             lower=low, upper=high, sd=float(a.std())))
        ax.axvline(0, color=GRAY, lw=.7)
        ax.set_yticks(range(5), labels[::-1])
        ax.set(xlabel="Prior temperature contribution / °C", ylim=(-.5, 4.5))
        ax.set_title(f"{int(h/4)} years", loc="left", weight="bold")
    ctx.save(fig, "prior_calibration")
    pd.DataFrame(rows).to_csv(ctx.output / "prior_calibration_summary.csv", index=False)


@figure("validation_decades")
def plot_validation_decades(ctx):
    cases = ctx.validation()
    origins = ctx.settings.get("historical_origins", ["1956-11", "1976-11", "1996-11"])
    check_origins(cases, origins)
    if len(origins) > 3:
        raise ValueError("Choose at most three historical origins for this compact layout")
    fig, axes = plt.subplots(6, 2, figsize=(6.5, 8.5), layout="constrained")
    rows = []
    for c, axs in zip(["TXm", "TNm", "TXx", "TNx", "TXn", "TNn"], axes):
        coverages = []
        for origin, color in zip(origins, [BLUE, TEAL, RED]):
            cover, widths = [], []
            for lo, hi in [(1, 10), (11, 20), (21, 30)]:
                d = cases[(cases.origin_date == origin) & (cases.channel == c)
                          & cases.horizon.between(4*(lo-1)+1, 4*hi)]
                if d.empty:
                    raise MissingInput(f"{origin}, {c}, years {lo}-{hi}: no observed cases")
                expected = np.arange(4*(lo-1)+1, 4*(lo-1)+1+len(d))
                if not np.array_equal(np.sort(d.horizon.astype(int)), expected):
                    raise ValueError(f"{origin}, {c}: missing seasons inside the validation window")
                coverage = 100 * d.observed.between(d.q025, d.q975).mean()
                width = (d.q975 - d.q025).mean()
                cover.append(coverage)
                widths.append(width)
                rows.append(dict(origin=origin, channel=c, years=f"{lo}-{hi}",
                                 n=len(d), n_requested=40, complete=len(d)==40,
                                 coverage95=coverage, width95=width))
            coverages.extend(cover)
            axs[0].plot([1, 2, 3], cover, "o-", ms=3, lw=1.1, color=color, label=origin[:4])
            axs[1].plot([1, 2, 3], widths, "o-", ms=3, lw=1.1, color=color)
        axs[0].axhline(95, color=GRAY, lw=.8, ls="--")
        axs[0].set_ylim(min(78, min(coverages)-4), 104)
        axs[0].set_ylabel("Coverage / %")
        axs[1].set_ylabel("Width / °C")
        axs[1].margins(y=.3)
        for ax in axs:
            ax.text(.02, .96, c, transform=ax.transAxes, va="top", weight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.8))
            ax.set_xticks([1, 2, 3], ["1–10", "11–20", "21–30"])
            ax.set_xlim(.8, 3.2)
    for axs in axes[:-1]:
        for ax in axs:
            ax.tick_params(labelbottom=False)
    for ax in axes[-1]:
        ax.set_xlabel("Forecast years")
    legend(fig, axes[0, 0], ncol=len(origins))
    ctx.save(fig, "validation_decades")
    pd.DataFrame(rows).to_csv(ctx.output / "validation_decades_summary.csv", index=False)


@figure("tight")
def plot_tight(ctx):
    if not ctx.settings.get("tight_root"):
        raise MissingInput("Set tight_root to the EARLIER tight-calibration reports")
    root = Path(ctx.settings["tight_root"]).expanduser().resolve()
    variants = ["half_level", "reference", "double_level", "half_slope", "double_slope"]
    mapping = ctx.settings.get("tight_reports", {})
    reports = {}
    for v in variants:
        if v in mapping:
            reports[v] = report_dir(mapping[v])
        else:
            choices = [root / f"posterior_{v}", root / f"final_posterior_{v}", root / v]
            existing = [p for p in choices if (p / "report/config.json").exists()
                        or (p / "config.json").exists()]
            if len(existing) != 1:
                raise MissingInput(f"Set tight_reports['{v}'] explicitly: found {len(existing)} reports")
            reports[v] = report_dir(existing[0])
    a = ctx.anchors(reports["reference"])
    if not np.allclose([a[c] for c in ["level", "slope", "seasonal"]], [.01, .0001, .01]):
        raise ValueError("tight_root is not the earlier (0.01, 0.0001, 0.01) experiment")
    fig, axes = panel((7, 4.7))
    labels = [r"$A_\alpha/2$", "Reference", r"$2A_\alpha$", r"$A_\beta/2$", r"$2A_\beta$"]
    for c, ax in zip(CH, axes.flat):
        color = BLUE if c.startswith("TX") else RED
        for i, v in enumerate(variants):
            d = ctx.read(f"{c}_slope_C_per_decade.csv", reports[v]).sort_values("time").iloc[-1]
            ax.hlines(4-i, d.lower, d.upper, color=color, lw=1.2)
            ax.scatter(d["mean"], 4-i, color=color, s=15)
        ax.set_yticks(range(5), labels[::-1])
        ax.set_title(c, loc="left", weight="bold")
        ax.set_xlabel("Final rate / °C per decade")
        ax.set_ylim(-.5, 4.5)
    ctx.save(fig, "earlier_tight_calibration")


def auto_config(explicit=None):
    """Make editor Run/Play use the same local settings as the CLI."""
    if explicit is not None:
        return explicit
    roots = list(dict.fromkeys([Path.cwd(), Path(__file__).resolve().parent]))
    for name in ("plotting_config.json", "plotting_config.example.json"):
        for root in roots:
            candidate = root / name
            if candidate.is_file():
                return candidate
    return None


def discover_reports(roots):
    """Search folder names inside the checkout; do not traverse environments."""
    reference, validation = set(), set()
    excluded = {"node_modules", "site-packages", "__pycache__", "venv", "env",
                "build", "dist", "tests", "startup_probes", "startup_checks"}
    for root in dict.fromkeys(Path(p).expanduser().resolve() for p in roots):
        if not root.is_dir():
            continue
        for directory, children, files in os.walk(root, followlinks=False):
            children[:] = [d for d in children if not d.startswith(".")
                           and d not in excluded and not d.startswith("bucex_env")]
            path = Path(directory)
            if path.name == "final_posterior_reference":
                if (path / "report/config.json").is_file() or (path / "config.json").is_file():
                    reference.add(path)
                children[:] = []
            elif path.name == "validation_report_horizon_reference":
                if "validation_cases.csv" in files:
                    validation.add(path)
                children[:] = []
            elif path.name == "report":
                children[:] = []
    return sorted(reference), sorted(validation)


def setup_paths(settings, args, requested):
    """Resolve missing defaults once, before starting any figures."""
    tier = settings.get("tier", "paper")
    root = Path(settings.get("results_root", "results/serra_1983")).expanduser().resolve()
    if (root / tier).is_dir():
        root /= tier
    reference = Path(settings.get("reference_report", root / "final_posterior_reference"))
    has_reference = (reference / "config.json").is_file() or (reference / "report/config.json").is_file()
    vroot = Path(settings.get("validation_root", "results/serra_1984_validation")).expanduser()
    vcase = settings.get("validation_cases")
    vexplicit = bool(args.validation_root or args.validation_cases or settings.get("validation_reports"))
    validation_missing = not ((vcase and Path(vcase).is_file()) or vroot.exists() or vexplicit)
    needs_reference = bool(set(requested) - {"validation_pits", "validation_paths", "validation_decades", "tight"})
    if not args.no_discover and (not has_reference or validation_missing):
        roots = [args.search_root] if args.search_root else [Path.cwd(), Path(__file__).resolve().parent]
        print("Looking for extracted BUCEX reports in " + ", ".join(map(str, dict.fromkeys(roots))), flush=True)
        refs, vals = discover_reports(roots)
        if not has_reference and not (args.results_root or args.reference_report):
            preferred = [p for p in refs if tier in p.parts]
            candidates = preferred or refs
            if len(candidates) == 1:
                reference = candidates[0]
                root = reference.parent
                settings["results_root"] = str(root)
                settings["reference_report"] = str(reference)
                has_reference = True
            elif len(candidates) > 1 and needs_reference:
                choices = "\n".join("  " + str(p.parent) for p in candidates)
                raise ValueError("More than one final fit was found. Choose its folder with --results-root:\n" + choices)
        if validation_missing:
            vtier = settings.get("validation_tier", tier)
            preferred = [p for p in vals if vtier in p.parts]
            candidates = preferred or vals
            if len(candidates) == 1:
                settings["validation_root"] = str(candidates[0])
                settings.pop("validation_cases", None)
                print(f"Validation report: {candidates[0]}", flush=True)
            elif len(candidates) > 1:
                choices = "\n".join("  " + str(p) for p in candidates)
                raise ValueError("More than one validation report was found. Choose with --validation-root:\n" + choices)
    if needs_reference and not has_reference:
        raise ValueError(
            f"The final-fit report is not available at:\n  {reference}\n"
            "Extract your downloaded results archive into this checkout first, then rerun.\n"
            "For archive_1981_biobot(3).zip the required file is paper/final_posterior_reference/report/config.json.\n"
            "If the extracted results are elsewhere, pass --results-root followed by the folder containing\n"
            "paper/final_posterior_reference (or the paper folder itself). This command only plots saved results.")
    if has_reference:
        print(f"Reference report: {reference}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON plotting settings; paths are relative to your current directory")
    parser.add_argument("--results-root", help="BUCEX results root or its paper/screen subfolder")
    parser.add_argument("--reference-report", help="Optional explicit reference task/report folder")
    parser.add_argument("--validation-root", help="1.9.8.4 results root or validation_report_horizon_reference folder")
    parser.add_argument("--validation-cases", help="Optional consolidated validation_cases.csv/forecast_cases.csv")
    parser.add_argument("--tight-root", help="Optional older tight-calibration results folder")
    parser.add_argument("--daily", help="Daily Uccle CSV; only needed for the exploratory figure")
    parser.add_argument("--out", dest="output", help="Output directory; default manuscript/figures")
    parser.add_argument("--tier", choices=["paper", "screen"])
    parser.add_argument("--figures", nargs="+", choices=["all"] + list(FIGURES), help="Select figure groups")
    parser.add_argument("--formats", nargs="+", choices=["pdf", "png", "svg"])
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--search-root", type=Path, help="Look for extracted reports in this folder instead of the checkout")
    parser.add_argument("--no-discover", action="store_true", help="Use only configured report paths")
    parser.add_argument("--list", action="store_true", help="List available groups without reading data")
    parser.add_argument("--strict", action="store_true", help="Return a nonzero exit code if any requested input is missing")
    args = parser.parse_args(argv)
    if args.list:
        print("Figure groups: " + ", ".join(FIGURES))
        return 0
    settings = {}
    config_path = auto_config(args.config)
    if config_path:
        print(f"Plot configuration: {config_path.resolve()}", flush=True)
        settings = json.loads(config_path.read_text(encoding="utf-8-sig"))
    settings.update({k: v for k, v in vars(args).items()
                     if k not in {"config", "strict", "list", "search_root", "no_discover"} and v is not None})
    requested = settings.get("figures", ["all"])
    if isinstance(requested, str):
        requested = [requested]
    if "all" in requested:
        requested = list(FIGURES)
    invalid = set(requested) - set(FIGURES)
    if invalid:
        parser.error(f"Unknown figure groups: {sorted(invalid)}")
    try:
        setup_paths(settings, args, requested)
    except ValueError as exc:
        print(f"INPUT ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    formats = settings.get("formats", ["pdf", "png"])
    if set(formats) - {"pdf", "png", "svg"}:
        parser.error("formats must be pdf, png or svg")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 11, "axes.labelsize": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#637582",
        "text.color": "#253441", "axes.labelcolor": "#253441", "axes.grid": True,
        "grid.alpha": .5, "grid.color": "#dce3e7", "axes.axisbelow": True,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
        "figure.facecolor": "white", "legend.frameon": False})
    ctx = Inputs(settings)
    failed, skipped = [], []
    for name in requested:
        before = len(ctx.outputs)
        try:
            FIGURES[name](ctx)
        except MissingInput as exc:
            skipped.append({"group": name, "reason": str(exc)})
            print(f"SKIP {name}: {exc}", flush=True)
            plt.close("all")
        except (ValueError, KeyError, IndexError, OSError) as exc:
            failed.append({"group": name, "reason": str(exc)})
            print(f"ERROR {name}: {exc}", file=sys.stderr, flush=True)
            plt.close("all")
        else:
            made = ", ".join(f["name"] for f in ctx.outputs[before:])
            print(f"DONE {name}: {made}", flush=True)
    ctx.output.mkdir(parents=True, exist_ok=True)
    comparison_csv = None
    if ctx.prior_comparisons:
        comparison_csv = ctx.output / "prior_sensitivity_summaries.csv"
        pd.DataFrame(ctx.prior_comparisons).to_csv(comparison_csv, index=False)
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "settings": settings, "point_summary": "mean", "interval": .95,
                "sensitivity_envelope": "range across posterior means, not a credible interval",
                "sources_sha256": ctx.sources, "figures": ctx.outputs,
                "prior_sensitivity_summaries": str(comparison_csv) if comparison_csv else None,
                "skipped": skipped, "notes": ctx.notes, "errors": failed}
    (ctx.output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(ctx.outputs)} figures to {ctx.output}", flush=True)
    if skipped or ctx.notes:
        print("Some inputs were unavailable; see figure_manifest.json. Existing unrelated files were kept.")
    return int(bool(failed) or (args.strict and bool(skipped or ctx.notes)) or not ctx.outputs)


if __name__ == "__main__":
    raise SystemExit(main())
