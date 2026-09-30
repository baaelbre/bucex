#!/usr/bin/env python3
"""Bayesian constant-trend benchmark for the BUCEX seasonal temperature analysis.

Run this file from your BUCEX checkout (tested with 1.9.8.2):
    python -u constant_trend_benchmark.py --start 1970-01 --origin 2015-11 --jobs 6

Quick TXx-only run:
    python -u constant_trend_benchmark.py --series TXx --chains 2 --warmup 500 --draws 1000

Optional comparisons against existing BUCEX validation reports:
    ... --compare pooled=/path/to/pooled/report --compare private=/path/to/private/report

Dependencies: your existing bucex installation, numpy, scipy, pandas, matplotlib.
No change to the BUCEX source or manuscript is required.

MODEL
    mu[j,t] = alpha[j,0] + t * beta[j] + gamma[j,season(t)].
The unknown beta is CONSTANT in time, not fixed at a known value. Each of the
six responses is fitted separately. All three innovation variances are exactly
zero; there are no shrinkage hyperparameters. Seasonal observation scales remain.
Means are Gaussian; maxima are GEV; minima use reflected GEV likelihoods.

DEFAULT PRIORS (matching the current seasonal specification)
    initial level: N(0, 10^2)
    initial zero-sum seasonal contrast coordinates: N(0, 10^2 I)
    beta: N(0, 0.01^2) degrees C per seasonal transition
          (SD 0.4 degrees C per decade; configurable with --slope-sd-decade)
    squared geometric-mean observation scale: inverse-gamma(shape=2, scale=2)
    seasonal log-scale contrast coordinates: N(0, 0.3^2 I)
    GEV shape: N(0, 0.3^2).

Only complete seasons ending by --origin enter the likelihood. Later data are
used only for verification. The forecast uses the same parameter draw for every
future season. Uncertainty in the fitted slope is propagated; no future state
innovations are generated. This is a single-origin forecast, not a rolling refit.

OUTPUT
    index.html                    overview and figures
    metrics.csv                   CRPS and predictive-median error, by season/all
    coverage.csv                  90%, 95%, 99% intervals, widths and tail misses
    event_counts.csv              expected/observed counts and P(N >= observed)
    count_distribution.csv        posterior predictive count probabilities
    predictions.csv               held-out cases, forecast quantiles, location means
    scores.csv                    individual CRPS cases, compatible comparison keys
    threshold_cases.csv           analytic predictive risks and Brier scores
    rate_estimates.csv            posterior mean constant slopes and 95% intervals
    numerical_status.csv          per-response MCMC diagnostic gate
    SERIES/                       fit, diagnostics, all forecast dates, risk curves
    paired_comparison.csv         if --compare is supplied; strictly matched cases

Posterior location, rate and risk summaries use means and equal-tailed 95%
intervals. Observation point forecasts and errors use the predictive MEDIAN,
matching the existing validation's median-error convention. GEV location is not
its observation mean. CRPS is computed from the empirical predictive sample.
MCMC diagnostics concern the retained posterior draws, not resampled forecasts.
More --predictive-draws improves observation Monte Carlo precision, not MCMC ESS.

The count distribution averages conditional Poisson-binomial probabilities over
posterior draws. It retains common-parameter uncertainty across years, and does
not substitute a binomial law with one average event probability. No observed
count, forecast error, or result from another model is hard-coded.

Defaults are 4 chains, 1000 warmup + 2000 retained per chain, and at least 20000
predictive observations per forecast time. Results are flagged for review if
rank-normalized R-hat >= 1.01 or bulk/tail ESS < 400 for a checked quantity.
Screening results that fail this gate should not be used for scientific claims.
Use --resume with exactly the same fit settings to re-use completed fits.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback

# Avoid nested numerical-library threading when responses run in parallel.
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

# A downloaded copy can run from the user's source checkout without installation.
if (Path.cwd() / "bucex" / "__init__.py").exists():
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
from scipy.special import ndtri
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import bucex as bx
except ImportError as exc:
    raise SystemExit("Run inside your BUCEX checkout or install it with: python -m pip install -e '.[plot]'") from exc

SERIES = ("TXm", "TNm", "TXx", "TXn", "TNx", "TNn")
DISPLAY_ORDER = ("TXm", "TXx", "TXn", "TNm", "TNx", "TNn")
SEASONS = {12: "DJF", 3: "MAM", 6: "JJA", 9: "SON"}
THRESHOLDS = {"TXm": (25.,), "TNm": (18.,), "TXx": (35., 36.6, 39.7),
              "TXn": (0.,), "TNx": (25.,), "TNn": (-10.,)}
PLOT_SEASON = {j: ("DJF" if j in ("TXn", "TNn") else "JJA") for j in SERIES}
LEVELS = (.90, .95, .99)
QUANTILES = (.005, .025, .05, .5, .95, .975, .995)
Q_NAMES = ("q005", "q025", "q05", "median", "q95", "q975", "q995")


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def band(values):
    lower, upper = np.quantile(values, [.025, .975], axis=0)
    return {"mean": np.mean(values, axis=0), "lower": lower, "upper": upper}


def calendar(dates):
    index = pd.DatetimeIndex(dates)
    return pd.DataFrame({"time": index,
        "season": [SEASONS[m] for m in index.month],
        "year": index.year + (index.month == 12).astype(int)})


def poisson_binomial_mixture(probabilities):
    """Exact conditional count recursion, followed by posterior averaging.

    Rows are posterior draws, columns are the selected forecast seasons.
    Taking the mean of probabilities BEFORE the recursion would be wrong.
    """
    p = np.asarray(probabilities, float)
    if p.ndim != 2 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected a finite draw-by-case matrix of probabilities.")
    distribution = np.zeros((len(p), p.shape[1] + 1))
    distribution[:, 0] = 1.
    for k in range(p.shape[1]):
        previous = distribution[:, :k+1].copy()
        distribution[:, :k+1] *= 1. - p[:, k, None]
        distribution[:, 1:k+2] += previous * p[:, k, None]
    return distribution.mean(axis=0)


def construct_model(name, settings):
    info = bx.UCCLE_INFO[name]
    scale = bx.LogScale(seasonal=bx.SeasonalScale(4, settings["log_scale_sd"],
                                               calendar="meteorological"))
    observation = bx.Gaussian(scale=scale) if info["family"] == "gaussian" else bx.GEV(scale=scale)
    components = (bx.LocalLinearTrend(level_mode="static", trend_mode="static"),
                  bx.DummySeasonal(4, mode="static"))
    channel = bx.Channel(name, observation, parameters={"mu": bx.Latent(components)},
                         tail=info["tail"])
    model = bx.MultiSeriesModel((channel,), copula=None)
    prior = bx.fs_priors(info["family"], period=4, innovation="normal",
        initial_level=bx.NormalPrior(0., 10.),
        initial_slope=bx.NormalPrior(0., settings["slope_sd_decade"] / 40.),
        seasonal_initial_sd=10., seasonal_initial_basis="orthonormal",
        observation_variance=bx.InverseGammaPrior(2., 2.),
        xi_prior=bx.NormalPrior(0., settings["xi_sd"]))
    return model, bx.MarginalPriors({name: prior})


def static_prediction(fit, name, dates, *, seed, historical=False):
    """Exact deterministic state propagation, with all posterior uncertainty.

    Direct propagation also supports zero process-noise columns, an edge case
    that fit.forecast() in BUCEX 1.9.8.2 does not handle. No library patch needed.
    A Forecast object provides the same density, PIT and quantile calculations.
    """
    if fit.compiled.noise_dim != 0:
        raise ValueError("This benchmark requires every innovation variance to be zero.")
    dates = pd.DatetimeIndex(dates)
    source_dates = pd.DatetimeIndex(fit.time)
    levels = fit.component_draws("level", channel=name)
    slopes = fit.component_draws("slope", channel=name)
    seasonal = fit.component_draws("seasonal", channel=name)
    scales = fit.sigma_draws(channel=name)
    # Assert the fitted model really has constant slopes and a repeating cycle.
    if not np.allclose(slopes, slopes[:, :1], atol=1e-8, rtol=1e-8):
        raise ValueError("The fitted slope is not constant.")
    if not np.allclose(seasonal[:, 4:], seasonal[:, :-4], atol=1e-8, rtol=1e-8):
        raise ValueError("The fitted seasonal cycle is not fixed.")
    if historical:
        if not dates.equals(source_dates):
            raise ValueError("Historical predictions must use the fitted dates.")
        level = levels
        mu = levels + seasonal
        sigma = scales
    else:
        expected = pd.date_range(source_dates[-1] + pd.DateOffset(months=3),
                                 periods=len(dates), freq="3MS")
        if not dates.equals(expected):
            raise ValueError("Forecast dates must continue the seasonal record without gaps.")
        positions = {m: np.flatnonzero(source_dates.month == m)[-1] for m in SEASONS}
        selected = np.array([positions[m] for m in dates.month])
        level = levels[:, -1, None] + slopes[:, -1, None] * np.arange(1, len(dates)+1)
        mu = level + seasonal[:, selected]
        sigma = scales[:, selected]
    item = fit.model.channels[0]
    sign = float(item.transform_sign)
    parameters = {"sigma": fit.parameter("sigma." + name), "sigma_path": sigma}
    if item.family == "gev":
        parameters["xi"] = fit.parameter("xi." + name)
    observations = sign * item.observation.sample(sign * mu, sigma=sigma,
        xi=None if item.family == "gaussian" else parameters["xi"][:, None],
        rng=np.random.default_rng(seed))
    forecast = bx.Forecast(observations=observations, eta=mu,
        states=np.empty((len(mu), len(dates), 0)), parameters=parameters,
        dates=dates.to_numpy(), family=item.family, tail="lower" if sign < 0 else "upper",
        observation_model=item.observation, transform_sign=sign, period=4,
        quantile_method="cdf")
    return forecast, level


def observation_ensemble(forecast, requested, seed):
    """Balanced repeated posterior draws; repetition does not increase MCMC ESS."""
    repeats = max(1, math.ceil(requested / forecast.n_draws))
    rng = np.random.default_rng(seed)
    chunks = []
    xi = forecast.parameters.get("xi")
    for _ in range(repeats):
        chunks.append(forecast.transform_sign * forecast.observation_model.sample(
            forecast.transform_sign * forecast.eta, sigma=forecast.sigma_draws(),
            xi=None if xi is None else xi[:, None], rng=rng))
    values = np.concatenate(chunks, axis=0)
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite predictive observations; inspect shape draws before scoring.")
    return values


def forecast_tables(fit, name, data, settings, target):
    horizon = settings["forecast_years"] * 4
    dates = pd.date_range(pd.Timestamp(fit.time[-1]) + pd.DateOffset(months=3),
                         periods=horizon, freq="3MS")
    forecast, level = static_prediction(fit, name, dates, seed=settings["seed"] + 501)
    quantiles = forecast.predictive_quantiles(QUANTILES)
    frame = calendar(dates)
    frame["channel"] = name
    frame["origin_date"] = settings["origin"]
    frame["horizon"] = np.arange(1, horizon+1)
    frame["observed"] = data[name].reindex(dates).to_numpy()
    for key, values in zip(Q_NAMES, quantiles):
        frame[key] = values
    # Also export one-sided upper predictive bounds, separately from central intervals.
    frame["upper95"] = frame.q95
    frame["upper99"] = forecast.predictive_quantiles([.99])[0]
    for prefix, values in (("location", forecast.eta), ("level", level)):
        for key, x in band(values).items():
            frame[prefix + "_" + key] = x
    frame["pit"] = forecast.pit(frame.observed.to_numpy())
    frame["median_error"] = frame["median"] - frame.observed
    observed_mask = frame.observed.notna().to_numpy()
    ensemble = observation_ensemble(forecast, settings["predictive_draws"], settings["seed"] + 502)
    frame["crps"] = np.nan
    if observed_mask.any():
        frame.loc[observed_mask, "crps"] = bx.crps_ensemble(
            ensemble[:, observed_mask], frame.loc[observed_mask, "observed"].to_numpy())
    frame.to_csv(target / "forecast.csv", index=False)
    tested = frame.loc[observed_mask].copy()
    tested.to_csv(target / "predictions.csv", index=False)
    tested[["time", "origin_date", "channel", "horizon", "crps"]].rename(
        columns={"crps": "value"}).assign(score="crps").to_csv(target / "scores.csv", index=False)

    coverage_rows, metrics, risks, counts, distributions, threshold_cases = [], [], [], [], [], []
    for season in ("ALL", "DJF", "MAM", "JJA", "SON"):
        selected = observed_mask & ((frame.season == season).to_numpy() if season != "ALL" else True)
        group = frame.loc[selected]
        if group.empty:
            continue
        metrics.append(dict(channel=name, origin_date=settings["origin"], season=season,
            n_cases=len(group), crps=group.crps.mean(), median_error=group.median_error.mean(),
            median_mae=group.median_error.abs().mean(),
            median_rmse=np.sqrt(np.mean(group.median_error**2))))
        for nominal, lo, hi in ((.9, "q05", "q95"), (.95, "q025", "q975"), (.99, "q005", "q995")):
            coverage_rows.append(dict(channel=name, origin_date=settings["origin"], season=season,
                nominal=nominal, n_cases=len(group),
                coverage=((group.observed >= group[lo]) & (group.observed <= group[hi])).mean(),
                mean_width=(group[hi]-group[lo]).mean(),
                lower_misses=int((group.observed < group[lo]).sum()),
                upper_misses=int((group.observed > group[hi]).sum())))

    direction = "<" if bx.UCCLE_INFO[name]["tail"] == "min" else ">"
    for threshold in THRESHOLDS[name]:
        probability_draws = forecast.probability_draws(threshold, direction=direction)
        risk = calendar(dates).assign(channel=name, origin_date=settings["origin"],
                                     threshold=threshold, direction=direction)
        for key, values in band(probability_draws).items():
            risk[key] = values
        risks.append(risk)
        event = (frame.observed < threshold) if direction == "<" else (frame.observed > threshold)
        cases = frame.loc[observed_mask, ["time", "season", "year", "channel", "origin_date", "horizon", "observed"]].copy()
        cases["threshold"], cases["direction"] = threshold, direction
        cases["event_probability"] = probability_draws[:, observed_mask].mean(axis=0)
        cases["observed_event"] = event[observed_mask].to_numpy()
        cases["brier"] = (cases.event_probability - cases.observed_event.astype(float))**2
        threshold_cases.append(cases)
        for season in ("ALL", "DJF", "MAM", "JJA", "SON"):
            selected = observed_mask & ((frame.season == season).to_numpy() if season != "ALL" else True)
            if not selected.any():
                continue
            probabilities = probability_draws[:, selected]
            pmf = poisson_binomial_mixture(probabilities)
            count_observed = int(event[selected].sum())
            expected_band = band(probabilities.sum(axis=1))
            predictive_limits = np.searchsorted(np.cumsum(pmf), [.025, .975])
            counts.append(dict(channel=name, origin_date=settings["origin"], season=season,
                threshold=threshold, direction=direction, n_cases=int(selected.sum()),
                observed_events=count_observed, expected_events=probabilities.mean(axis=0).sum(),
                expected_events_lower=expected_band["lower"], expected_events_upper=expected_band["upper"],
                predictive_count_lower=int(predictive_limits[0]), predictive_count_upper=int(predictive_limits[1]),
                probability_at_least_observed=float(pmf[count_observed:].sum()),
                brier=float(np.mean((probabilities.mean(axis=0)-event[selected].to_numpy())**2))))
            distributions.append(pd.DataFrame(dict(channel=name, origin_date=settings["origin"],
                season=season, threshold=threshold, direction=direction,
                count=np.arange(len(pmf)), probability=pmf)))
    for filename, rows in (("metrics", metrics), ("coverage", coverage_rows), ("event_counts", counts)):
        pd.DataFrame(rows).to_csv(target / (filename + ".csv"), index=False)
    for filename, rows in (("risks", risks), ("count_distribution", distributions), ("threshold_cases", threshold_cases)):
        pd.concat(rows, ignore_index=True).to_csv(target / (filename + ".csv"), index=False)

    # In-sample checks use the same original-temperature CDF orientation.
    fitted, _ = static_prediction(fit, name, pd.DatetimeIndex(fit.time),
                                  seed=settings["seed"] + 503, historical=True)
    train_values = data[name].reindex(pd.DatetimeIndex(fit.time)).to_numpy()
    in_sample = calendar(fit.time).assign(channel=name, observed=train_values,
                                        pit=fitted.pit(train_values))
    for key, values in band(fitted.eta).items():
        in_sample["location_" + key] = values
    in_sample.to_csv(target / "in_sample.csv", index=False)

    # Diagnose rates, forecast locations and a relevant conditional risk as well.
    shape = (fit.n_chains, fit.draws_per_chain)
    rate = fit.component_draws("slope", channel=name, combine_chains=False)[..., -1] * 40.
    representative = np.flatnonzero(frame.season == PLOT_SEASON[name])[-1]
    risk_check = forecast.probability_draws(THRESHOLDS[name][0], direction=direction)[:, representative]
    quantities = {"rate_C_per_decade": rate,
        "forecast_location_end": forecast.eta[:, representative].reshape(shape),
        "forecast_risk_end": risk_check.reshape(shape)}
    target_diagnostics = fit.contrast_diagnostics(quantities, credible_interval=.95)
    target_diagnostics.to_csv(target / "target_diagnostics.csv")
    slope = band(rate.reshape(-1))
    pd.DataFrame([dict(channel=name, mean=slope["mean"], lower=slope["lower"],
        upper=slope["upper"], prior_sd=settings["slope_sd_decade"], units="degC/decade")]).to_csv(
            target / "rate_estimates.csv", index=False)
    save_json(target / "forecast_details.json", dict(posterior_draws=forecast.n_draws,
        predictive_draws=len(ensemble), horizon=horizon, first_forecast=str(dates[0].date()),
        last_forecast=str(dates[-1].date()), verified_cases=int(observed_mask.sum()),
        prediction_method="exact constant-trend propagation; posterior mixture CDF quantiles"))
    return frame, in_sample, target_diagnostics


def numerical_check(diagnostics, targets, target):
    tables = [diagnostics.reset_index(), targets.reset_index()]
    checked, issues = 0, []
    for label, table in zip(("parameter", "scientific_target"), tables):
        for _, row in table.iterrows():
            name = str(row.get("parameter", row.get("quantity", row.get("target", "quantity"))))
            # Fixed indicators at exactly zero/one have no variance; retain their flag.
            if bool(row.get("constant", False)):
                continue
            checked += 1
            for key, threshold, upper in (("rhat", 1.01, True), ("ess_bulk", 400., False), ("ess_tail", 400., False)):
                v = row.get(key, np.nan)
                if not np.isfinite(v) or (v >= threshold if upper else v < threshold):
                    issues.append(dict(type=label, quantity=name, diagnostic=key, value=v))
    status = "passed" if checked and not issues else "needs_review"
    result = dict(status=status, checked=checked, issues=issues,
                  criterion="rank-normalized split R-hat < 1.01; bulk and tail ESS >= 400")
    save_json(target / "convergence.json", result)
    return result


def fit_one(payload):
    name, data, settings, target_string = payload
    target = Path(target_string)
    target.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        end = data.index + pd.DateOffset(months=3) - pd.Timedelta(days=1)
        training = data.loc[end <= pd.Period(settings["origin"], "M").end_time, [name]]
        model, prior = construct_model(name, settings)
        fit_spec = {k: settings[k] for k in ("origin", "start", "data_end", "daily_sha256",
            "chains", "warmup", "draws", "seed", "slope_sd_decade", "log_scale_sd", "xi_sd")}
        fit_spec.update(channel=name, bucex_version=bx.__version__, model=model.to_dict(),
                        script_sha256=settings["script_sha256"])
        checkpoint = target / "fit.bucex"
        if checkpoint.exists():
            if not settings["resume"] or json.loads((target / "fit_spec.json").read_text()) != clean_json(fit_spec):
                raise ValueError("Existing fit has different settings. Choose a new --out directory.")
            print(f"{name}: loading completed fit", flush=True)
            fit = bx.load_fit(checkpoint)
        else:
            save_json(target / "fit_spec.json", fit_spec)
            print(f"{name}: {len(training)} training seasons; {settings['chains']} chains; "
                  f"{settings['warmup']} warmup + {settings['draws']} retained each", flush=True)
            fit = bx.fit(training, model=model, priors=prior, parameterization="fs", asis=False,
                engine="ffbs" if bx.UCCLE_INFO[name]["family"] == "gaussian" else "laplace_mh",
                mcmc=bx.MCMC(draws=settings["draws"], warmup=settings["warmup"],
                    chains=settings["chains"], chain_workers=1, seed=settings["seed"],
                    progress=True, progress_every=500))
            fit.save(checkpoint)
        diagnostics = fit.diagnostics()["parameters"]
        diagnostics.to_csv(target / "mcmc_diagnostics.csv")
        data.loc[training.index, [name]].to_csv(target / "training_data.csv")
        frame, in_sample, targets = forecast_tables(fit, name, data, settings, target)
        assessment = numerical_check(diagnostics, targets, target)
        plot_response(name, frame, in_sample, settings, target)
        result = dict(channel=name, status=assessment["status"], checked=assessment["checked"],
            n_issues=len(assessment["issues"]), elapsed_seconds=time.monotonic()-started,
            directory=str(target), n_training=len(training), n_test=int(frame.observed.notna().sum()))
        save_json(target / "completed.json", result)
        print(f"{name}: finished in {result['elapsed_seconds']/60:.1f} min; {result['status']}", flush=True)
        return result
    except Exception:
        (target / "error.txt").write_text(traceback.format_exc())
        raise


def savefig(figure, path):
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def style_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=.18)


def plot_response(name, frame, in_sample, settings, target):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, season in zip(axes.flat, ("DJF", "MAM", "JJA", "SON")):
        history = in_sample[(in_sample.season == season) & (in_sample.year >= int(settings["origin"][:4])-15)]
        future = frame[frame.season == season]
        ax.scatter(history.year, history.observed, color=".55", s=12, label="Training observations")
        ax.fill_between(future.year, future.q005, future.q995, color="#24658a", alpha=.09, label="99% prediction interval")
        ax.fill_between(future.year, future.q025, future.q975, color="#24658a", alpha=.17, label="95% prediction interval")
        ax.plot(future.year, future["median"], color="#24658a", lw=1.5, label="Predictive median")
        ax.fill_between(future.year, future.location_lower, future.location_upper, color="#ac4939", alpha=.23, label="Location: 95% interval")
        ax.plot(future.year, future.location_mean, color="#ac4939", lw=1.5, label="Location: posterior mean")
        observed = future.dropna(subset=["observed"])
        ax.scatter(observed.year, observed.observed, color="black", marker="D", s=20, zorder=5, label="Held-out observations")
        ax.axvline(int(settings["origin"][:4]) + .5, color=".4", ls=":", lw=1)
        ax.set(title=f"{name}: {season}", xlabel="Season year", ylabel="Temperature / °C")
        style_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.subplots_adjust(bottom=.20, top=.95, hspace=.47, wspace=.25)
    savefig(fig, target / "forecasts.png")

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    for row, (title, table) in enumerate((("In sample (descriptive)", in_sample), ("Held-out forecasts", frame.dropna(subset=["observed"])))):
        u = table.pit.to_numpy()
        axes[row, 0].hist(u, bins=np.linspace(0, 1, 11), density=True, color="#24658a", alpha=.75)
        axes[row, 0].axhline(1, color=".4", ls="--")
        axes[row, 0].set(title=title, xlabel="PIT", ylabel="Density", xlim=(0, 1))
        z = ndtri(np.clip(np.sort(u), 1e-12, 1-1e-12))
        reference = ndtri((np.arange(len(u)) + .5) / len(u))
        axes[row, 1].scatter(reference, z, s=12, color="#24658a")
        if len(u):
            lo, hi = min(reference.min(), z.min()), max(reference.max(), z.max())
            axes[row, 1].plot([lo, hi], [lo, hi], color=".4", ls="--")
        axes[row, 1].set(title=title, xlabel="Standard-normal quantile", ylabel="Normal score")
        for ax in axes[row]:
            style_axis(ax)
    savefig(fig, target / "pit_qq.png")


def combined_figures(out, selected):
    names = [n for n in DISPLAY_ORDER if n in selected]
    frames = {n: pd.read_csv(out / n / "forecast.csv") for n in names}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, name in zip(axes.flat, names):
        f = frames[name]
        f = f[f.season == PLOT_SEASON[name]]
        ax.fill_between(f.year, f.q025, f.q975, color="#24658a", alpha=.18)
        ax.fill_between(f.year, f.location_lower, f.location_upper, color="#ac4939", alpha=.25)
        ax.plot(f.year, f["median"], color="#24658a", label="Predictive median")
        ax.plot(f.year, f.location_mean, color="#ac4939", label="Location mean")
        ax.scatter(f.year, f.observed, color="black", marker="D", s=18, label="Held out", zorder=4)
        ax.set(title=f"{name}: {PLOT_SEASON[name]}", xlabel="Season year", ylabel="Temperature / °C")
        style_axis(ax)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    savefig(fig, out / "forecast_six.png")

    fig, axes = plt.subplots(2, 3, figsize=(13, 6), constrained_layout=True)
    for ax, name in zip(axes.flat, names):
        f = frames[name]
        f = f[f.season == PLOT_SEASON[name]]
        ax.plot(f.year, f.q975-f.q025, color="#24658a", label="Observation PI")
        ax.plot(f.year, f.location_upper-f.location_lower, color="#ac4939", label="Location interval")
        ax.set(title=f"{name}: {PLOT_SEASON[name]}", xlabel="Season year", ylabel="95% interval width / °C")
        style_axis(ax)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    savefig(fig, out / "forecast_widths.png")

    risks = pd.read_csv(out / "risks.csv")
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), constrained_layout=True)
    for ax, name in zip(axes.flat, names):
        r = risks[(risks.channel == name) & (risks.season == PLOT_SEASON[name]) & (risks.threshold == THRESHOLDS[name][0])]
        ax.fill_between(r.year, 100*r.lower, 100*r.upper, color="#24658a", alpha=.18)
        ax.plot(r.year, 100*r["mean"], color="#24658a")
        ax.set(title=f"{name}: {PLOT_SEASON[name]} {r.direction.iloc[0]} {THRESHOLDS[name][0]:g} °C",
               xlabel="Season year", ylabel="Event probability / %")
        style_axis(ax)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    savefig(fig, out / "risk_six.png")
    if "TXx" in names:
        counts = pd.read_csv(out / "event_counts.csv")
        row = counts[(counts.channel == "TXx") & (counts.season == "JJA") & (counts.threshold == 35.)].iloc[0]
        dist = pd.read_csv(out / "count_distribution.csv")
        dist = dist[(dist.channel == "TXx") & (dist.season == "JJA") & (dist.threshold == 35.)]
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.bar(dist["count"], dist.probability, color="#24658a", alpha=.75)
        ax.axvline(row.observed_events, color="#ac4939", lw=2, label=f"Observed: {int(row.observed_events)}")
        ax.axvline(row.expected_events, color=".3", ls="--", label=f"Expected: {row.expected_events:.2f}")
        ax.set(title=f"JJA TXx > 35 °C: {int(row.n_cases)} held-out summers",
               xlabel="Number of summers", ylabel="Predictive probability", xticks=dist["count"])
        ax.text(.98, .94, f"P(N ≥ {int(row.observed_events)}) = {row.probability_at_least_observed:.4g}",
                ha="right", va="top", transform=ax.transAxes)
        ax.legend(loc="upper right", bbox_to_anchor=(1, .85))
        style_axis(ax)
        savefig(fig, out / "summer_TXx_count.png")


def compare_reports(specifications, out, origin):
    """Match dates/observations, resolving legacy numeric origins through folds.csv."""
    benchmark = pd.read_csv(out / "predictions.csv", parse_dates=["time"])
    results, issues = [], []
    for specification in specifications:
        if "=" not in specification:
            raise ValueError("--compare must be LABEL=/path/to/a/validation/report")
        label, path = specification.split("=", 1)
        root = Path(path).expanduser().resolve()
        if root == out.resolve() or out.resolve() in root.parents:
            raise ValueError("Comparison input cannot be this benchmark's output.")
        candidates = [root / "predictions.csv"] if (root / "predictions.csv").exists() else sorted(root.rglob("predictions.csv"))
        matched_frames = []
        for candidate in candidates:
            if out.resolve() == candidate.parent.resolve() or out.resolve() in candidate.parents:
                continue
            prediction = pd.read_csv(candidate, parse_dates=["time"])
            if "origin_date" in prediction:
                prediction = prediction[prediction.origin_date.astype(str).str[:7] == origin]
            else:
                folds_file = candidate.parent / "folds.csv"
                if not folds_file.exists() or "origin" not in prediction:
                    issues.append(f"{label}: skipped {candidate}: no origin metadata")
                    continue
                folds = pd.read_csv(folds_file)
                # Existing BUCEX validation stores the start of the last training block.
                ends = pd.to_datetime(folds.training_end) + pd.DateOffset(months=3) - pd.Timedelta(days=1)
                ids = folds.loc[ends.dt.to_period("M").astype(str) == origin, "origin"]
                prediction = prediction[prediction.origin.isin(ids)]
            if prediction.empty:
                continue
            if not {"channel", "observed", "median"} <= set(prediction):
                issues.append(f"{label}: skipped {candidate}: incompatible predictions columns")
                continue
            score_file = candidate.parent / "scores.csv"
            if score_file.exists():
                score = pd.read_csv(score_file, parse_dates=["time"])
                score = score[score.score == "crps"]
                keys = ["channel", "time"] + (["origin"] if "origin" in prediction and "origin" in score else [])
                if "origin_date" in prediction and "origin_date" in score:
                    keys.append("origin_date")
                prediction = prediction.drop(columns=["crps"], errors="ignore").merge(
                    score[keys + ["value"]].rename(columns={"value": "crps"}),
                    on=keys, how="left", validate="one_to_one")
            else:
                prediction["crps"] = np.nan
            matched_frames.append(prediction[["channel", "time", "observed", "median", "crps"]])
        if not matched_frames:
            issues.append(f"{label}: no usable {origin} forecast cases found under {root}")
            continue
        other = pd.concat(matched_frames, ignore_index=True)
        if other.duplicated(["channel", "time"]).any():
            raise ValueError(f"{label}: multiple forecasts for the same case. Point --compare to one specification's report.")
        joined = benchmark.merge(other, on=["channel", "time"], how="inner",
                                 suffixes=("_constant", "_other"), validate="one_to_one")
        if not np.allclose(joined.observed_constant, joined.observed_other, atol=1e-8, rtol=0):
            raise ValueError(f"{label}: held-out observations differ; comparison refused.")
        for name, series in joined.groupby("channel"):
            for season in ("ALL", "DJF", "MAM", "JJA", "SON"):
                g = series if season == "ALL" else series[series.season == season]
                if g.empty:
                    continue
                expected = benchmark[(benchmark.channel == name) & ((benchmark.season == season) if season != "ALL" else True)]
                results.append(dict(comparison=label, channel=name, season=season, origin_date=origin,
                    matched_cases=len(g), benchmark_cases=len(expected), complete_match=len(g) == len(expected),
                    constant_crps=g.crps_constant.mean(), other_crps=g.crps_other.mean(),
                    paired_crps_difference=(g.crps_constant-g.crps_other).mean(),
                    constant_median_error=(g.median_constant-g.observed_constant).mean(),
                    other_median_error=(g.median_other-g.observed_other).mean()))
    pd.DataFrame(results).to_csv(out / "paired_comparison.csv", index=False)
    save_json(out / "comparison_notes.json", {"issues": issues,
        "direction": "negative paired CRPS difference favours constant trend",
        "note": "A matched comparison does not certify convergence of the supplied alternative fits."})


def overview(out, results, settings):
    names = [r["channel"] for r in results if r.get("status") != "failed"]
    for filename in ("forecast", "predictions", "scores", "metrics", "coverage", "event_counts",
                     "count_distribution", "threshold_cases", "risks", "rate_estimates"):
        tables = [pd.read_csv(out / n / (filename + ".csv")) for n in names]
        if tables:
            pd.concat(tables, ignore_index=True).to_csv(out / (filename + ".csv"), index=False)
    pd.DataFrame(results).to_csv(out / "numerical_status.csv", index=False)
    if not names:
        return
    combined_figures(out, names)
    if settings["compare"]:
        compare_reports(settings["compare"], out, settings["origin"])
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Constant trend benchmark</title>",
        "<style>body{font:16px system-ui;max-width:1180px;margin:40px auto;padding:0 20px;color:#22303b}"
        "img{max-width:100%}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}"
        "th,td{padding:7px 10px;border-bottom:1px solid #ddd;text-align:right}"
        "h1,h2{font-weight:600}.note{background:#f2f5f7;padding:15px}</style></head><body>",
        f"<h1>Constant trend since {escape(settings['start'])}</h1>",
        f"<p>Training stops at {escape(settings['origin'])}. Six separate Gaussian/GEV models "
        "retain seasonal observation dispersion. The slope and seasonal cycle are constant in time. "
        "No observations after the forecast origin are used for fitting.</p>",
        "<p class='note'>A location line is a distributional location, not the GEV observation mean. "
        "Bias is forecast predictive median minus observation, matching the existing validation convention. "
        "Location, rate and probability point estimates are posterior means. Shaded intervals are 95%. "
        "The per-response figures also show 99% predictive intervals.</p>"]
    def table(filename, heading, selection=None):
        path = out / filename
        if path.exists() and path.stat().st_size > 2:
            frame = pd.read_csv(path)
            if selection is not None:
                frame = selection(frame)
            parts.extend(["<h2>" + escape(heading) + "</h2>", frame.to_html(index=False, float_format=lambda x: f"{x:.4g}")])
    table("numerical_status.csv", "MCMC checks: inspect these first")
    table("rate_estimates.csv", "Constant rates in °C per decade")
    table("metrics.csv", "Forecast accuracy", lambda f: f[f.season.isin(["ALL", "JJA", "DJF"])])
    table("event_counts.csv", "TXx summer threshold check", lambda f: f[(f.channel == "TXx") & (f.season == "JJA")])
    for filename, title in (("forecast_six.png", "Forecasts: observations and location"),
        ("forecast_widths.png", "Growth in interval widths"), ("risk_six.png", "Threshold risks"),
        ("summer_TXx_count.png", "How surprising was the observed summer count?")):
        if (out / filename).exists():
            parts.append(f"<h2>{title}</h2><img src='{filename}' alt='{title}'>")
    table("paired_comparison.csv", "Matched comparisons with the dynamic fits")
    for name in names:
        parts.append(f"<h2>{name}: all seasons and PIT checks</h2><img src='{name}/forecasts.png' alt='{name} forecasts'>"
                     f"<img src='{name}/pit_qq.png' alt='{name} PIT and QQ plots'>")
    parts.append("<p>Expected event counts use averaged conditional probabilities. The count distribution "
        "retains the same posterior parameter draw across all years. Its predictive interval describes "
        "the number of summers; the separate interval for the expected count describes uncertainty in "
        "that expectation. This benchmark is a forecast comparison, not a change-point test.</p></body></html>")
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--daily", type=Path, help="Daily DAY,TX,TN CSV; default: the unique data/Uccle_*.csv")
    parser.add_argument("--start", default="1970-01")
    parser.add_argument("--origin", default="2015-11", help="Last observed month used for fitting; must end a complete season")
    parser.add_argument("--data-end", default="2026-08", help="Last available observation month; used only for held-out verification")
    parser.add_argument("--series", nargs="+", choices=SERIES, default=list(SERIES))
    parser.add_argument("--forecast-years", type=int, default=30)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--predictive-draws", type=int, default=20000)
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1), help="Parallel responses; chains within each response run sequentially")
    parser.add_argument("--seed", type=int, default=19702015)
    parser.add_argument("--slope-sd-decade", type=float, default=.4)
    parser.add_argument("--log-scale-sd", type=float, default=.3)
    parser.add_argument("--xi-sd", type=float, default=.3)
    parser.add_argument("--out", type=Path, help="New output directory; default creates a timestamped directory in results/")
    parser.add_argument("--resume", action="store_true", help="Re-use saved fits only when data, priors, code and MCMC settings match")
    parser.add_argument("--compare", action="append", default=[], metavar="LABEL=REPORT_PATH")
    parser.add_argument("--dry-run", action="store_true", help="Check data, cutoff and job settings, without fitting or writing outputs")
    args = parser.parse_args()
    if args.chains < 2 or args.draws < 20 or args.warmup < 0 or args.jobs < 1 or args.predictive_draws < 2 or args.forecast_years < 1:
        parser.error("Need chains >= 2, draws >= 20, warmup >= 0, jobs/forecast-years >= 1 and predictive-draws >= 2.")
    if len(set(args.series)) != len(args.series):
        parser.error("Response names must be unique.")
    if not all(np.isfinite(v) and v > 0 for v in (args.slope_sd_decade, args.log_scale_sd, args.xi_sd)):
        parser.error("Prior SDs must be finite and positive.")
    if pd.Period(args.origin, "M").month not in (2, 5, 8, 11):
        parser.error("--origin must end a meteorological season (February, May, August, November).")
    return args


def main():
    args = arguments()
    daily = args.daily
    if daily is None:
        candidates = sorted((Path.cwd() / "data").glob("Uccle_*.csv"))
        if len(candidates) != 1:
            raise SystemExit("Specify --daily /path/to/Uccle_daily.csv (could not find exactly one data/Uccle_*.csv).")
        daily = candidates[0]
    daily = daily.expanduser().resolve()
    if not daily.is_file():
        raise SystemExit(f"Daily CSV not found: {daily}")
    data = bx.load_uccle_multiseries(series=args.series, start=args.start, end=args.data_end,
                                   frequency="seasonal", daily_source=str(daily))
    ends = data.index + pd.DateOffset(months=3) - pd.Timedelta(days=1)
    origin = pd.Period(args.origin, "M")
    training = data.loc[ends <= origin.end_time]
    if len(training) < 40:
        raise SystemExit("Need at least 10 years of complete training seasons.")
    last_month = (training.index[-1] + pd.DateOffset(months=2)).to_period("M")
    if last_month != origin:
        raise SystemExit("The requested forecast origin is not covered by a complete observed season.")
    verification = data.loc[ends > origin.end_time]
    verification = verification.iloc[:4*args.forecast_years]
    if verification.empty:
        raise SystemExit("No held-out observations after this forecast origin.")
    print(f"BUCEX {bx.__version__}; data {daily}", flush=True)
    print(f"Training: {training.index[0]:%Y-%m} season start through {origin}; {len(training)} blocks.", flush=True)
    print(f"Verification: {len(verification)} blocks; {(verification.index.month == 6).sum()} summers; "
          f"forecast horizon {args.forecast_years} years.", flush=True)
    if "TXx" in data:
        summer = verification.loc[verification.index.month == 6, "TXx"]
        print(f"Observed JJA TXx > 35 °C: {int((summer > 35.).sum())} / {len(summer)} summers.", flush=True)
    if args.dry_run:
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = (args.out or Path("results") / f"constant_trend_{args.start}_{args.origin}_{stamp}").expanduser().resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise SystemExit(f"Output directory is not empty: {out}. Use a new --out or --resume.")
    out.mkdir(parents=True, exist_ok=True)
    settings = vars(args).copy()
    settings.update(daily=str(daily), out=str(out), daily_sha256=sha256(daily),
                    script_sha256=sha256(__file__), bucex_version=bx.__version__)
    save_json(out / "run_settings.json", settings)
    data.to_csv(out / "seasonal_data.csv")
    save_json(out / "data_window.json", dict(first_training_block=str(training.index[0].date()),
        final_training_month=str(origin), n_training=len(training), n_test=len(verification),
        date_convention="season start; DJF year is its ending year", attrs=data.attrs))
    print(f"Output: {out}\nRunning {min(args.jobs, len(args.series))} responses in parallel.", flush=True)
    payloads = []
    for name in args.series:
        local = dict(settings, seed=args.seed + SERIES.index(name)*10007)
        payloads.append((name, data[[name]].copy(), local, str(out / name)))
    results = []
    if args.jobs == 1:
        for payload in payloads:
            try:
                results.append(fit_one(payload))
            except Exception as exc:
                traceback.print_exc()
                results.append(dict(channel=payload[0], status="failed", error=str(exc)))
    else:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(payloads)),
                                 mp_context=multiprocessing.get_context("spawn")) as executor:
            pending = {executor.submit(fit_one, p): p[0] for p in payloads}
            for future in as_completed(pending):
                name = pending[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    traceback.print_exc()
                    results.append(dict(channel=name, status="failed", error=str(exc)))
    results.sort(key=lambda x: SERIES.index(x["channel"]))
    overview(out, results, settings)
    print(f"\nReport: {out / 'index.html'}", flush=True)
    print(pd.DataFrame(results).to_string(index=False), flush=True)
    if any(r["status"] != "passed" for r in results):
        print("Some fits need numerical review. See convergence.json and the MCMC tables before interpreting differences.", flush=True)
    return int(any(r["status"] == "failed" for r in results))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
