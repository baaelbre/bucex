"""Rank-normalized split R-hat, bulk/tail ESS and predictive diagnostics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
Array = np.ndarray

def _split_chains(values: Array) -> Array:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must have shape (chains, draws).")
    half = values.shape[1] // 2
    if half < 2:
        return values
    return np.concatenate([values[:, :half], values[:, -half:]], axis=0)

def _rank_normalize(values: Array) -> Array:
    values = np.asarray(values, dtype=float)
    flat = values.reshape(-1)
    ranks = rankdata(flat, method="average")
    return norm.ppf((ranks - 0.375) / (flat.size + 0.25)).reshape(values.shape)

def _basic_rhat(values: Array) -> float:
    if values.shape[0] < 2 or values.shape[1] < 2:
        return np.nan
    n = values.shape[1]
    within = float(np.mean(np.var(values, axis=1, ddof=1)))
    between = float(n * np.var(np.mean(values, axis=1), ddof=1))
    if within <= 0.0:
        return 1.0 if between <= 0.0 else np.inf
    variance = (n - 1.0) / n * within + between / n
    return float(np.sqrt(variance / within))

def rhat(values: Array) -> float:
    """Rank-normalized split R-hat with the folded-tail diagnostic."""

    values = np.asarray(values, dtype=float)
    if values.shape[0] < 2 or values.size == 0 or np.all(values == values.reshape(-1)[0]):
        # R-hat compares within- and between-chain variation.  It is not one
        # when both are zero; it is mathematically undefined.
        return np.nan
    split = _split_chains(values)
    if split.shape[0] < 2 or split.shape[1] < 2:
        return np.nan
    bulk = _basic_rhat(_rank_normalize(split))
    folded = np.abs(split - float(np.median(split)))
    tail = _basic_rhat(_rank_normalize(folded))
    return float(max(bulk, tail))

def _autocovariance(values: Array) -> Array:
    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values)
    n = values.size
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=size)
    covariance = np.fft.irfft(spectrum * np.conjugate(spectrum), n=size)[:n]
    return covariance / n

def ess_bulk(values: Array) -> float:
    """Rank-normalized split-chain bulk effective sample size."""

    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(values == values.reshape(-1)[0]):
        # A constant sampled indicator may represent posterior structural
        # certainty, but it supplies no autocorrelation information.
        return np.nan
    values = _rank_normalize(_split_chains(values))
    chains, draws = values.shape
    if draws < 3:
        return float(chains * draws)
    autocov = np.vstack([_autocovariance(chain) for chain in values])
    within = float(np.mean(autocov[:, 0])) * draws / (draws-1)
    between = float(draws * np.var(np.mean(values, axis=1), ddof=1)) if chains > 1 else 0.0
    variance_plus = (draws - 1.0) / draws * within + between / draws
    if variance_plus <= 0.0:
        return float(chains * draws)
    rho = np.ones(draws)
    for lag in range(1, draws):
        rho[lag] = 1.0 - (within - np.mean(autocov[:, lag])) / variance_plus
    pair_sums: list[float] = []
    for lag in range(0, draws - 1, 2):
        pair = rho[lag] + rho[lag + 1]
        if pair < 0.0:
            break
        pair_sums.append(float(pair))
    for index in range(1, len(pair_sums)):
        pair_sums[index] = min(pair_sums[index], pair_sums[index - 1])
    positive_sum = float(np.sum(pair_sums))
    return float(
        min(
            chains * draws,
            chains * draws / max(-1.0 + 2.0 * positive_sum, 1e-12),
        )
    )

def ess_tail(values: Array) -> float:
    """Minimum ESS of the 5% and 95% empirical-quantile indicators."""
    values = np.asarray(values,float)
    if values.ndim != 2 or not np.all(np.isfinite(values)) or np.all(values == values.flat[0]):
        return np.nan
    lower,upper = np.quantile(values,[.05,.95])
    return float(min(ess_bulk((values <= lower).astype(float)),ess_bulk((values <= upper).astype(float))))


def summary(fit, *, interval=.95, include_paths=False):
    """Posterior means, central credible intervals, rank R-hat and ESS."""
    from .results import summarize
    values = {f'{name}.{key}': v for name, c in fit.channels.items() for key, v in c.parameters.items()}
    values.update({f'tau.{k}': v for k, v in fit.shared_scales.items()})
    for name, c in fit.channels.items():
        for component in ('level', 'slope', 'seasonal'):
            array = c.path(component)
            times = range(array.shape[-1]) if include_paths else [array.shape[-1]-1]
            for t in times:
                values[f'{name}.{component}[t={t+1}]'] = array[:, :, t]
    rows = []
    for key, v in values.items():
        if not np.all(np.isfinite(v)):
            raise ValueError(f'Nonfinite posterior values for {key}.')
        s = summarize(v, interval)
        rows.append(dict(parameter=key, **{k: float(x) for k, x in s.items()},
            sd=float(np.std(v)), rhat=rhat(v), ess_bulk=ess_bulk(v), ess_tail=ess_tail(v)))
    return pd.DataFrame(rows).set_index('parameter')


def pit(channel, *, values=None, draw_indices=None):
    """Conditional PIT by posterior draw (in-sample); not a validation PIT."""
    values = channel.y if values is None else np.asarray(values)
    location, sigma = channel.path(), channel.sigma()
    xi = channel.parameters.get('xi', np.zeros(location.shape[:2]))[:, :, None]
    if draw_indices is not None:
        from .prediction import _take
        location, sigma = _take(location, draw_indices), _take(sigma, draw_indices)
        xi = _take(xi, draw_indices)
    return channel.model.observation.cdf(values, location, {'sigma': sigma, 'xi': xi})


def normal_scores(channel):
    """Posterior mean conditional normal scores; clipping is for diagnostics only."""
    u = pit(channel)
    return np.mean(norm.ppf(np.clip(u, 1e-12, 1-1e-12)), axis=(0, 1))


def posterior_checks(fit, *, draws=1000, seed=2002):
    """Paired posterior predictive KS discrepancies, not classical fitted KS tests.

    Compare D(replication; theta) with D(data; theta) within the same posterior
    draw. The resulting Bayesian p-values are descriptive and not uniform
    frequentist p-values. PIT/QQ plots and serial checks remain necessary.
    """
    from .prediction import replicate
    rep = replicate(fit, draws=draws, seed=seed)
    rows = []
    for name, c in fit.channels.items():
        u_data = pit(c, draw_indices=rep.draw_indices)
        u_rep = c.model.observation.cdf(rep.y[name], rep.location[name],
                                      {'sigma': rep.sigma[name], 'xi': rep.xi[name]})
        def distance(u):
            n = u.shape[1]
            ordered = np.sort(u, axis=1)
            return np.maximum(np.max(np.arange(1, n+1)/n-ordered, axis=1),
                              np.max(ordered-np.arange(n)/n, axis=1))
        obs, pred = distance(u_data), distance(u_rep)
        rows.append(dict(channel=name, ks_observed_mean=float(np.mean(obs)),
                         ks_bayesian_p=float(np.mean(pred >= obs)), draws=draws))
    return pd.DataFrame(rows)


def residual_association(fit):
    """Descriptive correlations of posterior mean score residuals, not fitted dependence."""
    return pd.DataFrame({name: normal_scores(c) for name, c in fit.channels.items()}).corr()


def predictive_pit(prediction, observed, *, channel=None):
    """Held-out marginal predictive CDF, averaging conditional CDFs over paths."""
    name = prediction.channel_name(channel)
    y = np.asarray(observed, float)
    if y.shape != (len(prediction.index),) or not np.all(np.isfinite(y)):
        raise ValueError('Held-out observations must match all forecast steps.')
    return prediction.models[name].observation.cdf(y, prediction.location[name],
        {'sigma': prediction.sigma[name], 'xi': prediction.xi[name]}).mean(axis=0)
