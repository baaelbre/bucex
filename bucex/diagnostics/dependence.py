"""Posterior predictive checks of residual cross-series dependence."""
import numpy as np
import pandas as pd

from ..dependence.gaussian import normal_scores, sample_normal_scores


def residual_dependence_check(fit, *, draws=200, seed=None, level=.9, by_phase=False):
    """Compare observed normal-score correlations with joint replications.

    Works for private independent, copula and shared-state multiseries fits.
    Scores condition on each draw's latent location, scale and shape, so a
    common warming trend and seasonal heteroscedasticity are removed first.
    Reported predictive p-values are descriptive posterior predictive checks,
    not frequentist significance tests. Serial dependence needs a separate
    check and held-out comparisons remain necessary.
    """
    if not fit.is_multiseries_model:
        raise ValueError("Use an aligned MultiSeriesModel fit.")
    if not 0 < level < 1 or int(draws) < 1:
        raise ValueError("draws must be positive and level must lie in (0,1).")
    rng = np.random.default_rng(seed)
    indices = rng.choice(fit.n_draws, min(int(draws), fit.n_draws), replace=False)
    channels = fit.model.channels
    eta = fit.eta_draws(original_scale=False)
    sigmas = {c.name: fit.sigma_draws(channel=c.name) for c in channels}
    copula = fit.model.copula
    observed, replicated = [], []
    from ..core.calendar import seasonal_phases
    period = getattr(copula,"period",None) or fit.model.period or 12
    phase = seasonal_phases(period,fit.n_time,fit.dates) if by_phase else np.zeros(fit.n_time,int)
    groups = {int(p):phase == p for p in np.unique(phase) if np.sum(phase == p) >= 3}
    if not groups:
        raise ValueError("At least three observations per reported phase are required.")
    for index in indices:
        params = {name: array[index] for name in fit.parameter_draws
                  if np.asarray(array := fit.parameter(name)).ndim == 1}
        params.update({f"sigma.{name}": values[index] for name,values in sigmas.items()})
        z = normal_scores(fit.y, eta[index], channels, params)
        if not np.all(np.isfinite(z)):
            raise ValueError("This check requires complete aligned observations.")
        if copula is not None and copula.seasonal:
            R = copula.correlation_path(params,fit.channel_names,fit.n_time,fit.dates)
            replicate = np.einsum('tij,tj->ti',np.linalg.cholesky(R),rng.normal(size=z.shape))
        else:
            correlation = copula.correlation_matrix(params, fit.channel_names) if copula else np.eye(len(channels))
            replicate = sample_normal_scores(correlation,fit.n_time,rng)
        observed.append([np.corrcoef(z[mask], rowvar=False) for mask in groups.values()])
        replicated.append([np.corrcoef(replicate[mask], rowvar=False) for mask in groups.values()])
    observed, replicated = np.asarray(observed), np.asarray(replicated)
    rows = []
    for g,(label,mask) in enumerate(groups.items()):
        for i in range(len(channels)):
            for j in range(i):
                obs, rep = observed[:,g,i,j], replicated[:,g,i,j]
                lower, upper = np.quantile(rep, [(1-level)/2, (1+level)/2])
                rows.append(dict(first=channels[j].name, second=channels[i].name,
                    phase=label, observed_correlation_mean=float(obs.mean()), replicated_lower=float(lower),
                    replicated_upper=float(upper), predictive_p_absolute=float(np.mean(np.abs(rep) >= np.abs(obs))),
                    posterior_draws=len(indices), n_time=int(mask.sum())))
    return pd.DataFrame(rows)


def residual_serial_check(fit, *, lags=(1,12), draws=200, seed=None, level=.9):
    """Descriptive residual-score ACF across posterior draws; not a whiteness test.

    Smoothed states use the observations being checked. For model selection,
    combine this check with forecast residuals and held-out predictive scores.
    """
    from ..models import Channel
    channels = fit.model.channels if fit.is_multiseries_model else (
        Channel('response',fit.model.observation,fit.model.components,
                tail='lower' if fit.transform_sign < 0 else None),)
    rng = np.random.default_rng(seed)
    selected = rng.choice(fit.n_draws,min(int(draws),fit.n_draws),replace=False)
    eta = fit.eta_draws(original_scale=False)
    y = fit.y if fit.is_multiseries_model else fit.y[:,None]
    if eta.ndim == 2:
        eta = eta[...,None]
    scores = []
    for index in selected:
        params = {}
        for channel in channels:
            name = channel.name if fit.is_multiseries_model else None
            params['sigma.'+channel.name] = fit.sigma_draws(channel=name)[index]
            if channel.family == 'gev':
                params['xi.'+channel.name] = fit.parameter('xi.'+channel.name if name else 'xi')[index]
        scores.append(normal_scores(y,eta[index],channels,params))
    scores = np.asarray(scores)
    rows = []
    for lag in lags:
        if int(lag) != lag or lag < 1:
            raise ValueError('lags must be positive integers.')
        if fit.n_time-lag < 3:
            continue
        for j,channel in enumerate(channels):
            values = [np.corrcoef(z[:-lag,j],z[lag:,j])[0,1] for z in scores]
            low,median,high = np.quantile(values,[(1-level)/2,.5,(1+level)/2])
            rows.append(dict(channel=channel.name,lag=lag,lower=low,median=median,upper=high,
                             n_pairs=fit.n_time-lag,posterior_draws=len(selected)))
    return pd.DataFrame(rows)


__all__ = ["residual_dependence_check", "residual_serial_check"]
