"""Posterior predictive checks of residual cross-series dependence."""
import numpy as np
import pandas as pd

from ..dependence.gaussian import normal_scores, sample_normal_scores


def residual_dependence_check(fit, *, draws=200, seed=None, level=.9):
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
    for index in indices:
        params = {name: array[index] for name in fit.parameter_draws
                  if np.asarray(array := fit.parameter(name)).ndim == 1}
        params.update({f"sigma.{name}": values[index] for name,values in sigmas.items()})
        z = normal_scores(fit.y, eta[index], channels, params)
        if not np.all(np.isfinite(z)):
            raise ValueError("This check requires complete aligned observations.")
        correlation = copula.correlation_matrix(params, fit.channel_names) if copula else np.eye(len(channels))
        observed.append(np.corrcoef(z, rowvar=False))
        replicated.append(np.corrcoef(sample_normal_scores(correlation, fit.n_time, rng), rowvar=False))
    observed, replicated = np.asarray(observed), np.asarray(replicated)
    rows = []
    for i in range(len(channels)):
        for j in range(i):
            obs, rep = observed[:,i,j], replicated[:,i,j]
            lower, upper = np.quantile(rep, [(1-level)/2, (1+level)/2])
            rows.append(dict(first=channels[j].name, second=channels[i].name,
                observed_correlation_mean=float(obs.mean()), replicated_lower=float(lower),
                replicated_upper=float(upper), predictive_p_absolute=float(np.mean(np.abs(rep) >= np.abs(obs))),
                posterior_draws=len(indices), n_time=fit.n_time))
    return pd.DataFrame(rows)


__all__ = ["residual_dependence_check"]
