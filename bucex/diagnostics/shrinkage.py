"""Prior and posterior checks for shared innovation-scale hierarchies."""
import numpy as np
import pandas as pd
from scipy.special import ndtri

from ..priors import MarginalPriors
from ..priors.shrinkage import NORMAL_ABSOLUTE_MEDIAN
from .experiments import draw_structural_prior
from .posterior import rhat, ess_bulk, ess_tail


def draw_marginal_prior(priors, size=2000, *, seed=None):
    """Draw the joint prior, integrating shared hyperparameters exactly by sampling.

    A shared hyperparameter is drawn once per joint draw and reused for all
    channels. Returned ``channels`` contain the usual structural-prior draws;
    ``shared`` contains population medians. These are unconditional priors,
    never priors evaluated at posterior hyperparameter estimates.
    """
    if not isinstance(priors, MarginalPriors):
        raise TypeError("Supply MarginalPriors(...).")
    if int(size) != size or size < 1:
        raise ValueError("size must be a positive integer.")
    size = int(size)
    rng = np.random.default_rng(seed)
    shared = {} if priors.shrinkage is None else priors.shrinkage.sample_medians(size, rng=rng)
    channels = {}
    for name, prior in priors.channels.items():
        samples = draw_structural_prior(prior, size, seed=int(rng.integers(0, 2**32)))
        for component, median in shared.items():
            if component == "initial_slope":
                samples["initial.slope"] = rng.normal(size=size) * median
            else:
                samples[f"sd.{component}"] = np.abs(rng.normal(size=size)) * median / NORMAL_ABSOLUTE_MEDIAN
        channels[name] = samples
    return {"channels": channels, "shared": shared}


def compare_shared_shrinkage(fit, *, level=.95):
    """Hyperprior and posterior intervals in physical innovation-SD units."""
    if not 0 < level < 1:
        raise ValueError("level must lie between zero and one.")
    spec = getattr(fit.priors, "shrinkage", None)
    if spec is None:
        raise ValueError("This fit has no shared innovation-scale hierarchy.")
    quantiles = np.array([(1-level)/2, .5, (1+level)/2])
    rows = []
    for component, anchor in spec.anchors.items():
        values = fit.parameter(f"shrinkage.shared.{component}", combine_chains=False)
        for distribution in ("prior", "posterior"):
            interval = (anchor*np.exp(spec.log_sd*ndtri(quantiles)) if distribution == "prior"
                        else np.quantile(values, quantiles))
            rows.append(dict(component=component, distribution=distribution,
                scale="normal_SD" if component == "initial_slope" else "population_median",
                anchor=anchor, log_sd=spec.log_sd,
                lower=interval[0], median=interval[1], upper=interval[2], credible_interval=level,
                rhat=rhat(values) if distribution == "posterior" else np.nan,
                ess_bulk=ess_bulk(values) if distribution == "posterior" else np.nan,
                ess_tail=ess_tail(values) if distribution == "posterior" else np.nan,
                probability_above_anchor=float(np.mean(values > anchor)) if distribution == "posterior" else .5))
    return pd.DataFrame(rows)


def compare_initial_slope_priors(fit, *, size=20000, seed=None, rate_multiplier=120., level=.95):
    """Signed initial slopes on the original response scale, against full priors.

    ``rate_multiplier`` converts per-update slopes (120 for monthly to per
    decade). A shared prior scale is integrated, never replaced by its posterior.
    """
    if not isinstance(fit.priors, MarginalPriors):
        raise TypeError("Supply a joint fit with MarginalPriors.")
    if not 0 < level < 1 or not np.isfinite(rate_multiplier) or rate_multiplier <= 0:
        raise ValueError("Invalid interval level or rate_multiplier.")
    samples = draw_marginal_prior(fit.priors, size, seed=seed)["channels"]
    rows = []
    for name in fit.channel_names:
        key = f"initial.channel.{name}.slope"
        if key not in fit.parameter_draws:
            continue
        sign = fit.model.channel(name).transform_sign
        posterior = fit.parameter(key, combine_chains=False)*sign*rate_multiplier
        for distribution, values in [("prior", samples[name]["initial.slope"]*sign*rate_multiplier),
                                     ("posterior", posterior)]:
            lo, mid, hi = np.quantile(values, [(1-level)/2, .5, (1+level)/2])
            rows.append(dict(channel=name, component="initial_slope", distribution=distribution,
                scale="signed_rate", rate_multiplier=rate_multiplier, lower=lo, median=mid,
                upper=hi, credible_interval=level,
                rhat=rhat(posterior) if distribution == "posterior" else np.nan,
                ess_bulk=ess_bulk(posterior) if distribution == "posterior" else np.nan,
                ess_tail=ess_tail(posterior) if distribution == "posterior" else np.nan))
    return pd.DataFrame(rows)


__all__ = ["draw_marginal_prior", "compare_shared_shrinkage", "compare_initial_slope_priors"]
