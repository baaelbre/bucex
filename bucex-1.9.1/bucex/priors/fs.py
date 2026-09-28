"""Explicit, comparable continuous innovation priors for FS trajectories."""
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import betaln, erf, ndtri

from .structural import (BayesianLassoPrior, DiagonalNormalPrior, FSGaussianPriors,
    FSGEVPriors, InverseGammaPrior, NormalPrior, TripleGammaPrior)


@lru_cache(maxsize=32)
def triple_gamma_median(spike_shape=.5, tail_shape=.5):
    """Median |N(0,r/d)|, r~Gamma(a,1), d~Gamma(c,1), by quadrature.

    Fixed global variance multiplier is one; this is not a calibration of
    a learned-global or regularized hierarchy. No random calibration seed.
    """
    a,c = float(spike_shape),float(tail_shape)
    if not np.isfinite(a+c) or min(a,c) <= 0:
        raise ValueError("Triple-gamma shapes must be positive and finite.")
    def cdf(log_x):
        def integrand(log_v):
            log_density = a*log_v-(a+c)*np.logaddexp(0.,log_v)-betaln(a,c)
            z = log_x-.5*log_v-.5*np.log(2.)
            probability = 1. if z > 6 else erf(np.exp(z))
            return float(np.exp(log_density)*probability)
        return quad(integrand,-np.inf,np.inf,epsabs=1e-9,epsrel=1e-9,limit=250)[0]
    return float(np.exp(brentq(lambda x:cdf(x)-.5,-50.,50.,xtol=1e-9)))


def fs_priors(family, *, period=12, innovation="normal", innovation_median=None,
              initial_level=None, initial_slope=None, seasonal_initial_sd=2.25,
              observation_variance=None, xi_prior=None, xi_max_abs=None,
              spike_shape=.5, tail_shape=.5):
    """Construct proper, median-matched normal/lasso/triple-gamma FS priors.

    ``innovation_median`` declares prior medians of physical monthly SDs
    using keys level, trend, season. Defaults are 0.01, 0.00005, 0.02 in the
    response's units. They are starting assumptions, not estimated defaults.
    ``lasso`` fixes lambda²=1; ``triple_gamma`` fixes its global multiplier=1
    and shapes, while sampling every local mixing variable. Choose hyperprior
    learning explicitly with the lower-level prior dataclasses if required.
    All three families have zero probability of an exactly zero innovation.

    Initial level defaults to N(0,20²), without reading/centering on the data.
    Shape defaults to unrestricted N(0,.3²). Optional xi_max_abs and
    GEV.xi_bounds intersect the prior support; the observation-dependent GEV
    support is always enforced regardless of these optional prior bounds.
    For no seasonal component, period=None is accepted.
    """
    family = str(getattr(family, "name", family)).lower()
    if family not in {"gaussian", "gev"}:
        raise ValueError("family must be gaussian or gev.")
    if innovation not in {"normal", "lasso", "triple_gamma"}:
        raise ValueError("innovation must be normal, lasso, or triple_gamma.")
    medians = dict(innovation_median or {"level":.01,"trend":.00005,"season":.02})
    if set(medians) != {"level","trend","season"} or any(not np.isfinite(v) or v <= 0 for v in medians.values()):
        raise ValueError("Declare positive finite innovation medians for level, trend, and season.")
    if period is not None and (int(period) != period or period < 2):
        raise ValueError("period must be an integer >=2 or None.")
    k = 0 if period is None else int(period)-1
    fields = dict(alpha0=initial_level or NormalPrior(0.,20.),
                  beta0=initial_slope or NormalPrior(0.,.0025),
                  sigma2=observation_variance or InverseGammaPrior(2.,2.),
                  gamma0_season=DiagonalNormalPrior(np.zeros(k), np.full(k,seasonal_initial_sd)))
    if innovation == "normal":
        fields.update({"s_"+key:NormalPrior(0.,median/ndtri(.75)) for key,median in medians.items()})
    elif innovation == "lasso":
        fields["lasso"] = BayesianLassoPrior(fixed_lambda2=1.,
            coefficient_scale={key:value/np.log(2.) for key,value in medians.items()})
    else:
        unit_median = triple_gamma_median(spike_shape,tail_shape)
        fields["triple_gamma"] = TripleGammaPrior(
            coefficient_scale={key:value/unit_median for key,value in medians.items()},
            spike_shape=spike_shape,tail_shape=tail_shape,learn_global=False,learn_shapes=False)
    if family == "gev":
        return FSGEVPriors(**fields,xi=xi_prior or NormalPrior(0.,.3),xi_max_abs=xi_max_abs)
    return FSGaussianPriors(**fields)
