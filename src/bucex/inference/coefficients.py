"""Normal and Gaussian-reference elliptical-slice coefficient updates."""
import numpy as np
from ..priors import Fixed, Normal
from ._coefficients import gaussian_reference, reference_slice
from ._reference import coefficient_reference
from .paths import pseudo_observations


def coefficient_prior(state, shared):
    priors = list(state.compiled.priors)
    for g in state.compiled.groups:
        if g.name in shared:
            priors[g.coefficient] = Normal(0, shared[g.name])
    mean = np.array([p.value if isinstance(p, Fixed) else p.mean for p in priors], dtype=float)
    sd = np.array([0. if isinstance(p, Fixed) else p.sd for p in priors], dtype=float)
    return mean, sd


def coefficients_step(state, shared, controls, rng):
    X = state.compiled.coefficient_design(state.z)
    mean, sd = coefficient_prior(state, shared)
    free, fixed = np.flatnonzero(sd > 0), np.flatnonzero(sd == 0)
    values = mean.copy()
    if not free.size:
        state.theta = values
        return {'coefficient_evaluations': 0}
    offset = X[:, fixed] @ mean[fixed]
    y = state.y-offset
    design, pm, L = X[:, free], mean[free], np.diag(sd[free])
    op = state.observation_params()
    response, variance = pseudo_observations(y, y, state.obs, op, controls)
    reference = gaussian_reference(design, response, variance, pm, L)
    metric = {}
    if getattr(state.obs, 'gaussian_location', False):
        m, root, _ = reference
        w, evaluations = m+root@rng.normal(size=len(m)), 1
    else:
        (m, root, precision), metric = coefficient_reference(design, y, state.obs, op, pm, L, reference, controls)
        current = (state.theta[free]-pm)/sd[free]
        def correction(w):
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                ll = float(np.sum(state.obs.logpdf(y, design@(pm+L@w), op)))
            delta = w-m
            return ll-.5*(w@w)+.5*(delta@precision@delta) if np.isfinite(ll) else -np.inf
        w, evaluations = reference_slice(current, m, root, correction, rng)
    values[free] = pm+L@w
    state.theta = values
    return {**metric, 'coefficient_evaluations': evaluations}
