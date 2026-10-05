"""Conditional updates for constant or repeating seasonal dispersion and GEV shape."""
import numpy as np
from ..priors import Fixed
from ._slice import _slice_sample_real
from ._coefficients import reference_slice

def observation_step(state, rng):
    priors = state.model.priors
    eta = state.location()
    def ll(**kwargs):
        with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
            value = float(np.sum(state.obs.logpdf(state.y, eta, state.observation_params(**kwargs))))
        return value if np.isfinite(value) else -np.inf
    if not isinstance(priors.variance, Fixed):
        if getattr(state.obs, 'gaussian_location', False):
            factors = state.observation_params()['sigma']/np.sqrt(state.sigma2)
            b = priors.variance.scale + .5*np.sum(((state.y-eta)/factors)**2)
            state.sigma2 = b/rng.gamma(priors.variance.shape + len(eta)/2)
        else:
            def target(kappa):
                with np.errstate(over='ignore', under='ignore'):
                    v, inv = np.exp(2*kappa), np.exp(-2*kappa)
                if not np.isfinite(v) or v <= 0 or not np.isfinite(inv):
                    return -np.inf
                return ll(sigma2=v) - 2*priors.variance.shape*kappa - priors.variance.scale*inv
            kappa, _ = _slice_sample_real(.5*np.log(state.sigma2), target, rng, width=.1)
            state.sigma2 = np.exp(2*kappa)
    if state.scale_contrasts.size:
        size = state.scale_contrasts.size
        root = np.eye(size)*state.model.observation.scale.prior_sd
        state.scale_contrasts, _ = reference_slice(state.scale_contrasts, np.zeros(size), root,
                                                     lambda u: ll(u=u), rng)
    if state.obs.name == 'gev' and not isinstance(priors.shape, Fixed):
        def target(xi):
            return ll(xi=xi) - .5*((xi-priors.shape.mean)/priors.shape.sd)**2
        state.xi, _ = _slice_sample_real(state.xi, target, rng, width=.05)

