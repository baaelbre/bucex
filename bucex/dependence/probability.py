"""Conditional bivariate event probabilities without simulated event counts."""
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

from .gaussian import _gev_scores


def bivariate_normal_cdf(a, b, rho, *, rtol=1e-8):
    """Deterministic one-dimensional quadrature for a Gaussian lower rectangle.

    Integrates the smaller marginal tail directly, avoiding subtraction of
    near-one CDFs for rare joint upper events. Underflow beyond floating-point
    probability resolution remains possible. Raises if quadrature fails its
    reported relative-error criterion; no simulation zeros or clipping.
    """
    if not np.isfinite(rho) or not -1 < rho < 1:
        raise ValueError("rho must lie strictly between -1 and 1.")
    if np.isnan(a) or np.isnan(b):
        raise ValueError("Threshold scores must not be NaN.")
    if min(a,b) == -np.inf:
        return 0.
    if a == np.inf:
        return float(ndtr(b))
    if b == np.inf:
        return float(ndtr(a))
    if rho == 0:
        return float(ndtr(a)*ndtr(b))
    a,b = min(a,b),max(a,b)
    scale = np.sqrt((1-rho)*(1+rho))
    def integrand(z):
        return np.exp(-.5*z*z)/np.sqrt(2*np.pi)*ndtr((b-rho*z)/scale)
    value,error = quad(integrand,-np.inf,a,epsabs=0.,epsrel=rtol,limit=200)
    if value > 0 and error > 10*rtol*value:
        raise FloatingPointError("Bivariate probability quadrature did not meet its relative tolerance.")
    return float(value)


def _threshold_scores(forecast, channel, threshold):
    item = next(c for c in forecast.channels if c.name == channel)
    j = forecast.channel_names.index(channel)
    eta = forecast.eta[...,j]*item.transform_sign
    sigma = forecast.parameters.get('sigma_path.'+channel, forecast.parameters['sigma.'+channel][:,None])
    standardized = (item.transform_sign*threshold-eta)/sigma
    if item.family == 'gaussian':
        return item.transform_sign*standardized
    result = np.empty_like(eta)
    for draw in range(forecast.n_draws):
        xi = forecast.parameters['xi.'+channel][draw]
        supported = 1+xi*standardized[draw] > 0
        result[draw,~supported] = -np.inf if xi > 0 else np.inf
        result[draw,supported] = _gev_scores(standardized[draw,supported],np.zeros(supported.sum()),1.,xi)
    return item.transform_sign*result


def pair_probability_draws(forecast, events, *, operation='all', rtol=1e-8):
    """Condition on each retained parameter and latent/future-state draw."""
    if not forecast.is_multiseries_forecast or len(events) != 2:
        raise ValueError("Conditional quadrature currently supports exactly two distinct channels.")
    if operation not in {'all','any'}:
        raise ValueError("operation must be all or any.")
    bounds, directions, indices = [],[],[]
    for channel,(operator,threshold) in events.items():
        if channel not in forecast.channel_names:
            raise ValueError(f"Unknown channel {channel!r}.")
        if operator not in {'<','<=','>','>='} or not np.isfinite(threshold):
            raise ValueError("Use a finite threshold and <, <=, >, or >=.")
        direction = -1. if operator.startswith('>') else 1.
        directions.append(direction)
        bounds.append(direction*_threshold_scores(forecast,channel,threshold))
        indices.append(forecast.channel_names.index(channel))
    output = np.empty((forecast.n_draws,forecast.horizon))
    for draw in range(forecast.n_draws):
        params = {k:float(v[draw]) for k,v in forecast.parameters.items() if np.ndim(v[draw]) == 0}
        if forecast.copula is None:
            rho = np.zeros(forecast.horizon)
        else:
            R = forecast.copula.correlation_matrix(params,forecast.channel_names)
            if forecast.copula.seasonal:
                R = R[forecast.parameters['copula_phase'][draw].astype(int)-1]
            rho = np.broadcast_to(R[...,indices[0],indices[1]],(forecast.horizon,))*np.prod(directions)
        for t in range(forecast.horizon):
            a,b = bounds[0][draw,t],bounds[1][draw,t]
            output[draw,t] = (bivariate_normal_cdf(a,b,rho[t],rtol=rtol) if operation == 'all' else
                             1-bivariate_normal_cdf(-a,-b,rho[t],rtol=rtol))
    return output
