"""Laplace/FFBS independence proposals with the exact likelihood correction."""
import numpy as np
from .smoothers import filter_gaussian, _mean_gaussian_smoother_1d, _draw_gaussian_smoother_1d


def pseudo_observations(y, eta, observation, params, controls):
    if getattr(observation, 'gaussian_location', False):
        return np.asarray(y), np.asarray(params['sigma'])**2
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        gradient = observation.grad_eta(y, eta, params)
        curvature = -observation.hess_eta(y, eta, params)
    curvature = np.maximum(np.nan_to_num(curvature, nan=controls.curvature_floor,
                        posinf=1/np.finfo(float).eps, neginf=controls.curvature_floor), controls.curvature_floor)
    variance = np.minimum(1/curvature, controls.maximum_variance)
    shift = np.nan_to_num(gradient*variance, nan=0., posinf=1e6, neginf=-1e6)
    limit = 10*np.max(params['sigma'])
    return eta+np.clip(shift, -limit, limit), variance


def project_path(path, F, Q):
    # Eliminate only numerical error in deterministic transition directions.
    projection = Q @ np.linalg.pinv(Q, hermitian=True)
    result = np.zeros_like(path)
    for t in range(1, len(path)):
        innovation = path[t]-F@path[t-1]
        result[t] = F@result[t-1]+projection@innovation
    return result


def finite_start(y, observation, params, F, Q, H, offset):
    """Deterministic proposal initialization, independent of the current path."""
    T, d = H.shape
    zero = np.zeros((T+1, d))
    def finite(z):
        eta = offset+np.sum(H*z[1:], axis=1)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return np.all(np.isfinite(observation.logpdf(y, eta, params)))
    if finite(zero):
        return zero
    z = zero.copy()
    for t in range(T):
        z[t+1] = F @ z[t]
        eta = offset[t]+H[t]@z[t+1]
        pp = {k: np.asarray(v)[t] if np.ndim(v) else v for k, v in params.items()}
        if np.isfinite(observation.logpdf(y[t], eta, pp)):
            continue
        direction = Q @ H[t]
        variance = H[t] @ direction
        if variance > np.finfo(float).tiny:
            z[t+1] += direction*((y[t]-eta)/variance)
    if finite(z):
        return z
    # Handles states whose innovations reach the observation after a lag.
    for factor in (1e-2, 1e-4, 1e-8, 1e-12):
        filt = filter_gaussian(y-offset, F, Q, H, np.asarray(params['sigma'])**2*factor)
        z = project_path(_mean_gaussian_smoother_1d(filt), F, Q)
        if finite(z):
            return z
    raise ValueError('No finite deterministic path proposal: inspect fixed parameters and observation support.')


def laplace_mh(y, observation, params, F, Q, H, offset, current, controls, rng):
    if H.shape[1] == 0:
        return current, {}
    inverse = np.linalg.pinv(Q, hermitian=True)
    def eta(z):
        return offset+np.sum(H*z[1:], axis=1)
    def log_likelihood(z):
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            value = float(np.sum(observation.logpdf(y, eta(z), params)))
        return value if np.isfinite(value) else -np.inf
    def objective(z):
        innovations = z[1:]-z[:-1]@F.T
        return log_likelihood(z)-.5*np.einsum('ti,ij,tj->', innovations, inverse, innovations)
    mode = finite_start(y, observation, params, F, Q, H, offset)
    value = objective(mode)
    converged = False
    for iteration in range(1, controls.max_iterations+1):
        pseudo, variance = pseudo_observations(y, eta(mode), observation, params, controls)
        filt = filter_gaussian(pseudo-offset, F, Q, H, variance)
        candidate = project_path(_mean_gaussian_smoother_1d(filt), F, Q)
        step = 1.
        improved = False
        for _ in range(35):
            next_mode = mode+step*(candidate-mode)
            next_value = objective(next_mode)
            if np.isfinite(next_value) and next_value >= value-1e-10:
                improved = True
                break
            step *= .5
        if not improved:
            break
        change = np.max(np.abs(eta(next_mode)-eta(mode)))
        mode, value = next_mode, next_value
        if change <= controls.tolerance*(1+np.max(np.abs(eta(mode)))):
            converged = True
            break
    pseudo, variance = pseudo_observations(y, eta(mode), observation, params, controls)
    filt = filter_gaussian(pseudo-offset, F, Q, H, variance)
    def correction(z):
        exact = log_likelihood(z)
        return exact+.5*np.sum((pseudo-eta(z))**2/variance) if np.isfinite(exact) else -np.inf
    accepted, support = 0, 0
    score = correction(current)
    if not np.isfinite(score):
        raise FloatingPointError('Current path has nonfinite exact target.')
    for _ in range(controls.mh_steps):
        proposed = project_path(_draw_gaussian_smoother_1d(filt, rng), F, Q)
        proposed_score = correction(proposed)
        if not np.isfinite(proposed_score):
            support += 1
        elif np.log(rng.random()) < min(0., proposed_score-score):
            current, score = proposed, proposed_score
            accepted += 1
    return current, dict(path_acceptance=accepted/controls.mh_steps,
        path_support_rejections=support, laplace_iterations=iteration, laplace_converged=float(converged))
