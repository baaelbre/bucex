"""Exact likelihood updates for a private secular log-observation-scale path."""
import numpy as np

from .fs_utils import _slice_sample_real


def gaussian_path_slice(current, direction, log_likelihood, rng):
    """Elliptical slice using a Gaussian-prior direction, without a dense root."""
    target = float(log_likelihood(current))
    if not np.isfinite(target):
        raise FloatingPointError("Current scale path is outside likelihood support.")
    height = target + np.log(rng.random())
    angle = rng.uniform(0, 2*np.pi)
    lo, hi = angle-2*np.pi, angle
    for evaluations in range(1, 10001):
        proposed = current*np.cos(angle) + direction*np.sin(angle)
        if log_likelihood(proposed) >= height:
            return proposed, evaluations
        if angle < 0:
            lo = angle
        else:
            hi = angle
        angle = rng.uniform(lo, hi)
    raise FloatingPointError("Scale-path elliptical slice exhausted its bracket budget.")


def secular_scale_step(state, specification, likelihood, rng):
    """Update secular parameters with the exact conditional copula likelihood."""
    mode = getattr(specification, "mode", "constant")
    n_time = len(state.y)
    if mode == "constant":
        return {}
    if mode == "structural":
        from .evolution import structural_scale_step
        return structural_scale_step(state, specification, likelihood, rng)
    metric = {}
    if mode == "linear":
        basis = np.arange(1, n_time+1)/specification.time_unit
        current = state.params_obs.get("scale_slope", 0.)
        value, evaluations = _slice_sample_real(current,
            lambda b: -.5*(b/specification.slope_sd)**2 + likelihood(b*basis),
            rng, width=specification.slope_sd)
        state.params_obs.update(scale_slope=value, log_scale_offset=value*basis)
        return {"scale_slope_slice_evaluations": evaluations}
    path = np.asarray(state.params_obs.get("scale_z", np.zeros(n_time)))
    signed_sd = state.params_obs.get("scale_signed_sd", specification.innovation_sd/2)
    path, evaluations = gaussian_path_slice(path, np.cumsum(rng.normal(size=n_time)),
        lambda z: likelihood(signed_sd*z), rng)
    metric["scale_path_slice_evaluations"] = evaluations
    signed_sd, evaluations = _slice_sample_real(signed_sd,
        lambda s: -.5*(s/specification.innovation_sd)**2 + likelihood(s*path),
        rng, width=specification.innovation_sd)
    metric["scale_innovation_slice_evaluations"] = evaluations
    if rng.random() < .5:
        signed_sd, path = -signed_sd, -path
    state.params_obs.update(scale_z=path, scale_signed_sd=signed_sd,
                            log_scale_offset=signed_sd*path)
    return metric
