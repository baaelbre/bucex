"""Gaussian structural evolution against an arbitrary scalar log likelihood.

The likelihood callback includes the observation family and, when present,
the conditional copula density. This kernel has no temperature, Uccle, GEV,
or scale-specific likelihood algebra. Its initial level is anchored at zero;
the caller owns the distribution parameter's overall intercept.
"""
import numpy as np

from . import fs_utils as fs
from .continuous import ShrinkageState, coefficient_prior, reference_slice, sign_step
from .scale_path import gaussian_path_slice


class GaussianEvolutionState:
    def __init__(self, specification, n_time, saved=None):
        from ...models import Model
        from ...observation import Gaussian
        self.model = Model(Gaussian(), specification.components)
        self.layout = fs.infer_ncp_layout(self.model)
        self.prior = specification.priors.resolve(self.model.period)
        self.params_state = dict(alpha0=0., beta0=0.,
            gamma0_season=np.zeros(self.layout.season_dim))
        active = fs._active_scale_names(self.layout)
        for key in ("level", "trend", "season"):
            value = specification.priors.innovation_median[key] if key in active else 0.
            self.params_state["s_"+key], self.params_state["q_"+key] = value, value**2
        self.params_obs = {"sigma2": 1.}  # evolution priors are in link units
        self.z_path = np.zeros((n_time+1, self.layout.ncp_state_dim))
        self.mixing = ShrinkageState.initialize(self.prior, self.layout)
        self.G, Q = fs.build_ncp_system(self.layout)
        self.root = np.diag(np.sqrt(np.diag(Q)))
        _, names, _ = self.design()
        self.names = names
        if saved and "evolution.z" in saved:
            self.z_path = np.asarray(saved["evolution.z"]).copy()
            if self.z_path.shape != (n_time+1, self.layout.ncp_state_dim):
                raise ValueError("Warm-start evolution has incompatible state dimensions.")
            self.params_state = fs.apply_theta_draw(self.params_state,
                saved["evolution.theta"], self.names, self.layout)
            restored = {k.removeprefix("evolution.mixing."): v for k,v in saved.items()
                        if k.startswith("evolution.mixing.")}
            self.mixing.restore(restored, "parameter")

    def design(self):
        X, names, _ = fs.design_matrix_ncp(self.z_path, self.layout, center_time=False)
        active = {"s_"+k for k in fs._active_scale_names(self.layout)}
        selected = [i for i,n in enumerate(names) if n != "alpha0" and
                    (not n.startswith("s_") or n in active)]
        return X[:,selected], [names[i] for i in selected], 0.

    def predictor(self):
        return fs.mu_from_ncp(self.z_path, self.params_state, self.layout)

    def prior_direction(self, rng):
        direction = np.zeros_like(self.z_path)
        innovations = rng.normal(size=direction.shape) @ self.root.T
        for t in range(1, len(direction)):
            direction[t] = self.G @ direction[t-1] + innovations[t]
        return direction

    def step(self, likelihood, rng):
        self.z_path, path_evals = gaussian_path_slice(self.z_path, self.prior_direction(rng),
            lambda z: likelihood(fs.mu_from_ncp(z, self.params_state, self.layout)), rng)
        X, names, _ = self.design()
        coefficient_evals = 0
        if names:
            mean, root = coefficient_prior(self.prior, names, 0., self.mixing, 1.)
            current = fs.theta_vector_from_params(self.params_state, names, self.layout, tbar=0.)
            theta, coefficient_evals = reference_slice(current, mean, root,
                lambda value: likelihood(X @ value), rng)
            self.params_state = fs.apply_theta_draw(self.params_state, theta, names, self.layout)
        sign_step(self, self.prior, self.mixing, rng)
        self.mixing.update(self, self.prior, rng)
        return {"evolution_path_slice_evaluations": path_evals,
                "evolution_coefficient_slice_evaluations": coefficient_evals}

    def values(self):
        result = {"evolution.z": self.z_path.copy(),
                  "evolution.theta": fs.theta_vector_from_params(self.params_state, self.names, self.layout, tbar=0.),
                  "evolution.state": fs.map_ncp_to_centered(self.z_path, self.params_state, self.layout)}
        for key, legacy in (("level", "level"), ("slope", "trend"), ("seasonal", "season")):
            if legacy in fs._active_scale_names(self.layout):
                result["evolution.sd."+key] = abs(self.params_state["s_"+legacy])
        if self.layout.has_beta:
            result["evolution.initial.slope"] = self.params_state["beta0"]
        result.update({"evolution.mixing."+k:v for k,v in self.mixing.parameter_values("parameter").items()})
        return result


def structural_scale_step(state, specification, likelihood, rng):
    evolution = getattr(state, "scale_evolution", None)
    if evolution is None:
        evolution = GaussianEvolutionState(specification, len(state.y), state.params_obs)
        state.scale_evolution = evolution
    metrics = evolution.step(likelihood, rng)
    state.params_obs.update(evolution.values(), log_scale_offset=evolution.predictor())
    return metrics


def forecast_evolution(specification, fit, indices, horizon, rng, suffix=""):
    """Propagate the last centered state and draw every future innovation."""
    template = GaussianEvolutionState(specification, fit.n_time)
    theta = fit.parameter("evolution.theta"+suffix)[indices]
    last = fit.parameter("evolution.state"+suffix)[indices,-1]
    result = np.empty((len(indices), horizon))
    for d in range(len(indices)):
        params = fs.apply_theta_draw(template.params_state, theta[d], template.names, template.layout)
        system = template.model.system(1, params)
        root = system.R @ np.diag(np.sqrt(np.diag(system.Q)))
        design = template.model.design(1, params).Z[0]
        state = last[d].copy()
        for t in range(horizon):
            state = system.T @ state + system.c + root @ rng.normal(size=root.shape[1])
            result[d,t] = design @ state
    return result


def simulate_evolution(specification, n_time, params, rng, suffix=""):
    """Generate a declared evolution from physical SDs and initial coefficients."""
    state = GaussianEvolutionState(specification, n_time)
    for process, legacy in (("level", "level"), ("slope", "trend"), ("seasonal", "season")):
        if legacy not in fs._active_scale_names(state.layout):
            continue
        key = "scale.sd."+process+suffix
        if key not in params:
            raise ValueError(f"Structural scale simulation requires {key}.")
        value = float(params[key])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
        state.params_state["s_"+legacy] = value
        state.params_state["q_"+legacy] = value**2
    state.params_state["beta0"] = float(params.get("scale.initial.slope"+suffix, 0.))
    seasonal = np.asarray(params.get("scale.initial.seasonal"+suffix, np.zeros(state.layout.season_dim)))
    if seasonal.shape != (state.layout.season_dim,) or not np.all(np.isfinite(seasonal)):
        raise ValueError("scale.initial.seasonal must contain period-1 finite coefficients.")
    if not np.isfinite(state.params_state["beta0"]):
        raise ValueError("scale.initial.slope must be finite.")
    state.params_state["gamma0_season"] = seasonal
    state.z_path = state.prior_direction(rng)
    return state.predictor()
