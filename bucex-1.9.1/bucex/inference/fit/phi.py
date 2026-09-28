"""Conditional inference for the GEV log-scale process.

The structural state still describes the location predictor.  This module is
deliberately separate: it owns only ``phi_t = log(sigma_t)`` and can therefore
grow new scale models without expanding the public :func:`bucex.fit` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ...models.base import StateSpaceModel
from ...priors.structural import FSGEVPriors
from .fs_utils import ffbs_gaussian_1d_tvR, gaussian_smoother_mean_1d_tvR


Array = np.ndarray
PHI_MODE_NAMES = ("stationary", "linear", "rw")
PHI_MODE_CODES = {name: index for index, name in enumerate(PHI_MODE_NAMES)}


def phi_time_basis(n_time: int) -> Array:
    """Centered basis whose coefficient is the end/start log-scale change."""

    n_time = int(n_time)
    if n_time < 1:
        raise ValueError("n_time must be positive.")
    if n_time == 1:
        return np.zeros(1, dtype=float)
    return (np.arange(n_time, dtype=float) - 0.5 * (n_time - 1)) / (n_time - 1)


def phi_future_basis(n_time: int, horizon: int) -> Array:
    """Continue the fitted centered linear-scale basis into the future."""

    n_time, horizon = int(n_time), int(horizon)
    if n_time < 1 or horizon < 1:
        raise ValueError("n_time and horizon must be positive.")
    if n_time == 1:
        return np.arange(1, horizon + 1, dtype=float)
    time = np.arange(n_time, n_time + horizon, dtype=float)
    return (time - 0.5 * (n_time - 1)) / (n_time - 1)


def _normal_logpdf(value: float, mean: float, sd: float) -> float:
    standardized = (float(value) - float(mean)) / float(sd)
    return float(-0.5 * standardized**2 - np.log(float(sd)) - 0.5 * np.log(2.0 * np.pi))


def _copy_observation(params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (
            np.asarray(value, dtype=float).copy()
            if isinstance(value, np.ndarray)
            else value
        )
        for key, value in params.items()
    }


@dataclass
class PhiState:
    """All scale-model coordinates used by fixed modes and product-space SSVS."""

    mode: str
    stationary: float
    linear_intercept: float
    linear_slope: float
    rw_path: Array
    rw_variance: float

    def copy(self) -> "PhiState":
        return PhiState(
            mode=str(self.mode),
            stationary=float(self.stationary),
            linear_intercept=float(self.linear_intercept),
            linear_slope=float(self.linear_slope),
            rw_path=np.asarray(self.rw_path, dtype=float).copy(),
            rw_variance=float(self.rw_variance),
        )

    def path(self, n_time: int) -> Array:
        if self.mode == "stationary":
            return np.full(int(n_time), float(self.stationary), dtype=float)
        if self.mode == "linear":
            return float(self.linear_intercept) + float(
                self.linear_slope
            ) * phi_time_basis(n_time)
        if self.mode == "rw":
            path = np.asarray(self.rw_path, dtype=float)
            if path.shape != (int(n_time) + 1,):
                raise ValueError("The random-walk phi path must have length T+1.")
            return path[1:].copy()
        raise ValueError(f"Unknown active phi mode {self.mode!r}.")

    def reference(self) -> float:
        if self.mode == "stationary":
            return float(self.stationary)
        if self.mode == "linear":
            return float(self.linear_intercept)
        if self.mode == "rw":
            return float(np.asarray(self.rw_path, dtype=float)[0])
        raise ValueError(f"Unknown active phi mode {self.mode!r}.")


@dataclass(frozen=True)
class PhiLaplaceApproximation:
    mode_path: Array
    pseudo_y: Array
    pseudo_variance: Array
    prior_mean: float
    prior_variance: float
    rw_variance: float
    converged: bool
    iterations: int
    relative_change: float
    objective: float


@dataclass(frozen=True)
class PhiUpdate:
    state: PhiState
    params_observation: dict[str, Any]
    accepted: dict[str, bool]
    diagnostics: dict[str, float]


class PhiKernel:
    """One conditional update for stationary, linear, RW, or SSVS log scale."""

    def __init__(
        self,
        model: StateSpaceModel,
        priors: FSGEVPriors,
        rng: np.random.Generator,
    ) -> None:
        self.model = model
        self.priors = priors
        self.rng = rng
        self.specification = str(getattr(model.obs, "phi", "stationary"))

    def _intercept_logprior(self, value: float) -> float:
        if not np.isfinite(value):
            return -np.inf
        if self.priors.sigma2 is not None:
            prior = self.priors.sigma2
            return float(-2.0 * prior.a * value - prior.b * np.exp(-2.0 * value))
        assert self.priors.log_sigma is not None
        prior = self.priors.log_sigma
        return _normal_logpdf(value, prior.mean, prior.sd)

    def _intercept_gaussian_approximation(self) -> tuple[float, float]:
        if self.priors.log_sigma is not None:
            return (
                float(self.priors.log_sigma.mean),
                float(self.priors.log_sigma.sd) ** 2,
            )
        assert self.priors.sigma2 is not None
        prior = self.priors.sigma2
        return 0.5 * float(np.log(prior.b / prior.a)), 1.0 / (4.0 * prior.a)

    def _sample_intercept_prior(self) -> float:
        if self.priors.log_sigma is not None:
            prior = self.priors.log_sigma
            return float(self.rng.normal(prior.mean, prior.sd))
        assert self.priors.sigma2 is not None
        prior = self.priors.sigma2
        precision = float(self.rng.gamma(prior.a, 1.0 / prior.b))
        return 0.5 * float(np.log(1.0 / precision))

    def _sample_rw_variance_prior(self) -> float:
        prior = self.priors.phi.rw_variance
        precision = float(self.rng.gamma(prior.a, 1.0 / prior.b))
        return float(1.0 / precision)

    def _sample_rw_prior(self, n_time: int) -> tuple[Array, float]:
        variance = self._sample_rw_variance_prior()
        path = np.empty(int(n_time) + 1, dtype=float)
        path[0] = self._sample_intercept_prior()
        path[1:] = path[0] + np.cumsum(
            self.rng.normal(scale=np.sqrt(variance), size=int(n_time))
        )
        return path, variance

    def initialise(
        self,
        n_time: int,
        params_observation: Mapping[str, Any],
    ) -> PhiState:
        """Resolve old scalar starts and new explicit phi warm starts."""

        sigma = np.asarray(params_observation["sigma"], dtype=float)
        if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError("Initial sigma must be finite and positive.")
        reference = float(np.mean(np.log(sigma)))
        stationary = float(params_observation.get("phi_stationary", reference))
        linear_intercept = float(params_observation.get("phi_intercept", reference))
        linear_slope = float(params_observation.get("phi_slope", 0.0))
        prior = self.priors.phi.rw_variance
        default_variance = float(prior.b / (prior.a + 1.0))
        rw_variance = float(
            params_observation.get(
                "phi_rw_variance",
                float(params_observation.get("phi_rw_sd", np.sqrt(default_variance)))
                ** 2,
            )
        )
        if not np.isfinite(rw_variance) or rw_variance <= 0.0:
            raise ValueError("Initial phi_rw_variance must be finite and positive.")
        supplied_path = params_observation.get(
            "phi_rw_path", params_observation.get("phi_path")
        )
        if supplied_path is None:
            rw_path = np.full(int(n_time) + 1, reference, dtype=float)
        else:
            supplied = np.asarray(supplied_path, dtype=float).reshape(-1)
            if supplied.size == int(n_time):
                rw_path = np.r_[supplied[0], supplied]
            elif supplied.size == int(n_time) + 1:
                rw_path = supplied.copy()
            else:
                raise ValueError("Initial phi path must have length T or T+1.")
        if np.any(~np.isfinite(rw_path)):
            raise ValueError("Initial phi path must be finite.")

        if self.specification == "ssvs":
            requested = params_observation.get(
                "phi_mode", params_observation.get("phi_model", "stationary")
            )
            if isinstance(requested, (int, np.integer)):
                try:
                    mode = PHI_MODE_NAMES[int(requested)]
                except IndexError as error:
                    raise ValueError(
                        "Initial phi_model code must be 0, 1, or 2."
                    ) from error
            else:
                mode = str(requested).lower().replace("-", "_")
                mode = "rw" if mode in {"random_walk", "randomwalk"} else mode
            if mode not in PHI_MODE_NAMES:
                raise ValueError("Initial phi_mode must be stationary, linear, or rw.")
        else:
            mode = self.specification
        return PhiState(
            mode=mode,
            stationary=stationary,
            linear_intercept=linear_intercept,
            linear_slope=linear_slope,
            rw_path=rw_path,
            rw_variance=rw_variance,
        )

    def observation_parameters(
        self,
        state: PhiState,
        params_observation: Mapping[str, Any],
        n_time: int,
    ) -> dict[str, Any]:
        output = _copy_observation(params_observation)
        phi = state.path(n_time)
        if np.any(~np.isfinite(phi)) or np.any(np.abs(phi) > 50.0):
            output["sigma"] = np.full(int(n_time), np.nan)
        elif state.mode == "stationary":
            output["sigma"] = float(np.exp(phi[0]))
        else:
            output["sigma"] = np.exp(phi)
        output["sigma_reference"] = float(np.exp(state.reference()))
        return output

    def _loglik(
        self,
        y: Array,
        mu: Array,
        phi: Array,
        params_observation: Mapping[str, Any],
    ) -> float:
        phi = np.asarray(phi, dtype=float).reshape(-1)
        if phi.size != np.asarray(y).size or np.any(~np.isfinite(phi)):
            return -np.inf
        if np.any(np.abs(phi) > 50.0):
            return -np.inf
        try:
            values = np.asarray(
                self.model.obs.logpdf(
                    np.asarray(y, dtype=float),
                    np.asarray(mu, dtype=float),
                    sigma=np.exp(phi),
                    xi=float(params_observation["xi"]),
                ),
                dtype=float,
            )
        except Exception:
            return -np.inf
        if values.shape != np.asarray(y).shape:
            try:
                values = np.broadcast_to(values, np.asarray(y).shape)
            except ValueError:
                return -np.inf
        return float(np.sum(values)) if np.all(np.isfinite(values)) else -np.inf

    def _stationary_logtarget(
        self,
        value: float,
        y: Array,
        mu: Array,
        params_observation: Mapping[str, Any],
    ) -> float:
        return self._loglik(
            y, mu, np.full(np.asarray(y).size, value), params_observation
        ) + self._intercept_logprior(value)

    def _linear_logtarget(
        self,
        intercept: float,
        slope: float,
        y: Array,
        mu: Array,
        params_observation: Mapping[str, Any],
    ) -> float:
        slope_prior = self.priors.phi.linear
        phi = intercept + slope * phi_time_basis(np.asarray(y).size)
        return (
            self._loglik(y, mu, phi, params_observation)
            + self._intercept_logprior(intercept)
            + _normal_logpdf(slope, slope_prior.mean, slope_prior.sd)
        )

    def _update_stationary(
        self,
        state: PhiState,
        y: Array,
        mu: Array,
        params_observation: Mapping[str, Any],
        *,
        step: float,
    ) -> tuple[PhiState, bool]:
        output = state.copy()
        current = float(output.stationary)
        proposal = current + float(self.rng.normal(scale=step))
        current_log = self._stationary_logtarget(current, y, mu, params_observation)
        proposal_log = self._stationary_logtarget(proposal, y, mu, params_observation)
        accepted = bool(
            np.isfinite(proposal_log)
            and np.log(self.rng.random()) < min(0.0, proposal_log - current_log)
        )
        if accepted:
            output.stationary = proposal
        return output, accepted

    def _update_linear(
        self,
        state: PhiState,
        y: Array,
        mu: Array,
        params_observation: Mapping[str, Any],
        *,
        intercept_step: float,
        slope_step: float,
    ) -> tuple[PhiState, bool, bool]:
        output = state.copy()
        current_log = self._linear_logtarget(
            output.linear_intercept,
            output.linear_slope,
            y,
            mu,
            params_observation,
        )
        proposal = output.linear_intercept + float(
            self.rng.normal(scale=intercept_step)
        )
        proposal_log = self._linear_logtarget(
            proposal, output.linear_slope, y, mu, params_observation
        )
        intercept_accepted = bool(
            np.isfinite(proposal_log)
            and np.log(self.rng.random()) < min(0.0, proposal_log - current_log)
        )
        if intercept_accepted:
            output.linear_intercept, current_log = proposal, proposal_log

        proposal = output.linear_slope + float(self.rng.normal(scale=slope_step))
        proposal_log = self._linear_logtarget(
            output.linear_intercept, proposal, y, mu, params_observation
        )
        slope_accepted = bool(
            np.isfinite(proposal_log)
            and np.log(self.rng.random()) < min(0.0, proposal_log - current_log)
        )
        if slope_accepted:
            output.linear_slope = proposal
        return output, intercept_accepted, slope_accepted

    def _rw_logprior(self, path: Array, variance: float) -> float:
        path = np.asarray(path, dtype=float).reshape(-1)
        if variance <= 0.0 or np.any(~np.isfinite(path)):
            return -np.inf
        differences = np.diff(path)
        return float(
            self._intercept_logprior(path[0])
            - 0.5 * differences.size * np.log(2.0 * np.pi * variance)
            - 0.5 * np.sum(differences**2) / variance
        )

    def _deterministic_rw_start(
        self,
        y: Array,
        mu: Array,
        xi: float,
        prior_mean: float,
        *,
        minimum_support: float = 0.25,
    ) -> Array:
        product = float(xi) * (np.asarray(y, dtype=float) - np.asarray(mu, dtype=float))
        observed = np.full(np.asarray(y).size, float(prior_mean), dtype=float)
        constrained = product < 0.0
        if np.any(constrained):
            required = -product[constrained] / (1.0 - float(minimum_support))
            observed[constrained] = np.maximum(
                observed[constrained], np.log(np.maximum(required, 1e-300))
            )
        path = np.r_[float(prior_mean), observed]
        for _ in range(20):
            if np.isfinite(self._loglik(y, mu, path[1:], {"xi": float(xi)})):
                return path
            path[1:] += np.log(2.0)
        raise FloatingPointError(
            "Could not initialize the log-scale random walk inside GEV support."
        )

    def _rw_pseudo_data(
        self,
        y: Array,
        mu: Array,
        phi: Array,
        xi: float,
        *,
        curvature_floor: float,
        maximum_variance: float,
        shift_limit: float,
    ) -> tuple[Array, Array]:
        sigma = np.exp(np.asarray(phi, dtype=float))
        try:
            gradient = np.asarray(
                self.model.obs.grad_phi(y, mu, sigma=sigma, xi=xi), dtype=float
            )
            hessian = np.asarray(
                self.model.obs.hess_phi(y, mu, sigma=sigma, xi=xi), dtype=float
            )
            gradient = np.broadcast_to(gradient, np.asarray(y).shape)
            hessian = np.broadcast_to(hessian, np.asarray(y).shape)
        except Exception as error:
            raise FloatingPointError(
                "Invalid GEV log-scale derivatives at the Laplace mode."
            ) from error
        if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(hessian)):
            raise FloatingPointError(
                "Invalid GEV log-scale derivatives at the Laplace mode."
            )
        information = np.maximum(-hessian, float(curvature_floor))
        variance = np.minimum(1.0 / information, float(maximum_variance))
        pseudo = np.asarray(phi, dtype=float) + np.clip(
            gradient / information, -float(shift_limit), float(shift_limit)
        )
        return pseudo, variance

    def _build_rw_approximation(
        self,
        y: Array,
        mu: Array,
        xi: float,
        variance: float,
        *,
        max_iterations: int,
        tolerance: float,
        curvature_floor: float,
        maximum_variance: float,
        shift_limit: float,
    ) -> PhiLaplaceApproximation:
        prior_mean, prior_variance = self._intercept_gaussian_approximation()
        mode = self._deterministic_rw_start(y, mu, xi, prior_mean)
        transition = np.ones((1, 1), dtype=float)
        process = np.asarray([[float(variance)]], dtype=float)
        measurement = np.ones(1, dtype=float)

        def objective(path: Array) -> float:
            return self._loglik(y, mu, path[1:], {"xi": xi}) + self._rw_logprior(
                path, variance
            )

        current_objective = objective(mode)
        if not np.isfinite(current_objective):
            raise FloatingPointError(
                "Could not initialize the GEV log-scale Laplace mode."
            )
        converged = False
        relative_change = np.inf
        pseudo_y = mode[1:].copy()
        pseudo_variance = np.ones(np.asarray(y).size, dtype=float)
        iteration = 0
        for iteration in range(1, int(max_iterations) + 1):
            pseudo_y, pseudo_variance = self._rw_pseudo_data(
                y,
                mu,
                mode[1:],
                xi,
                curvature_floor=curvature_floor,
                maximum_variance=maximum_variance,
                shift_limit=shift_limit,
            )
            candidate = gaussian_smoother_mean_1d_tvR(
                pseudo_y,
                transition,
                process,
                measurement,
                pseudo_variance,
                m0=np.asarray([prior_mean]),
                C0=np.asarray([[prior_variance]]),
            )[:, 0]
            accepted = False
            step_size = 1.0
            proposal = mode
            candidate_objective = -np.inf
            while step_size >= 2.0**-12:
                proposal = mode + step_size * (candidate - mode)
                candidate_objective = objective(proposal)
                if (
                    np.isfinite(candidate_objective)
                    and candidate_objective >= current_objective - 1e-8
                ):
                    accepted = True
                    break
                step_size *= 0.5
            if not accepted:
                relative_change = 0.0
                converged = True
                break
            previous = mode
            mode = proposal
            current_objective = candidate_objective
            relative_change = float(
                np.max(np.abs(mode[1:] - previous[1:]))
                / (1.0 + np.max(np.abs(previous[1:])))
            )
            if relative_change < float(tolerance):
                converged = True
                break

        pseudo_y, pseudo_variance = self._rw_pseudo_data(
            y,
            mu,
            mode[1:],
            xi,
            curvature_floor=curvature_floor,
            maximum_variance=maximum_variance,
            shift_limit=shift_limit,
        )
        return PhiLaplaceApproximation(
            mode_path=mode,
            pseudo_y=pseudo_y,
            pseudo_variance=pseudo_variance,
            prior_mean=float(prior_mean),
            prior_variance=float(prior_variance),
            rw_variance=float(variance),
            converged=bool(converged),
            iterations=int(iteration),
            relative_change=float(relative_change),
            objective=float(current_objective),
        )

    def _draw_rw_proposal(self, approximation: PhiLaplaceApproximation) -> Array:
        path = ffbs_gaussian_1d_tvR(
            approximation.pseudo_y,
            np.ones((1, 1), dtype=float),
            np.asarray([[approximation.rw_variance]], dtype=float),
            np.ones(1, dtype=float),
            approximation.pseudo_variance,
            m0=np.asarray([approximation.prior_mean]),
            C0=np.asarray([[approximation.prior_variance]]),
            rng=self.rng,
        )
        return path[:, 0]

    @staticmethod
    def _pseudo_loglik(path: Array, approximation: PhiLaplaceApproximation) -> float:
        residual = approximation.pseudo_y - np.asarray(path, dtype=float)[1:]
        variance = approximation.pseudo_variance
        return float(
            -0.5 * np.sum(np.log(2.0 * np.pi * variance) + residual**2 / variance)
        )

    def _rw_log_weight(
        self,
        path: Array,
        y: Array,
        mu: Array,
        xi: float,
        approximation: PhiLaplaceApproximation,
    ) -> float:
        exact = self._loglik(y, mu, np.asarray(path)[1:], {"xi": xi})
        if not np.isfinite(exact):
            return -np.inf
        target_initial = self._intercept_logprior(float(path[0]))
        proposal_initial = _normal_logpdf(
            float(path[0]),
            approximation.prior_mean,
            np.sqrt(approximation.prior_variance),
        )
        return float(
            exact
            + target_initial
            - self._pseudo_loglik(path, approximation)
            - proposal_initial
        )

    def _update_rw_path(
        self,
        state: PhiState,
        y: Array,
        mu: Array,
        params_observation: Mapping[str, Any],
        *,
        exact: bool,
        options: Mapping[str, Any],
    ) -> tuple[PhiState, bool, PhiLaplaceApproximation, float, int]:
        output = state.copy()
        approximation = self._build_rw_approximation(
            y,
            mu,
            float(params_observation["xi"]),
            output.rw_variance,
            max_iterations=int(options.get("phi_laplace_max_iterations", 30)),
            tolerance=float(options.get("phi_laplace_tolerance", 1e-5)),
            curvature_floor=float(options.get("phi_curvature_floor", 1e-5)),
            maximum_variance=float(options.get("phi_maximum_variance", 1e5)),
            shift_limit=float(options.get("phi_shift_limit", 2.0)),
        )
        if exact:
            current_weight = self._rw_log_weight(
                output.rw_path,
                y,
                mu,
                float(params_observation["xi"]),
                approximation,
            )
            if not np.isfinite(current_weight):
                raise ValueError("The current log-scale path is outside GEV support.")
            accepted_any = False
            support_failures = 0
            steps = int(options.get("phi_laplace_mh_steps", 1))
            if steps < 1:
                raise ValueError("phi_laplace_mh_steps must be positive.")
            for _ in range(steps):
                proposal = self._draw_rw_proposal(approximation)
                proposal_weight = self._rw_log_weight(
                    proposal,
                    y,
                    mu,
                    float(params_observation["xi"]),
                    approximation,
                )
                if not np.isfinite(proposal_weight):
                    support_failures += 1
                    continue
                if np.log(self.rng.random()) < min(
                    0.0, proposal_weight - current_weight
                ):
                    output.rw_path = proposal
                    current_weight = proposal_weight
                    accepted_any = True
            return output, accepted_any, approximation, current_weight, support_failures

        accepted = False
        support_failures = 0
        attempts = int(options.get("phi_draw_attempts", 30))
        for _ in range(max(attempts, 1)):
            proposal = self._draw_rw_proposal(approximation)
            if np.isfinite(
                self._loglik(
                    y,
                    mu,
                    proposal[1:],
                    {"xi": float(params_observation["xi"])},
                )
            ):
                output.rw_path = proposal
                accepted = True
                break
            support_failures += 1
        if not accepted:
            output.rw_path = approximation.mode_path.copy()
        return output, accepted, approximation, np.nan, support_failures

    def _update_rw_variance(self, state: PhiState) -> PhiState:
        output = state.copy()
        prior = self.priors.phi.rw_variance
        differences = np.diff(output.rw_path)
        shape = prior.a + 0.5 * differences.size
        rate = prior.b + 0.5 * float(np.sum(differences**2))
        precision = float(self.rng.gamma(shape, 1.0 / rate))
        output.rw_variance = float(1.0 / precision)
        return output

    def _update_rw(
        self,
        state: PhiState,
        y: Array,
        mu: Array,
        params_observation: Mapping[str, Any],
        *,
        exact: bool,
        options: Mapping[str, Any],
    ) -> tuple[PhiState, bool, dict[str, float]]:
        output, accepted, approximation, weight, support_failures = (
            self._update_rw_path(
                state,
                y,
                mu,
                params_observation,
                exact=exact,
                options=options,
            )
        )
        output = self._update_rw_variance(output)
        return (
            output,
            accepted,
            {
                "phi_laplace_iterations": float(approximation.iterations),
                "phi_laplace_converged": float(approximation.converged),
                "phi_laplace_relative_change": float(approximation.relative_change),
                "phi_laplace_mh_accepted": float(accepted) if exact else np.nan,
                "phi_laplace_mh_log_weight": float(weight),
                "phi_laplace_support_rejections": float(support_failures),
            },
        )

    def _refresh_inactive(self, state: PhiState, n_time: int) -> PhiState:
        output = state.copy()
        if output.mode != "stationary":
            output.stationary = self._sample_intercept_prior()
        if output.mode != "linear":
            output.linear_intercept = self._sample_intercept_prior()
            prior = self.priors.phi.linear
            output.linear_slope = float(self.rng.normal(prior.mean, prior.sd))
        if output.mode != "rw":
            output.rw_path, output.rw_variance = self._sample_rw_prior(n_time)
        return output

    def _sample_mode(
        self,
        state: PhiState,
        y: Array,
        mu: Array,
        params_observation: Mapping[str, Any],
    ) -> tuple[PhiState, bool, Array]:
        output = state.copy()
        paths = (
            np.full(np.asarray(y).size, output.stationary),
            output.linear_intercept
            + output.linear_slope * phi_time_basis(np.asarray(y).size),
            output.rw_path[1:],
        )
        probabilities = np.asarray(self.priors.phi.model_probabilities, dtype=float)
        log_weights = np.asarray(
            [
                (-np.inf if probability <= 0.0 else np.log(probability))
                + self._loglik(y, mu, path, params_observation)
                for probability, path in zip(probabilities, paths)
            ],
            dtype=float,
        )
        finite = np.isfinite(log_weights)
        if not np.any(finite):
            raise FloatingPointError("Every phi SSVS candidate is outside GEV support.")
        normalized = np.zeros(3, dtype=float)
        maximum = float(np.max(log_weights[finite]))
        normalized[finite] = np.exp(log_weights[finite] - maximum)
        normalized /= float(np.sum(normalized))
        previous = output.mode
        output.mode = PHI_MODE_NAMES[int(self.rng.choice(3, p=normalized))]
        return output, output.mode != previous, normalized

    def update(
        self,
        y: Array,
        mu: Array,
        state: PhiState,
        params_observation: Mapping[str, Any],
        *,
        state_method: str,
        options: Mapping[str, Any],
    ) -> PhiUpdate:
        """Update the active scale model and return observation-ready sigma."""

        exact = str(state_method) == "laplace_mh"
        output = state.copy()
        accepted: dict[str, bool] = {}
        diagnostics = {
            "phi_laplace_iterations": np.nan,
            "phi_laplace_converged": np.nan,
            "phi_laplace_relative_change": np.nan,
            "phi_laplace_mh_accepted": np.nan,
            "phi_laplace_mh_log_weight": np.nan,
            "phi_laplace_support_rejections": np.nan,
            "phi_model_switched": np.nan,
            "phi_probability_stationary": np.nan,
            "phi_probability_linear": np.nan,
            "phi_probability_rw": np.nan,
        }
        intercept_step = float(options.get("phi_step_intercept", 0.04))
        slope_step = float(options.get("phi_step_linear", 0.05))
        if intercept_step <= 0.0 or slope_step <= 0.0:
            raise ValueError("phi MH step sizes must be positive.")

        if self.specification == "ssvs":
            output = self._refresh_inactive(output, np.asarray(y).size)
            output, switched, probabilities = self._sample_mode(
                output, y, mu, params_observation
            )
            accepted["phi_model_switch"] = bool(switched)
            diagnostics["phi_model_switched"] = float(switched)
            for name, probability in zip(PHI_MODE_NAMES, probabilities):
                diagnostics[f"phi_probability_{name}"] = float(probability)

        if output.mode == "stationary":
            output, accepted_intercept = self._update_stationary(
                output,
                y,
                mu,
                params_observation,
                step=intercept_step,
            )
            accepted["phi_intercept_mh"] = accepted_intercept
        elif output.mode == "linear":
            output, accepted_intercept, accepted_slope = self._update_linear(
                output,
                y,
                mu,
                params_observation,
                intercept_step=intercept_step,
                slope_step=slope_step,
            )
            accepted["phi_intercept_mh"] = accepted_intercept
            accepted["phi_linear_mh"] = accepted_slope
        elif output.mode == "rw":
            output, accepted_rw, rw_diagnostics = self._update_rw(
                output,
                y,
                mu,
                params_observation,
                exact=exact,
                options=options,
            )
            accepted["phi_rw_laplace_mh" if exact else "phi_rw_laplace_draw"] = (
                accepted_rw
            )
            diagnostics.update(rw_diagnostics)
        else:
            raise ValueError(f"Unknown active phi mode {output.mode!r}.")

        params = self.observation_parameters(
            output, params_observation, np.asarray(y).size
        )
        return PhiUpdate(
            state=output,
            params_observation=params,
            accepted=accepted,
            diagnostics=diagnostics,
        )


__all__ = [
    "PHI_MODE_CODES",
    "PHI_MODE_NAMES",
    "PhiKernel",
    "PhiLaplaceApproximation",
    "PhiState",
    "PhiUpdate",
    "phi_future_basis",
    "phi_time_basis",
]
