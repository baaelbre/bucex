"""One convergence-checked iterated Laplace backend for GEV models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...models.compiler import CompiledModel
from .kalman import KalmanResult, ffbs, kalman_filter, kalman_smoother
from ...core.numerics import (
    factored_singular_normal_logpdf,
    gaussian_support,
    singular_normal_logpdf,
)


Array = np.ndarray


@dataclass(frozen=True)
class LaplaceApproximation:
    """Deterministic Gaussian approximation to a conditional state posterior.

    The object contains everything needed to draw repeatedly from the same
    approximation.  Keeping construction and simulation separate is essential
    for independence Metropolis--Hastings: the proposal must depend on the data
    and parameters, not on the current state trajectory.
    """

    mode_path: Array
    pseudo_y: Array
    pseudo_variance: Array
    converged: bool
    iterations: int
    relative_change: float
    objective: float
    filter: KalmanResult


@dataclass(frozen=True)
class LaplaceMHResult:
    """Result of one or more exact Laplace independence-MH state updates."""

    path: Array
    approximation: LaplaceApproximation
    accepted: Array
    log_acceptance_ratio: Array
    log_weight: float
    proposal_support_failures: int
    exact_invariant: bool = True

    @property
    def attempts(self) -> int:
        return int(np.asarray(self.accepted).size)

    @property
    def accepted_steps(self) -> int:
        return int(np.sum(np.asarray(self.accepted, dtype=bool)))

    @property
    def acceptance_rate(self) -> float:
        return float(self.accepted_steps / max(self.attempts, 1))


@dataclass
class LaplaceDraw:
    path: Array
    mode_path: Array
    pseudo_y: Array
    pseudo_variance: Array
    converged: bool
    iterations: int
    relative_change: float
    objective: float
    support_rejections: int
    filter: KalmanResult


def observation_log_likelihood(
    y: Array,
    eta: Array,
    compiled: CompiledModel,
    params: dict[str, float],
) -> float:
    if hasattr(compiled, "observation_log_likelihood"):
        return compiled.observation_log_likelihood(y, eta, params)
    mask = np.isfinite(y)
    if not np.any(mask):
        return 0.0
    values = compiled.model.observation.logpdf(
        np.asarray(y)[mask],
        np.asarray(eta)[mask],
        sigma=float(params["sigma"]),
        xi=params.get("xi"),
    )
    if not np.all(np.isfinite(values)):
        return -np.inf
    return float(np.sum(values))


def state_log_density(path: Array, compiled: CompiledModel, params: dict[str, float]) -> float:
    path = np.asarray(path, dtype=float)
    value = singular_normal_logpdf(path[0], compiled.initial_mean, compiled.initial_cov)
    covariance = compiled.transition_cov(params)
    means = path[:-1] @ compiled.transition.T
    transition_values = factored_singular_normal_logpdf(
        path[1:], means, gaussian_support(covariance)
    )
    if not np.all(np.isfinite(transition_values)):
        return -np.inf
    return float(value + np.sum(transition_values))


def joint_state_log_density(
    y: Array,
    path: Array,
    compiled: CompiledModel,
    params: dict[str, float],
) -> float:
    eta = compiled.eta(path, params=params)
    return state_log_density(path, compiled, params) + observation_log_likelihood(
        y, eta, compiled, params
    )


def _pseudo_data(
    y: Array,
    eta: Array,
    compiled: CompiledModel,
    params: dict[str, float],
    *,
    curvature_floor: float,
    maximum_variance: float,
    shift_limit: float,
) -> tuple[Array, Array]:
    y = np.asarray(y, dtype=float)
    pseudo_y = np.full(y.shape, np.nan)
    pseudo_variance = np.ones(y.shape)
    mask = np.isfinite(y)
    if not np.any(mask):
        return pseudo_y, pseudo_variance
    if hasattr(compiled, "observation_derivatives"):
        all_grad, all_hess = compiled.observation_derivatives(y, eta, params)
        grad = np.asarray(all_grad[mask], dtype=float)
        hess = np.asarray(all_hess[mask], dtype=float)
    else:
        obs = compiled.model.observation
        grad = np.asarray(
            obs.grad_eta(y[mask], eta[mask], sigma=params["sigma"], xi=params.get("xi")),
            dtype=float,
        )
        hess = np.asarray(
            obs.hess_eta(y[mask], eta[mask], sigma=params["sigma"], xi=params.get("xi")),
            dtype=float,
        )
    if np.any(~np.isfinite(grad)) or np.any(~np.isfinite(hess)):
        raise FloatingPointError("Invalid GEV derivatives at the Laplace expansion point.")
    information = np.maximum(-hess, float(curvature_floor))
    variance = np.minimum(1.0 / information, float(maximum_variance))
    shift = np.clip(grad / information, -float(shift_limit), float(shift_limit))
    pseudo_y[mask] = eta[mask] + shift
    pseudo_variance[mask] = variance
    return pseudo_y, pseudo_variance


def build_laplace_approximation(
    y: Array,
    compiled: CompiledModel,
    params: dict[str, float],
    *,
    initial_path: Array | None = None,
    max_iterations: int = 30,
    tolerance: float = 1e-5,
    curvature_floor: float = 1e-6,
    maximum_variance: float = 1e8,
    shift_limit: float | None = None,
 ) -> LaplaceApproximation:
    """Build the mode-matched Gaussian state proposal.

    ``initial_path=None`` gives a deterministic proposal conditional on
    ``(y, params)`` and is therefore the required setting for
    :func:`laplace_mh`.  Supplying an initial path remains useful for the fast,
    approximate :func:`iterated_laplace` update.
    """
    if bool(getattr(compiled, "all_gaussian", compiled.family == "gaussian")):
        raise ValueError("The iterated Laplace backend requires at least one non-Gaussian channel.")
    y = np.asarray(y, dtype=float)
    if hasattr(compiled, "channel_names"):
        expected = (compiled.n_time, len(compiled.channel_names))
        if y.shape != expected:
            raise ValueError(f"y must have shape {expected}; got {y.shape}.")
    else:
        y = y.reshape(-1)
    if y.shape[0] != compiled.n_time:
        raise ValueError("y and compiled model have different lengths.")
    if max_iterations < 1 or tolerance <= 0.0:
        raise ValueError("max_iterations and tolerance must be positive.")
    if shift_limit is None:
        if hasattr(compiled, "channel_names"):
            maximum_sigma = max(float(params[f"sigma.{name}"]) for name in compiled.channel_names)
        else:
            maximum_sigma = float(params["sigma"])
        shift_limit = 10.0 * maximum_sigma
    shift_limit = float(shift_limit)
    initial_variance = (
        compiled.observation_variance(params)
        if hasattr(compiled, "observation_variance")
        else max(float(params["sigma"]) ** 2, 1e-8)
    )

    if initial_path is None:
        initial_filter = kalman_filter(
            y,
            compiled,
            params,
            observation_variance=initial_variance,
        )
        mode_path = compiled.project_path(
            kalman_smoother(initial_filter, compiled).mean, params
        )
    else:
        mode_path = np.asarray(initial_path, dtype=float).copy()
        if mode_path.shape != (compiled.n_time + 1, compiled.state_dim):
            raise ValueError("initial_path has the wrong shape.")

    current_objective = joint_state_log_density(y, mode_path, compiled, params)
    if not np.isfinite(current_objective):
        # A Gaussian warm start can cross a finite GEV endpoint. Use the data as
        # pseudo observations once to obtain a support-compatible location.
        fallback = kalman_filter(
            y,
            compiled,
            params,
            observation_variance=initial_variance,
        )
        mode_path = compiled.project_path(kalman_smoother(fallback, compiled).mean, params)
        current_objective = joint_state_log_density(y, mode_path, compiled, params)
    if not np.isfinite(current_objective):
        raise FloatingPointError("Could not initialize the GEV Laplace mode inside the support.")

    converged = False
    relative_change = np.inf
    pseudo_y = y.copy()
    pseudo_variance = np.asarray(initial_variance, dtype=float).copy()
    filter_result: KalmanResult | None = None

    for iteration in range(1, int(max_iterations) + 1):
        eta = compiled.eta(mode_path, params=params)
        pseudo_y, pseudo_variance = _pseudo_data(
            y,
            eta,
            compiled,
            params,
            curvature_floor=curvature_floor,
            maximum_variance=maximum_variance,
            shift_limit=shift_limit,
        )
        filter_result = kalman_filter(
            pseudo_y,
            compiled,
            params,
            observation_variance=pseudo_variance,
        )
        candidate = compiled.project_path(kalman_smoother(filter_result, compiled).mean, params)

        accepted = False
        step = 1.0
        candidate_objective = -np.inf
        proposal = mode_path
        while step >= 2.0**-12:
            proposal = mode_path + step * (candidate - mode_path)
            candidate_objective = joint_state_log_density(y, proposal, compiled, params)
            if np.isfinite(candidate_objective) and candidate_objective >= current_objective - 1e-8:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            relative_change = 0.0
            converged = True
            break

        old_eta = eta
        mode_path = proposal
        current_objective = candidate_objective
        new_eta = compiled.eta(mode_path, params=params)
        relative_change = float(
            np.max(np.abs(new_eta - old_eta)) / (1.0 + np.max(np.abs(old_eta)))
        )
        if relative_change < tolerance:
            converged = True
            break

    # Refresh the quadratic approximation at the final accepted mode.
    pseudo_y, pseudo_variance = _pseudo_data(
        y,
        compiled.eta(mode_path, params=params),
        compiled,
        params,
        curvature_floor=curvature_floor,
        maximum_variance=maximum_variance,
        shift_limit=shift_limit,
    )
    filter_result = kalman_filter(
        pseudo_y,
        compiled,
        params,
        observation_variance=pseudo_variance,
    )
    return LaplaceApproximation(
        mode_path=mode_path,
        pseudo_y=pseudo_y,
        pseudo_variance=pseudo_variance,
        converged=converged,
        iterations=iteration,
        relative_change=relative_change,
        objective=current_objective,
        filter=filter_result,
    )


def draw_laplace_proposal(
    approximation: LaplaceApproximation,
    compiled: CompiledModel,
    params: dict[str, float],
    rng: np.random.Generator,
) -> Array:
    """Draw once from a previously constructed Gaussian state proposal."""

    path, _ = ffbs(
        approximation.pseudo_y,
        compiled,
        params,
        rng,
        observation_variance=approximation.pseudo_variance,
        filter_result=approximation.filter,
    )
    return path


def gaussian_approximation_log_likelihood(
    path: Array,
    approximation: LaplaceApproximation,
    compiled: CompiledModel,
    params: dict[str, float],
) -> float:
    """Log pseudo-observation density used by the Gaussian proposal."""

    eta = np.asarray(compiled.eta(path, params=params), dtype=float)
    pseudo_y = np.asarray(approximation.pseudo_y, dtype=float)
    variance = np.asarray(approximation.pseudo_variance, dtype=float)
    mask = np.isfinite(pseudo_y)
    if not np.any(mask):
        return 0.0
    residual = pseudo_y[mask] - eta[mask]
    selected_variance = variance[mask]
    if (
        np.any(~np.isfinite(residual))
        or np.any(~np.isfinite(selected_variance))
        or np.any(selected_variance <= 0.0)
    ):
        return -np.inf
    return float(
        -0.5
        * np.sum(
            np.log(2.0 * np.pi * selected_variance)
            + residual * residual / selected_variance
        )
    )


def laplace_log_correction(
    y: Array,
    path: Array,
    approximation: LaplaceApproximation,
    compiled: CompiledModel,
    params: dict[str, float],
) -> float:
    """Return ``log L(path) - log L_tilde(path)`` for MH or IS weights.

    The exact and Gaussian state priors are identical, including any singular
    transition directions, so they cancel from the density ratio.  This is why
    no determinant or pseudo-density for the latent transition is needed.
    """

    exact = observation_log_likelihood(
        y, compiled.eta(path, params=params), compiled, params
    )
    if not np.isfinite(exact):
        return -np.inf
    approximate = gaussian_approximation_log_likelihood(
        path, approximation, compiled, params
    )
    if not np.isfinite(approximate):
        return -np.inf
    return float(exact - approximate)


def laplace_mh(
    y: Array,
    compiled: CompiledModel,
    params: dict[str, float],
    current_path: Array,
    rng: np.random.Generator,
    *,
    mh_steps: int = 1,
    max_iterations: int = 30,
    tolerance: float = 1e-5,
    curvature_floor: float = 1e-6,
    maximum_variance: float = 1e8,
    shift_limit: float | None = None,
) -> LaplaceMHResult:
    """Exact-invariant independence MH using an iterated-Laplace proposal.

    The Gaussian approximation is constructed once, deterministically from the
    observations and the current static parameters.  Every proposal is a full
    state trajectory drawn by FFBS.  Invalid GEV-support proposals are ordinary
    MH rejections; the proposal is never truncated and never falls back to an
    atom at the mode.
    """

    if int(mh_steps) < 1:
        raise ValueError("mh_steps must be positive.")
    current = np.asarray(current_path, dtype=float).copy()
    expected = (compiled.n_time + 1, compiled.state_dim)
    if current.shape != expected:
        raise ValueError(f"current_path must have shape {expected}.")
    approximation = build_laplace_approximation(
        y,
        compiled,
        params,
        initial_path=None,
        max_iterations=max_iterations,
        tolerance=tolerance,
        curvature_floor=curvature_floor,
        maximum_variance=maximum_variance,
        shift_limit=shift_limit,
    )
    current_weight = laplace_log_correction(
        y, current, approximation, compiled, params
    )
    if not np.isfinite(current_weight):
        raise ValueError("The current state trajectory is outside the exact target support.")

    accepted = np.zeros(int(mh_steps), dtype=bool)
    log_ratios = np.full(int(mh_steps), -np.inf, dtype=float)
    support_failures = 0
    for step in range(int(mh_steps)):
        proposal = draw_laplace_proposal(approximation, compiled, params, rng)
        proposal_weight = laplace_log_correction(
            y, proposal, approximation, compiled, params
        )
        if not np.isfinite(proposal_weight):
            support_failures += 1
            continue
        log_ratio = float(proposal_weight - current_weight)
        log_ratios[step] = log_ratio
        if np.log(rng.random()) < min(0.0, log_ratio):
            current = proposal
            current_weight = proposal_weight
            accepted[step] = True

    return LaplaceMHResult(
        path=current,
        approximation=approximation,
        accepted=accepted,
        log_acceptance_ratio=log_ratios,
        log_weight=float(current_weight),
        proposal_support_failures=int(support_failures),
    )


def iterated_laplace(
    y: Array,
    compiled: CompiledModel,
    params: dict[str, float],
    rng: np.random.Generator,
    *,
    initial_path: Array | None = None,
    max_iterations: int = 30,
    tolerance: float = 1e-5,
    curvature_floor: float = 1e-6,
    maximum_variance: float = 1e8,
    shift_limit: float | None = None,
    draw_attempts: int = 30,
) -> LaplaceDraw:
    """Fast approximate state draw retained for backward compatibility."""

    if int(draw_attempts) < 1:
        raise ValueError("draw_attempts must be positive.")
    approximation = build_laplace_approximation(
        y,
        compiled,
        params,
        initial_path=initial_path,
        max_iterations=max_iterations,
        tolerance=tolerance,
        curvature_floor=curvature_floor,
        maximum_variance=maximum_variance,
        shift_limit=shift_limit,
    )
    support_rejections = 0
    path = approximation.mode_path.copy()
    for _ in range(int(draw_attempts)):
        proposal = draw_laplace_proposal(approximation, compiled, params, rng)
        if np.isfinite(
            observation_log_likelihood(
                y, compiled.eta(proposal, params=params), compiled, params
            )
        ):
            path = proposal
            break
        support_rejections += 1

    return LaplaceDraw(
        path=path,
        mode_path=approximation.mode_path,
        pseudo_y=approximation.pseudo_y,
        pseudo_variance=approximation.pseudo_variance,
        converged=approximation.converged,
        iterations=approximation.iterations,
        relative_change=approximation.relative_change,
        objective=approximation.objective,
        support_rejections=support_rejections,
        filter=approximation.filter,
    )
