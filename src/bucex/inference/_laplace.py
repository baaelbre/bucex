"""Deterministic Laplace--FFBS proposals with exact likelihood MH correction."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
Array = np.ndarray
ParamDict = Dict[str, Any]

def _symmetrize(a):
    return (a + a.T) * .5

from ._state import (GaussianFilter1D, NCPLayout, _filter_gaussian_1d_tvR, baseline_mu_path, build_ncp_system, ffbs_gaussian_1d_tvR, gaussian_smoother_mean_1d_tvR, measurement_vector, mu_from_ncp)

@dataclass(frozen=True)
class NCPLaplaceApproximation:
    """Deterministic Gaussian proposal for the FS non-centred state block."""

    mode_path: Array
    pseudo_y: Array
    pseudo_variance: Array
    transition: Array
    process_covariance: Array
    measurement: Array
    offset: Array
    filter: GaussianFilter1D
    converged: bool
    iterations: int
    relative_change: float
    objective: float
    initial_support_repaired: bool = False

@dataclass(frozen=True)
class NCPLaplaceMHResult:
    """One or more exact Laplace independence-MH updates of an FS path."""

    z_path: Array
    approximation: NCPLaplaceApproximation
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

def _ncp_transition_structure(Q: Array) -> tuple[Array, Array, Array]:
    diagonal = np.diag(np.asarray(Q, dtype=float))
    active = np.flatnonzero(diagonal > 1e-12)
    deterministic = np.flatnonzero(diagonal <= 1e-12)
    loading = np.eye(Q.shape[0], dtype=float)[:, active]
    return active, deterministic, loading

def _project_ncp_path_to_support(path: Array, G: Array, Q: Array) -> Array:
    """Remove floating-point drift in deterministic transition coordinates.

    Singular Gaussian simulation can leave order-``sqrt(eps)`` noise in a
    direction whose theoretical conditional variance is zero.  Replacing only
    those coordinates by their affine recursion realizes the intended
    degenerate Gaussian draw and makes support checks auditable.
    """

    projected = np.asarray(path, dtype=float).copy()
    if projected.ndim != 2 or projected.shape[1] != G.shape[0]:
        raise ValueError("path has the wrong state dimension.")
    _, deterministic, _ = _ncp_transition_structure(Q)
    projected[0] = 0.0
    if deterministic.size:
        for t in range(1, projected.shape[0]):
            projected[t, deterministic] = (G @ projected[t - 1])[deterministic]
    return projected

def _ncp_transition_logpdf(
    value: Array,
    means: Array,
    active: Array,
    deterministic: Array,
    *,
    tolerance: float = 1e-12,
) -> Array:
    means = np.asarray(means, dtype=float)
    value = np.asarray(value, dtype=float)
    residual = value - means
    one = residual.ndim == 1
    if one:
        residual = residual[None, :]
        means = means[None, :]
    output = np.full(residual.shape[0], -np.inf, dtype=float)
    on_support = np.ones(residual.shape[0], dtype=bool)
    if deterministic.size:
        deterministic_error = np.max(np.abs(residual[:, deterministic]), axis=1)
        reference_scale = 1.0 + np.max(np.abs(means[:, deterministic]), axis=1)
        on_support &= deterministic_error <= tolerance * reference_scale
    if np.any(on_support):
        stochastic = residual[on_support][:, active]
        output[on_support] = -0.5 * (
            active.size * np.log(2.0 * np.pi)
            + np.sum(stochastic * stochastic, axis=1)
        )
    return output[0] if one else output

def _ncp_exact_observation_loglik(
    y: Array,
    z_path: Array,
    params_state: ParamDict,
    params_obs: ParamDict,
    model: Any,
    layout: NCPLayout,
) -> float:
    mu = mu_from_ncp(z_path, params_state, layout)
    try:
        values = np.asarray(
            model.obs.logpdf(
                y=np.asarray(y, dtype=float), eta=mu, params=params_obs
            ),
            dtype=float,
        )
    except Exception:
        return -np.inf
    if values.ndim == 0:
        values = np.full(mu.shape, float(values), dtype=float)
    try:
        values = np.broadcast_to(values, mu.shape)
    except ValueError:
        return -np.inf
    if not np.all(np.isfinite(values)):
        return -np.inf
    return float(np.sum(values))

def _ncp_state_log_density(z_path: Array, G: Array, Q: Array) -> float:
    z_path = np.asarray(z_path, dtype=float)
    if np.linalg.norm(z_path[0]) > 1e-7:
        return -np.inf
    active, deterministic, _ = _ncp_transition_structure(Q)
    values = _ncp_transition_logpdf(
        z_path[1:], z_path[:-1] @ G.T, active, deterministic
    )
    if not np.all(np.isfinite(values)):
        return -np.inf
    return float(np.sum(values))

def _ncp_laplace_pseudo_data(
    y: Array,
    eta: Array,
    model: Any,
    params_obs: ParamDict,
    *,
    curvature_floor: float,
    maximum_variance: float,
    shift_limit: float,
) -> tuple[Array, Array]:
    y = np.asarray(y, dtype=float).reshape(-1)
    eta = np.asarray(eta, dtype=float).reshape(-1)
    try:
        gradient = np.asarray(model.obs.grad_eta(y, eta, params_obs), dtype=float)
        hessian = np.asarray(model.obs.hess_eta(y, eta, params_obs), dtype=float)
        gradient = np.broadcast_to(gradient, y.shape)
        hessian = np.broadcast_to(hessian, y.shape)
    except Exception as error:
        raise FloatingPointError(
            "Invalid GEV derivatives at the Laplace mode."
        ) from error
    if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(hessian)):
        raise FloatingPointError("Invalid GEV derivatives at the Laplace mode.")
    information = np.maximum(-hessian, float(curvature_floor))
    pseudo_variance = np.minimum(1.0 / information, float(maximum_variance))
    pseudo_y = eta + np.clip(
        gradient / information, -float(shift_limit), float(shift_limit)
    )
    return pseudo_y, pseudo_variance

def _deterministic_feasible_ncp_path(
    y: Array,
    model: Any,
    params_state: ParamDict,
    params_obs: ParamDict,
    layout: NCPLayout,
    G: Array,
    Q: Array,
    H: Array,
    offset: Array,
    *,
    minimum_support: float = 0.25,
) -> tuple[Array, bool]:
    """Return a deterministic FS path from which Laplace iteration can start.

    The Laplace-MH proposal must be fixed conditional on the observations and
    static parameters.  Initialising its mode search from the current MCMC path
    would make the proposal state dependent and would require an additional
    forward/reverse proposal correction.  A zero non-centred path is normally
    sufficient, but it can cross a finite GEV endpoint even when the current
    full trajectory is valid.

    This helper repairs that case without consulting the current trajectory.
    It exploits the causal FS state structure.  A contemporaneously loaded
    level or seasonal innovation is used when available; otherwise an
    integrated-slope innovation is chosen one step ahead.  Every repair moves
    the affected predictor to ``eta_t = y_t``, where the GEV support value is
    exactly one and its derivatives are well behaved.  The resulting path is
    deterministic in ``(y, params_state, params_obs)`` and lies on the exact
    singular transition support.
    """

    y = np.asarray(y, dtype=float).reshape(-1)
    path = np.zeros((y.size + 1, layout.ncp_state_dim), dtype=float)
    if np.isfinite(
        _ncp_exact_observation_loglik(
            y, path, params_state, params_obs, model, layout
        )
    ):
        return path, False

    if "xi" not in params_obs or "sigma" not in params_obs:
        raise FloatingPointError(
            "The deterministic Laplace initializer is outside observation support."
        )
    sigma = np.asarray(params_obs["sigma"], dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(y.size, float(sigma), dtype=float)
    elif sigma.shape != y.shape:
        raise FloatingPointError("GEV sigma must be scalar or have length T.")
    xi = float(params_obs["xi"])
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0) or not np.isfinite(xi):
        raise FloatingPointError("Invalid GEV sigma or xi at Laplace initialization.")
    if abs(xi) < 1e-12:
        raise FloatingPointError(
            "The Gumbel Laplace initializer has a non-finite objective."
        )

    minimum_support = float(minimum_support)
    if not 0.0 < minimum_support < 1.0:
        raise ValueError("minimum_support must lie strictly between zero and one.")

    def support_value(time: int, observation: float, predictor: float) -> float:
        return float(1.0 + xi * (observation - predictor) / sigma[int(time)])

    active, _, _ = _ncp_transition_structure(Q)
    immediate = np.asarray(H[active], dtype=float)
    usable_now = np.flatnonzero(
        np.isfinite(immediate) & (immediate != 0.0)
    )

    if usable_now.size:
        local = int(usable_now[np.argmax(np.abs(immediate[usable_now]))])
        innovation_index = int(active[local])
        loading = float(H[innovation_index])
        for time in range(1, y.size + 1):
            state = G @ path[time - 1]
            predictor = float(offset[time - 1] + H @ state)
            support = support_value(time - 1, float(y[time - 1]), predictor)
            if not np.isfinite(support) or support < minimum_support:
                state[innovation_index] += (
                    float(y[time - 1]) - predictor
                ) / loading
            path[time] = state
    else:
        # With only a dynamic slope, its innovation is measured through the
        # integrated coordinate at the next observation.  The first predictor
        # is deterministic; a valid current FS path therefore guarantees that
        # its first-observation support cannot require repair.
        one_step = np.asarray(H @ G[:, active], dtype=float)
        usable_next = np.flatnonzero(
            np.isfinite(one_step) & (one_step != 0.0)
        )
        if not usable_next.size:
            raise FloatingPointError(
                "No stochastic FS component can move the predictor into GEV support."
            )
        local = int(usable_next[np.argmax(np.abs(one_step[usable_next]))])
        innovation_index = int(active[local])
        loading = float(one_step[local])
        for time in range(1, y.size + 1):
            state = G @ path[time - 1]
            predictor = float(offset[time - 1] + H @ state)
            support = support_value(time - 1, float(y[time - 1]), predictor)
            if not np.isfinite(support) or support <= 0.0:
                raise FloatingPointError(
                    "The deterministic first-lag FS predictor is outside GEV support."
                )
            if time < y.size:
                next_predictor = float(offset[time] + H @ (G @ state))
                next_support = support_value(time, float(y[time]), next_predictor)
                if not np.isfinite(next_support) or next_support < minimum_support:
                    state[innovation_index] += (
                        float(y[time]) - next_predictor
                    ) / loading
            path[time] = state

    path = _project_ncp_path_to_support(path, G, Q)
    if not np.isfinite(_ncp_state_log_density(path, G, Q)):
        raise FloatingPointError(
            "The repaired Laplace initializer left FS transition support."
        )
    if not np.isfinite(
        _ncp_exact_observation_loglik(
            y, path, params_state, params_obs, model, layout
        )
    ):
        raise FloatingPointError(
            "Could not construct a deterministic FS path inside GEV support."
        )
    return path, True

def build_ncp_laplace_approximation(
    y: Array,
    model: Any,
    params_state: ParamDict,
    params_obs: ParamDict,
    layout: NCPLayout,
    *,
    initial_path: Optional[Array] = None,
    max_iterations: int = 30,
    tolerance: float = 1e-5,
    curvature_floor: float = 1e-6,
    maximum_variance: float = 1e8,
    shift_limit: Optional[float] = None,
) -> NCPLaplaceApproximation:
    """Construct the Gaussian approximation for the FS state block.

    With ``initial_path=None`` construction is deterministic conditional on the
    data and static parameters.  That is the proposal used by
    :func:`ncp_laplace_mh`; a current-chain path is never used to define its own
    independence proposal.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    if int(max_iterations) < 1 or float(tolerance) <= 0.0:
        raise ValueError("max_iterations and tolerance must be positive.")
    Tn = y.size
    G, Q = build_ncp_system(layout)
    H = measurement_vector(params_state, layout)
    offset = baseline_mu_path(Tn, params_state, layout)
    initial_support_repaired = False
    if initial_path is None:
        mode, initial_support_repaired = _deterministic_feasible_ncp_path(
            y,
            model,
            params_state,
            params_obs,
            layout,
            G,
            Q,
            H,
            offset,
        )
    else:
        mode = np.asarray(initial_path, dtype=float).copy()
    if mode.shape != (Tn + 1, layout.ncp_state_dim):
        raise ValueError("initial_path has the wrong shape.")
    mode = _project_ncp_path_to_support(mode, G, Q)
    sigma_values = np.asarray(params_obs["sigma"], dtype=float)
    shift_limit = float(
        10.0 * float(np.max(sigma_values))
        if shift_limit is None
        else shift_limit
    )

    def objective(path: Array) -> float:
        return _ncp_state_log_density(path, G, Q) + _ncp_exact_observation_loglik(
            y, path, params_state, params_obs, model, layout
        )

    current_objective = objective(mode)
    if not np.isfinite(current_objective):
        raise FloatingPointError(
            "Could not initialize the FS GEV Laplace mode inside the support."
        )
    converged = False
    relative_change = np.inf
    pseudo_y = np.asarray(y, dtype=float).copy()
    pseudo_variance = np.broadcast_to(sigma_values, (Tn,)).astype(float) ** 2
    iteration = 0

    for iteration in range(1, int(max_iterations) + 1):
        eta = mu_from_ncp(mode, params_state, layout)
        pseudo_y, pseudo_variance = _ncp_laplace_pseudo_data(
            y,
            eta,
            model,
            params_obs,
            curvature_floor=curvature_floor,
            maximum_variance=maximum_variance,
            shift_limit=shift_limit,
        )
        candidate = gaussian_smoother_mean_1d_tvR(
            pseudo_y - offset,
            G,
            Q,
            H,
            pseudo_variance,
            m0=np.zeros(layout.ncp_state_dim),
            C0=np.zeros((layout.ncp_state_dim, layout.ncp_state_dim)),
        )
        candidate = _project_ncp_path_to_support(candidate, G, Q)
        accepted = False
        step_size = 1.0
        proposal = mode
        candidate_objective = -np.inf
        while step_size >= 2.0**-12:
            proposal = mode + step_size * (candidate - mode)
            proposal = _project_ncp_path_to_support(proposal, G, Q)
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
        old_eta = eta
        mode = proposal
        current_objective = candidate_objective
        new_eta = mu_from_ncp(mode, params_state, layout)
        relative_change = float(
            np.max(np.abs(new_eta - old_eta))
            / (1.0 + np.max(np.abs(old_eta)))
        )
        if relative_change < tolerance:
            converged = True
            break

    pseudo_y, pseudo_variance = _ncp_laplace_pseudo_data(
        y,
        mu_from_ncp(mode, params_state, layout),
        model,
        params_obs,
        curvature_floor=curvature_floor,
        maximum_variance=maximum_variance,
        shift_limit=shift_limit,
    )
    filter_result = _filter_gaussian_1d_tvR(
        pseudo_y - offset,
        G,
        Q,
        H,
        pseudo_variance,
        m0=np.zeros(layout.ncp_state_dim),
        C0=np.zeros((layout.ncp_state_dim, layout.ncp_state_dim)),
    )
    return NCPLaplaceApproximation(
        mode_path=mode,
        pseudo_y=pseudo_y,
        pseudo_variance=pseudo_variance,
        transition=G,
        process_covariance=Q,
        measurement=H,
        offset=offset,
        filter=filter_result,
        converged=converged,
        iterations=int(iteration),
        relative_change=float(relative_change),
        objective=float(current_objective),
        initial_support_repaired=bool(initial_support_repaired),
    )

def draw_ncp_laplace_proposal(
    approximation: NCPLaplaceApproximation,
    rng: Optional[np.random.Generator] = None,
) -> Array:
    """Draw one full FS trajectory from a fixed Gaussian approximation."""

    rng = np.random.default_rng() if rng is None else rng
    dimension = int(approximation.transition.shape[0])
    path = ffbs_gaussian_1d_tvR(
        approximation.pseudo_y - approximation.offset,
        approximation.transition,
        approximation.process_covariance,
        approximation.measurement,
        approximation.pseudo_variance,
        m0=np.zeros(dimension),
        C0=np.zeros((dimension, dimension)),
        rng=rng,
        filter_result=approximation.filter,
    )
    return _project_ncp_path_to_support(
        path,
        approximation.transition,
        approximation.process_covariance,
    )

def _ncp_gaussian_approximation_loglik(
    z_path: Array,
    approximation: NCPLaplaceApproximation,
) -> float:
    predictor = (
        approximation.offset
        + np.asarray(z_path, dtype=float)[1:] @ approximation.measurement
    )
    variance = np.asarray(approximation.pseudo_variance, dtype=float)
    residual = np.asarray(approximation.pseudo_y, dtype=float) - predictor
    if (
        np.any(~np.isfinite(residual))
        or np.any(~np.isfinite(variance))
        or np.any(variance <= 0.0)
    ):
        return -np.inf
    return float(
        -0.5
        * np.sum(np.log(2.0 * np.pi * variance) + residual * residual / variance)
    )

def ncp_laplace_log_correction(
    y: Array,
    z_path: Array,
    approximation: NCPLaplaceApproximation,
    model: Any,
    params_state: ParamDict,
    params_obs: ParamDict,
    layout: NCPLayout,
) -> float:
    """Exact-over-Gaussian likelihood ratio for MH and importance weights."""

    exact = _ncp_exact_observation_loglik(
        y, z_path, params_state, params_obs, model, layout
    )
    if not np.isfinite(exact):
        return -np.inf
    approximate = _ncp_gaussian_approximation_loglik(z_path, approximation)
    if not np.isfinite(approximate):
        return -np.inf
    return float(exact - approximate)

def ncp_laplace_mh(
    y: Array,
    model: Any,
    params_state: ParamDict,
    params_obs: ParamDict,
    layout: NCPLayout,
    current_path: Array,
    *,
    rng: Optional[np.random.Generator] = None,
    mh_steps: int = 1,
    max_iterations: int = 30,
    tolerance: float = 1e-5,
    curvature_floor: float = 1e-6,
    maximum_variance: float = 1e8,
    shift_limit: Optional[float] = None,
) -> NCPLaplaceMHResult:
    """Exact-invariant independence MH for the FS non-centred trajectory.

    The exact FS Gaussian state law and the state law inside the Laplace
    proposal are identical.  They therefore cancel from the MH ratio even when
    ``Q`` is singular (integrated slope and dummy-seasonal lag coordinates).
    """

    if int(mh_steps) < 1:
        raise ValueError("mh_steps must be positive.")
    rng = np.random.default_rng() if rng is None else rng
    y = np.asarray(y, dtype=float).reshape(-1)
    current = np.asarray(current_path, dtype=float).copy()
    expected = (y.size + 1, layout.ncp_state_dim)
    if current.shape != expected:
        raise ValueError(f"current_path must have shape {expected}.")
    G, Q = build_ncp_system(layout)
    if not np.isfinite(_ncp_state_log_density(current, G, Q)):
        raise ValueError("The current FS trajectory is outside transition support.")

    approximation = build_ncp_laplace_approximation(
        y,
        model,
        params_state,
        params_obs,
        layout,
        initial_path=None,
        max_iterations=max_iterations,
        tolerance=tolerance,
        curvature_floor=curvature_floor,
        maximum_variance=maximum_variance,
        shift_limit=shift_limit,
    )
    current_weight = ncp_laplace_log_correction(
        y,
        current,
        approximation,
        model,
        params_state,
        params_obs,
        layout,
    )
    if not np.isfinite(current_weight):
        raise ValueError("The current FS trajectory is outside exact GEV support.")

    accepted = np.zeros(int(mh_steps), dtype=bool)
    log_ratios = np.full(int(mh_steps), -np.inf, dtype=float)
    support_failures = 0
    for step in range(int(mh_steps)):
        proposal = draw_ncp_laplace_proposal(approximation, rng)
        proposal_weight = ncp_laplace_log_correction(
            y,
            proposal,
            approximation,
            model,
            params_state,
            params_obs,
            layout,
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

    return NCPLaplaceMHResult(
        z_path=current,
        approximation=approximation,
        accepted=accepted,
        log_acceptance_ratio=log_ratios,
        log_weight=float(current_weight),
        proposal_support_failures=int(support_failures),
    )
