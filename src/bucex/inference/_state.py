"""Non-centred structural states and singular-support-aware Gaussian smoothing."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
Array = np.ndarray
ParamDict = Dict[str, Any]

def _symmetrize(a):
    return (a + a.T) * .5

@dataclass(frozen=True)
class NCPLayout:
    centered_state_names: tuple[str, ...]
    centered_state_dim: int
    has_alpha: bool
    has_beta: bool
    season_dim: int
    idx_alpha: Optional[int]
    idx_beta: Optional[int]
    season_slice: slice
    idx_tilde_alpha: Optional[int]
    idx_tilde_beta: Optional[int]
    idx_A: Optional[int]
    season_ncp_slice: slice
    ncp_state_names: tuple[str, ...]
    ncp_state_dim: int
    active_components: tuple[str, ...] | None = None

def spd_solve(A: Array, B: Array, jitter: float = 1e-12) -> Array:
    A = _symmetrize(np.asarray(A, dtype=float))
    B = np.asarray(B, dtype=float)
    n = A.shape[0]
    try:
        L = np.linalg.cholesky(A)
        Y = np.linalg.solve(L, B)
        return np.linalg.solve(L.T, Y)
    except np.linalg.LinAlgError:
        pass
    values = np.linalg.eigvalsh(A)
    scale = max(float(np.max(np.abs(values))), 1.0)
    if np.any(values < -1e-6 * scale):
        raise np.linalg.LinAlgError("Matrix is materially indefinite.")
    return np.linalg.pinv(A, rcond=1e-12, hermitian=True) @ B

def _sample_gaussian(
    mean: Array,
    cov: Array,
    rng: np.random.Generator,
    jitter: float = 1e-12,
) -> Array:
    """Sample from a Gaussian after deterministic PSD repair.

    This avoids NumPy's warning-based fallback, which can silently alter an
    indefinite covariance matrix.
    """

    mean = np.asarray(mean, dtype=float)
    covariance = _symmetrize(np.asarray(cov, dtype=float))
    values, vectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(np.abs(values))) if values.size else 0.0, 1.0)
    if np.any(values < -1e-6 * scale):
        raise np.linalg.LinAlgError("Gaussian covariance is materially indefinite.")
    values = np.clip(values, 0.0, None)
    return mean + vectors @ (np.sqrt(values) * rng.normal(size=mean.size))

def _backward_smoothing_moments(
    filtered_mean: Array,
    filtered_covariance: Array,
    next_state: Array,
    transition: Array,
    process_covariance: Array,
    predicted_covariance: Array,
    *,
    jitter: float = 1e-12,
) -> tuple[Array, Array]:
    """Conditional moments used by the FFBS backward sampler.

    The familiar subtraction

    ``C_t - J_t R_{t+1} J_t'``

    is algebraically correct but numerically fragile when the transition is
    singular or an innovation variance is close to zero.  In long series it
    can subtract two nearly equal matrices and create a materially negative
    eigenvalue.  The Joseph-style factorisation below is equivalent in exact
    arithmetic and preserves positive semidefiniteness much more reliably:

    ``(I - JG) C_t (I - JG)' + J Q J'``.

    A direct symmetric solve is used for the smoothing gain.  ``spd_solve``
    deliberately falls back to a Moore--Penrose inverse for valid singular
    predicted covariances, so deterministic state directions remain exactly
    deterministic instead of being hidden by diagonal jitter.
    """

    filtered_covariance = _symmetrize(
        np.asarray(filtered_covariance, dtype=float)
    )
    predicted_covariance = _symmetrize(
        np.asarray(predicted_covariance, dtype=float)
    )
    transition = np.asarray(transition, dtype=float)
    process_covariance = _symmetrize(
        np.asarray(process_covariance, dtype=float)
    )
    gain = spd_solve(
        predicted_covariance,
        transition @ filtered_covariance,
        jitter=jitter,
    ).T
    mean = np.asarray(filtered_mean, dtype=float) + gain @ (
        np.asarray(next_state, dtype=float)
        - transition @ np.asarray(filtered_mean, dtype=float)
    )
    residual_map = np.eye(filtered_covariance.shape[0]) - gain @ transition
    covariance = _symmetrize(
        residual_map @ filtered_covariance @ residual_map.T
        + gain @ process_covariance @ gain.T
    )
    return mean, covariance

def seasonal_rotation_matrix(K: int) -> Array:
    if K <= 0:
        return np.zeros((0, 0), dtype=float)
    S = np.zeros((K, K), dtype=float)
    S[0, :] = -1.0
    if K > 1:
        S[1:, :-1] = np.eye(K - 1)
    return S

def seasonal_state_from_phase_effects(effects: Array) -> Array:
    """Encode chronological zero-sum phase effects as the first dummy state.

    With the lag rotation, the observations visit coordinates in the order
    ``g[0], -sum(g), g[-1], ..., g[1]``.  In particular, storing the first
    ``p-1`` chronological effects as the state gives a different annual cycle.
    """
    full = np.asarray(effects, dtype=float).reshape(-1)
    if full.size < 2 or not np.all(np.isfinite(full)) or not np.isclose(full.sum(), 0.):
        raise ValueError("Seasonal phase effects must be finite, zero-sum, and have period >= 2.")
    return np.r_[full[0], full[:1:-1]]

def static_seasonal_design(Tn: int, K: int) -> Array:
    """Design for the deterministic dummy-seasonal initial-state vector.

    ``gamma0_season`` is the seasonal *state at the first observation*, not a
    vector of chronological phase effects.  For the standard sum-to-zero
    dummy-seasonal transition, the next observed effect is the negative sum
    of that vector and subsequent effects are rotated lag coordinates.  The
    design therefore consists of ``e_1' S^(t-1)`` rows.

    Keeping this construction in one place is important: using ordinary dummy
    columns here would fit one seasonal path and reconstruct a different one.
    """

    Tn = int(Tn)
    K = int(K)
    if Tn < 0 or K < 0:
        raise ValueError("Tn and K must be non-negative.")
    if K == 0:
        return np.zeros((Tn, 0), dtype=float)
    rotation = seasonal_rotation_matrix(K)
    mapping = np.eye(K, dtype=float)
    design = np.zeros((Tn, K), dtype=float)
    for index in range(Tn):
        design[index] = mapping[0]
        mapping = rotation @ mapping
    return design

def build_ncp_system(layout: NCPLayout) -> tuple[Array, Array]:
    d = layout.ncp_state_dim
    G = np.zeros((d, d), dtype=float)
    Q = np.zeros((d, d), dtype=float)

    if layout.idx_tilde_alpha is not None:
        i = layout.idx_tilde_alpha
        G[i, i] = 1.0
        Q[i, i] = 1.0

    if layout.idx_tilde_beta is not None:
        ib = layout.idx_tilde_beta
        iA = int(layout.idx_A)
        G[ib, ib] = 1.0
        G[iA, iA] = 1.0
        G[iA, ib] = 1.0
        Q[ib, ib] = 1.0

    if layout.season_dim > 0:
        gs = layout.season_ncp_slice.start
        ge = layout.season_ncp_slice.stop
        G[gs:ge, gs:ge] = seasonal_rotation_matrix(layout.season_dim)
        Q[gs, gs] = 1.0

    return G, Q

def baseline_mu_path(Tn: int, params_state: ParamDict, layout: NCPLayout) -> Array:
    t1 = np.arange(1, Tn + 1, dtype=float)
    mu = np.full(Tn, float(params_state.get("alpha0", 0.0)), dtype=float)
    if layout.has_beta:
        mu += float(params_state.get("beta0", 0.0)) * t1

    if layout.season_dim > 0:
        g1 = np.asarray(params_state["gamma0_season"], dtype=float).reshape(layout.season_dim)
        S = seasonal_rotation_matrix(layout.season_dim)
        g = g1.copy()
        for t in range(1, Tn + 1):
            if t > 1:
                g = S @ g
            mu[t - 1] += float(g[0])
    return mu

def baseline_centered_season_path(Tn: int, params_state: ParamDict, layout: NCPLayout) -> Array:
    if layout.season_dim == 0:
        return np.zeros((Tn + 1, 0), dtype=float)

    g1 = np.asarray(params_state["gamma0_season"], dtype=float).reshape(layout.season_dim)
    S = seasonal_rotation_matrix(layout.season_dim)
    out = np.zeros((Tn + 1, layout.season_dim), dtype=float)

    if Tn >= 1:
        out[1] = g1
    if Tn >= 0:
        out[0] = np.linalg.solve(S, g1)
    for t in range(2, Tn + 1):
        out[t] = S @ out[t - 1]
    return out

def measurement_vector(params_state: ParamDict, layout: NCPLayout) -> Array:
    H = np.zeros(layout.ncp_state_dim, dtype=float)
    if layout.idx_tilde_alpha is not None:
        H[layout.idx_tilde_alpha] = float(params_state.get("s_level", 0.0))
    if layout.idx_A is not None:
        H[layout.idx_A] = float(params_state.get("s_trend", 0.0))
    if layout.season_dim > 0:
        H[layout.season_ncp_slice.start] = float(params_state.get("s_season", 0.0))
    return H

def map_ncp_to_centered(z_path: Array, params_state: ParamDict, layout: NCPLayout) -> Array:
    z_path = np.asarray(z_path, dtype=float)
    Tn = z_path.shape[0] - 1
    x = np.zeros((Tn + 1, layout.centered_state_dim), dtype=float)

    t0 = np.arange(0, Tn + 1, dtype=float)
    alpha0 = float(params_state.get("alpha0", 0.0))
    beta0 = float(params_state.get("beta0", 0.0))
    s_level = float(params_state.get("s_level", 0.0))
    s_trend = float(params_state.get("s_trend", 0.0))
    s_season = float(params_state.get("s_season", 0.0))

    alpha = alpha0 + beta0 * t0 + s_level * z_path[:, int(layout.idx_tilde_alpha)]
    if layout.has_beta:
        alpha = alpha + s_trend * z_path[:, int(layout.idx_A)]
        beta = beta0 + s_trend * z_path[:, int(layout.idx_tilde_beta)]
        x[:, int(layout.idx_beta)] = beta

    x[:, int(layout.idx_alpha)] = alpha

    if layout.season_dim > 0:
        g_base = baseline_centered_season_path(Tn, params_state, layout)
        g = g_base + s_season * z_path[:, layout.season_ncp_slice]
        x[:, layout.season_slice] = g

    return x

def mu_from_ncp(z_path: Array, params_state: ParamDict, layout: NCPLayout) -> Array:
    z_path = np.asarray(z_path, dtype=float)
    Tn = z_path.shape[0] - 1
    mu = baseline_mu_path(Tn, params_state, layout)
    if layout.idx_tilde_alpha is not None:
        mu += float(params_state.get("s_level", 0.0)) * z_path[1:, int(layout.idx_tilde_alpha)]
    if layout.idx_A is not None:
        mu += float(params_state.get("s_trend", 0.0)) * z_path[1:, int(layout.idx_A)]
    if layout.season_dim > 0:
        mu += float(params_state.get("s_season", 0.0)) * z_path[1:, layout.season_ncp_slice.start]
    return mu

@dataclass(frozen=True)
class GaussianFilter1D:
    """Cached scalar-observation Kalman filter for repeated FFBS draws."""

    filtered_mean: Array
    filtered_covariance: Array
    predicted_mean: Array
    predicted_covariance: Array
    transition: Array
    process_covariance: Array

def _filter_gaussian_1d_tvR(
    y: Array,
    G: Array,
    Q: Array,
    H: Array,
    R_t: Array,
    *,
    m0: Optional[Array] = None,
    C0: Optional[Array] = None,
    jitter: float = 1e-12,
    R_floor: float = 0.0,
) -> GaussianFilter1D:
    y = np.asarray(y, dtype=float).reshape(-1)
    R_t = np.asarray(R_t, dtype=float).reshape(-1)
    Tn = int(y.size)
    d = int(G.shape[0])
    H = np.asarray(H, dtype=float).reshape(1, d)
    if np.any(R_t <= 0) or not np.all(np.isfinite(R_t)):
        raise ValueError("Observation variances must be positive and finite.")
    if R_t.size != Tn:
        raise ValueError("R_t must have length T")
    if m0 is None:
        m0 = np.zeros(d, dtype=float)
    if C0 is None:
        C0 = np.zeros((d, d), dtype=float)

    m = np.zeros((Tn + 1, d), dtype=float)
    C = np.zeros((Tn + 1, d, d), dtype=float)
    a = np.zeros((Tn + 1, d), dtype=float)
    Rm = np.zeros((Tn + 1, d, d), dtype=float)
    m[0] = np.asarray(m0, dtype=float)
    C[0] = _symmetrize(np.asarray(C0, dtype=float))
    for t in range(1, Tn + 1):
        observation_variance = float(max(R_floor, R_t[t - 1]))
        a[t] = G @ m[t - 1]
        Rm[t] = _symmetrize(G @ C[t - 1] @ G.T + Q)
        innovation_variance = float(
            (H @ Rm[t] @ H.T).item() + observation_variance
        )
        if not np.isfinite(innovation_variance) or innovation_variance <= 0.0:
            innovation_variance = float(
                (H @ (Rm[t] + jitter * np.eye(d)) @ H.T).item()
                + observation_variance
            )
        gain = (Rm[t] @ H.T) / innovation_variance
        innovation = float(y[t - 1] - (H @ a[t]).item())
        m[t] = a[t] + gain[:, 0] * innovation
        identity_minus = np.eye(d) - gain @ H
        C[t] = _symmetrize(
            identity_minus @ Rm[t] @ identity_minus.T
            + observation_variance * (gain @ gain.T)
        )
    return GaussianFilter1D(
        filtered_mean=m,
        filtered_covariance=C,
        predicted_mean=a,
        predicted_covariance=Rm,
        transition=np.asarray(G, dtype=float),
        process_covariance=np.asarray(Q, dtype=float),
    )

def _draw_gaussian_smoother_1d(
    result: GaussianFilter1D,
    rng: np.random.Generator,
    *,
    jitter: float = 1e-12,
) -> Array:
    Tn, d = result.filtered_mean.shape[0] - 1, result.filtered_mean.shape[1]
    path = np.zeros((Tn + 1, d), dtype=float)
    path[Tn] = _sample_gaussian(
        result.filtered_mean[Tn], result.filtered_covariance[Tn], rng, jitter=jitter
    )
    for t in range(Tn - 1, -1, -1):
        mean, covariance = _backward_smoothing_moments(
            result.filtered_mean[t],
            result.filtered_covariance[t],
            path[t + 1],
            result.transition,
            result.process_covariance,
            result.predicted_covariance[t + 1],
            jitter=jitter,
        )
        path[t] = _sample_gaussian(mean, covariance, rng, jitter=jitter)
    return path

def _mean_gaussian_smoother_1d(
    result: GaussianFilter1D,
    *,
    jitter: float = 1e-12,
) -> Array:
    smooth = result.filtered_mean.copy()
    for t in range(smooth.shape[0] - 2, -1, -1):
        predicted = result.predicted_covariance[t + 1]
        if np.linalg.norm(predicted) <= np.finfo(float).tiny:
            continue
        gain = spd_solve(
            predicted,
            result.transition @ result.filtered_covariance[t],
            jitter=jitter,
        ).T
        smooth[t] = result.filtered_mean[t] + gain @ (
            smooth[t + 1] - result.predicted_mean[t + 1]
        )
    return smooth

def ffbs_gaussian_1d_tvR(
    y: Array,
    G: Array,
    Q: Array,
    H: Array,
    R_t: Array,
    *,
    m0: Optional[Array] = None,
    C0: Optional[Array] = None,
    rng: Optional[np.random.Generator] = None,
    jitter: float = 1e-12,
    R_floor: float = 0.0,
    filter_result: Optional[GaussianFilter1D] = None,
) -> Array:
    if rng is None:
        rng = np.random.default_rng()
    if filter_result is None and C0 is None:
        C0 = 1e-6 * np.eye(np.asarray(G).shape[0])
    result = filter_result or _filter_gaussian_1d_tvR(
        y,
        G,
        Q,
        H,
        R_t,
        m0=m0,
        C0=C0,
        jitter=jitter,
        R_floor=R_floor,
    )
    return _draw_gaussian_smoother_1d(result, rng, jitter=jitter)

def gaussian_smoother_mean_1d_tvR(
    y: Array,
    G: Array,
    Q: Array,
    H: Array,
    R_t: Array,
    *,
    m0: Optional[Array] = None,
    C0: Optional[Array] = None,
    jitter: float = 1e-12,
    R_floor: float = 0.0,
) -> Array:
    """Kalman/RTS posterior mean for a scalar, time-varying Gaussian layer."""
    result = _filter_gaussian_1d_tvR(
        y,
        G,
        Q,
        H,
        R_t,
        m0=m0,
        C0=C0,
        jitter=jitter,
        R_floor=R_floor,
    )
    return _mean_gaussian_smoother_1d(result, jitter=jitter)

def design_matrix_ncp(
    z_path: Array,
    layout: NCPLayout,
    *,
    center_time: bool = True,
) -> tuple[Array, list[str], float]:
    """Build the Fruehwirth-Schnatter regression design.

    Time is centred by default. This is important for long records: independent
    priors on ``alpha0`` and ``beta0`` induce a correlated prior on the centred
    intercept ``alpha_c = alpha0 + tbar * beta0``.
    """
    z_path = np.asarray(z_path, dtype=float)
    Tn = z_path.shape[0] - 1
    cols: list[Array] = []
    names: list[str] = []

    t1 = np.arange(1, Tn + 1, dtype=float)
    tbar = float(t1.mean()) if center_time else 0.0
    cols.append(np.ones(Tn, dtype=float))
    names.append("alpha_c" if center_time and layout.has_beta else "alpha0")
    if layout.has_beta:
        cols.append(t1 - tbar)
        names.append("beta0")

    if layout.season_dim > 0:
        seasonal_design = static_seasonal_design(Tn, layout.season_dim)
        for j in range(layout.season_dim):
            cols.append(seasonal_design[:, j])
            names.append(f"gamma0_season_{j+1}")

    cols.append(z_path[1:, int(layout.idx_tilde_alpha)])
    names.append("s_level")

    if layout.has_beta:
        cols.append(z_path[1:, int(layout.idx_A)])
        names.append("s_trend")

    if layout.season_dim > 0:
        cols.append(z_path[1:, layout.season_ncp_slice.start])
        names.append("s_season")

    X = np.column_stack(cols) if cols else np.zeros((Tn, 0), dtype=float)
    return X, names, tbar


def apply_theta_draw(
    params_state: ParamDict,
    theta: Array,
    theta_names: list[str],
    layout: NCPLayout,
    *,
    tbar: float = 0.0,
) -> ParamDict:
    out = dict(params_state)
    season_vals = np.zeros(layout.season_dim, dtype=float)
    alpha_c: Optional[float] = None
    beta0 = float(out.get("beta0", 0.0))
    for val, name in zip(theta, theta_names):
        if name == "alpha_c":
            alpha_c = float(val)
        elif name == "beta0":
            beta0 = float(val)
            out[name] = beta0
        elif name.startswith("gamma0_season_"):
            j = int(name.rsplit("_", 1)[1]) - 1
            season_vals[j] = float(val)
        else:
            out[name] = float(val)
    if alpha_c is not None:
        out["alpha0"] = alpha_c - float(tbar) * beta0
    if layout.season_dim > 0:
        out["gamma0_season"] = season_vals
    out["q_level"] = float(out.get("s_level", 0.0)) ** 2
    out["q_trend"] = float(out.get("s_trend", 0.0)) ** 2
    out["q_season"] = float(out.get("s_season", 0.0)) ** 2
    return out

def theta_vector_from_params(
    params_state: ParamDict,
    theta_names: list[str],
    layout: NCPLayout,
    *,
    tbar: float,
) -> Array:
    """Extract the FS regression vector in the same coordinates as its design."""
    values = np.zeros(len(theta_names), dtype=float)
    gamma = np.asarray(
        params_state.get("gamma0_season", np.zeros(layout.season_dim)), dtype=float
    ).reshape(-1)
    beta0 = float(params_state.get("beta0", 0.0))
    for index, name in enumerate(theta_names):
        if name == "alpha_c":
            values[index] = float(params_state.get("alpha0", 0.0)) + tbar * beta0
        elif name == "alpha0":
            values[index] = float(params_state.get("alpha0", 0.0))
        elif name == "beta0":
            values[index] = beta0
        elif name.startswith("gamma0_season_"):
            component = int(name.rsplit("_", 1)[1]) - 1
            values[index] = float(gamma[component])
        else:
            values[index] = float(params_state.get(name, 0.0))
    return values

