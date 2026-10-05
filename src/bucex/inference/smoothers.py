"""Gaussian filtering and simulation smoothing, including singular transitions.

Matrices come from the model compiler. No component-specific cases occur here.
A time-dependent observation design supports regressors and TVP components.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
Array = np.ndarray

def _symmetrize(a):
    return (a+a.T)/2

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
    designs = np.broadcast_to(np.asarray(H, dtype=float), (Tn, d))
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
        H = designs[t-1:t]
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
        C0 = np.zeros_like(G, dtype=float)
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


# Public numerical entry points. States include the unobserved time zero.
filter_gaussian = _filter_gaussian_1d_tvR
smooth_gaussian = gaussian_smoother_mean_1d_tvR
sample_gaussian = ffbs_gaussian_1d_tvR
