"""Residual Gaussian copulas, with LKJ priors and explicit tail orientation.

Correlation matrices refer to the original response orientation, including
minima. The sampling coordinates are Fisher transforms of rowwise Cholesky
partial correlations. See the Stan Reference Manual, ``Constraint Transforms``
(correlation matrices), for the transform and Jacobian. No CDF clipping or
positive-definite matrix repair is used in likelihood calculations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import betaln, log_ndtr, ndtr, ndtri_exp


def validate_correlation(correlation: Any, n_channels: int | None = None) -> np.ndarray:
    """Validate an SPD correlation matrix; only remove roundoff asymmetry."""
    matrix = np.asarray(correlation, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("correlation must be a nonempty square matrix.")
    if n_channels is not None and matrix.shape != (n_channels, n_channels):
        raise ValueError(f"correlation must have shape ({n_channels}, {n_channels}).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("correlation must contain only finite values.")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0):
        raise ValueError("correlation must be symmetric.")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-12, rtol=0):
        raise ValueError("correlation must have unit diagonal.")
    matrix = (matrix + matrix.T) * 0.5
    np.fill_diagonal(matrix, 1.0)
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("correlation must be strictly positive definite.") from exc
    return matrix


def _coordinates(theta: Any, n_channels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if int(n_channels) != n_channels or n_channels < 1:
        raise ValueError("n_channels must be a positive integer.")
    rows, columns = np.tril_indices(int(n_channels), k=-1)
    values = np.asarray(theta, dtype=float)
    if values.shape != (rows.size,) or not np.all(np.isfinite(values)):
        raise ValueError(f"theta must contain {rows.size} finite coordinates.")
    return values, rows, columns


def _log_sech_squared(theta: np.ndarray) -> np.ndarray:
    absolute = np.abs(theta)
    return 2.0 * (np.log(2.0) - absolute - np.log1p(np.exp(-2.0 * absolute)))


def correlation_from_unconstrained(theta: Any, n_channels: int) -> np.ndarray:
    """Map row-major strict-lower partial-correlation coordinates to R.

    ``theta[i,j]`` represents ``atanh(cor(i,j | 0,...,j-1))`` for j<i.
    Numerically singular proposals raise instead of silently adding jitter.
    """
    values, rows, columns = _coordinates(theta, n_channels)
    partial = np.zeros((n_channels, n_channels))
    log_remaining = np.zeros_like(partial)
    partial[rows, columns] = np.tanh(values)
    log_remaining[rows, columns] = 0.5 * _log_sech_squared(values)
    factor = np.zeros_like(partial)
    for row in range(n_channels):
        remaining = 1.0
        for column in range(row):
            factor[row, column] = partial[row, column] * remaining
            remaining *= np.exp(log_remaining[row, column])
        factor[row, row] = remaining
    try:
        return validate_correlation(factor @ factor.T)
    except ValueError as exc:
        raise FloatingPointError("Copula correlation is numerically singular.") from exc


def unconstrained_from_correlation(correlation: Any) -> np.ndarray:
    """Inverse of :func:`correlation_from_unconstrained`."""
    matrix = validate_correlation(correlation)
    factor = np.linalg.cholesky(matrix)
    values = []
    for row in range(matrix.shape[0]):
        remaining = 1.0
        for column in range(row):
            partial = factor[row, column] / remaining
            if not abs(partial) < 1.0:
                raise FloatingPointError("Correlation inverse reached a numerical boundary.")
            values.append(np.arctanh(partial))
            remaining *= np.sqrt((1.0 - partial) * (1.0 + partial))
    return np.asarray(values, dtype=float)


def log_lkj_unconstrained(theta: Any, n_channels: int, eta: float = 2.0) -> float:
    """Normalized LKJ(eta) log density in theta coordinates, WITH Jacobian.

    For zero-based column j, partial correlation z has density proportional
    to (1-z²)^(a-1), a=eta+(K-j-2)/2. Transforming z=tanh(theta) adds one
    to this exponent. Its normalizing constant is B(1/2,a).
    """
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("LKJ eta must be finite and positive.")
    values, _, columns = _coordinates(theta, n_channels)
    exponent = float(eta) + 0.5 * (n_channels - columns - 2)
    return float(np.sum(exponent * _log_sech_squared(values) - betaln(0.5, exponent)))


@dataclass(frozen=True)
class GaussianCopula:
    """Contemporaneous residual dependence with fixed or estimated correlation.

    ``GaussianCopula()`` estimates R with an LKJ(2) prior, shrinking toward
    independence. ``GaussianCopula(correlation=R)`` fixes R. Channels retain
    their Gaussian/GEV margins; this copula does not impose ordering or
    asymptotic tail dependence. R is in original response orientation.
    """

    eta: float = 2.0
    correlation: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.eta) or self.eta <= 0:
            raise ValueError("LKJ eta must be finite and positive.")
        object.__setattr__(self, "eta", float(self.eta))
        if self.correlation is not None:
            matrix = validate_correlation(self.correlation)
            object.__setattr__(self, "correlation", tuple(map(tuple, matrix.tolist())))

    @property
    def estimated(self) -> bool:
        return self.correlation is None

    @property
    def seasonal(self) -> bool:
        return False

    def parameter_names(self, channel_names: Sequence[str]) -> tuple[str, ...]:
        if not self.estimated:
            validate_correlation(self.correlation, len(channel_names))
            return ()
        return tuple(f"copula.z.{i}.{j}" for i in range(len(channel_names)) for j in range(i))

    def initial_parameters(self, channel_names: Sequence[str]) -> dict[str, float]:
        return dict.fromkeys(self.parameter_names(channel_names), 0.0)

    def sample_parameters(self, channel_names: Sequence[str],
                          rng: np.random.Generator | None = None) -> dict[str, float]:
        """Draw unconstrained correlation parameters from their exact LKJ prior."""
        names = self.parameter_names(channel_names)
        if not names:
            return {}
        generator = np.random.default_rng() if rng is None else rng
        _, columns = np.tril_indices(len(channel_names), k=-1)
        shape = self.eta + 0.5 * (len(channel_names) - columns - 2)
        partials = 2.0 * generator.beta(shape, shape) - 1.0
        if np.any(np.abs(partials) >= 1.0):
            raise FloatingPointError("LKJ prior draw reached a floating-point boundary.")
        return dict(zip(names, map(float, np.arctanh(partials))))

    def sample_correlation(self, channel_names: Sequence[str],
                           rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw R from LKJ, or return the declared fixed matrix."""
        return self.correlation_matrix(self.sample_parameters(channel_names, rng), channel_names)

    def correlation_matrix(self, params: Mapping[str, float], channel_names: Sequence[str], *, phase=None) -> np.ndarray:
        if not self.estimated:
            return validate_correlation(self.correlation, len(channel_names))
        theta = [params[name] for name in self.parameter_names(channel_names)]
        return correlation_from_unconstrained(theta, len(channel_names))

    def correlation_path(self, params, channel_names, n_time, dates=None, *, start_index=0):
        matrix = self.correlation_matrix(params, channel_names)
        return np.broadcast_to(matrix, (n_time, *matrix.shape))

    def logpdf(self, scores, params, channel_names, *, dates=None, start_index=0):
        return gaussian_copula_logpdf(scores, self.correlation_matrix(params, channel_names))

    def log_prior(self, params: Mapping[str, float], channel_names: Sequence[str]) -> float:
        if not self.estimated:
            return 0.0
        theta = [params[name] for name in self.parameter_names(channel_names)]
        return log_lkj_unconstrained(theta, len(channel_names), self.eta)

    def to_dict(self) -> dict[str, Any]:
        return {"family": "gaussian", "eta": self.eta,
                "correlation": None if self.correlation is None else [list(row) for row in self.correlation]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GaussianCopula:
        if value.get("family", "gaussian") != "gaussian":
            raise ValueError("Only a Gaussian residual copula is supported.")
        if value.get("seasonal", False):
            from .seasonal import SeasonalGaussianCopula
            return SeasonalGaussianCopula(**{k:v for k,v in value.items() if k not in {"family", "seasonal"}})
        return cls(eta=value.get("eta", 2.0), correlation=value.get("correlation"))


def _gev_scores(y: np.ndarray, eta: np.ndarray, sigma: float, xi: float) -> np.ndarray:
    standardized = (y - eta) / sigma
    if xi == 0.0:
        log_exponent = -standardized
    else:
        product = xi * standardized
        if np.any(product <= -1.0):
            raise ValueError("GEV observation is outside its open support.")
        log_exponent = -np.log1p(product) / xi
    score = np.empty_like(log_exponent)
    # F=exp(-a). Invert whichever tail has probability <= 1/2.
    left = log_exponent >= np.log(np.log(2.0))
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        score[left] = ndtri_exp(-np.exp(log_exponent[left]))
        log_survival = log_exponent[~left].copy()
        # Below -36, log(1-exp(-exp(v))) equals v to double precision.
        moderate = log_survival >= -36.0
        log_survival[moderate] = np.log(-np.expm1(-np.exp(log_survival[moderate])))
        score[~left] = -ndtri_exp(log_survival)
    if not np.all(np.isfinite(score)):
        raise FloatingPointError("GEV PIT normal score exceeds numerical precision; no clipping was applied.")
    return score


def normal_scores(y: Any, eta: Any, channels: Sequence[Any], params: Mapping[str, float]) -> np.ndarray:
    """Original-orientation PIT normal scores from INTERNAL observations/eta.

    Missing observations remain NaN. Gaussian scores are evaluated directly;
    GEV scores use log-tail arithmetic, retaining probabilities that would
    round to zero or one if the CDF were evaluated first.
    """
    observations, predictor = np.broadcast_arrays(np.asarray(y, float), np.asarray(eta, float))
    if observations.ndim < 1 or observations.shape[-1] != len(channels):
        raise ValueError("y and eta must have channel as their final dimension.")
    if np.any(np.isinf(observations)):
        raise ValueError("Infinite observations are invalid; use NaN for missing values.")
    scores = np.full_like(observations, np.nan)
    for j, channel in enumerate(channels):
        mask = np.isfinite(observations[..., j])
        if not np.any(mask):
            continue
        location = predictor[..., j][mask]
        sigma = np.broadcast_to(np.asarray(params[f"sigma.{channel.name}"], float), observations.shape[:-1])[mask]
        if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0) or not np.all(np.isfinite(location)):
            raise ValueError("Finite eta and a positive finite marginal scale are required.")
        if channel.family == "gaussian":
            score = (observations[..., j][mask] - location) / sigma
        elif channel.family == "gev":
            xi = float(params[f"xi.{channel.name}"])
            if not np.isfinite(xi):
                raise ValueError("GEV xi must be finite.")
            score = _gev_scores(observations[..., j][mask], location, sigma, xi)
        else:
            raise ValueError(f"Copula margins must be Gaussian or GEV; got {channel.family!r}.")
        if not np.all(np.isfinite(score)):
            raise FloatingPointError("Marginal PIT normal score exceeds numerical precision.")
        channel_scores = np.full(observations.shape[:-1], np.nan)
        channel_scores[mask] = float(channel.transform_sign) * score
        scores[..., j] = channel_scores
    return scores


def gaussian_copula_logpdf(scores: Any, correlation: Any) -> np.ndarray | float:
    """Copula log density at normal scores, marginalizing missing channels.

    Returns one value per leading index. NaNs denote unobserved margins;
    rows with zero or one observed margin have copula correction zero.
    Infinite observed scores are errors, never treated as missing.
    """
    matrices = np.asarray(correlation)
    if matrices.ndim == 3:
        values = np.asarray(scores)
        if values.ndim != 2 or len(values) != len(matrices):
            raise ValueError("Time-varying correlation requires matching time by channel scores.")
        unique, labels = np.unique(matrices, axis=0, return_inverse=True)
        result = np.empty(len(values))
        for group, matrix in enumerate(unique):
            selected = labels == group
            result[selected] = gaussian_copula_logpdf(values[selected], matrix)
        return result
    matrix = validate_correlation(correlation)
    values = np.asarray(scores, dtype=float)
    if values.ndim < 1 or values.shape[-1] != matrix.shape[0]:
        raise ValueError("scores must have final dimension equal to correlation dimension.")
    if np.any(np.isinf(values)):
        raise FloatingPointError("Copula normal scores must be finite or missing (NaN).")
    flat = values.reshape(-1, matrix.shape[0])
    masks, labels = np.unique(~np.isnan(flat), axis=0, return_inverse=True)
    output = np.zeros(flat.shape[0])
    for group, mask in enumerate(masks):
        if np.sum(mask) < 2:
            continue
        selection = labels == group
        observed = flat[selection][:, mask]
        submatrix = matrix[np.ix_(mask, mask)]
        if np.array_equal(submatrix, np.eye(submatrix.shape[0])):
            continue
        factor = np.linalg.cholesky(submatrix)
        transformed = solve_triangular(factor, observed.T, lower=True, check_finite=False).T
        output[selection] = -np.sum(np.log(np.diag(factor))) - 0.5 * np.sum(
            (transformed - observed) * (transformed + observed), axis=1)
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("Copula log density exceeds numerical precision.")
    result = output.reshape(values.shape[:-1])
    return float(result) if result.ndim == 0 else result


def copula_log_likelihood(y: Any, eta: Any, channels: Sequence[Any],
                          params: Mapping[str, float], correlation: Any) -> float:
    """Sum copula corrections; impossible/numerically invalid proposals get -inf."""
    try:
        return float(np.sum(gaussian_copula_logpdf(normal_scores(y, eta, channels, params), correlation)))
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return -np.inf


def copula_observation_derivatives(y: Any, eta: Any, channels: Sequence[Any],
                                   params: Mapping[str, float], correlation: Any
                                   ) -> tuple[np.ndarray, np.ndarray]:
    """Full marginal-plus-copula derivatives with respect to INTERNAL eta.

    For (T,K) observations, returns gradient (T,K) and Hessian (T,K,K).
    Missing coordinates have zero derivatives; the observed principal
    submatrix of R supplies their correctly marginalized joint likelihood.
    Other leading dimensions, including a single (K,) vector, are accepted.

    If z_j=s_j*Phi^-1(F_j(y_j|eta_j)), J_j=dz_j/deta_j, and
    A=R^-1-I, the copula gradient is J*(-Az). Its Hessian is
    diag(z''*(-Az))-diag(J) A diag(J). Marginal derivatives are added.
    This retains cross-channel curvature for the joint Laplace proposal.
    """
    observations, predictor = np.broadcast_arrays(np.asarray(y, float), np.asarray(eta, float))
    matrix = validate_correlation(correlation, len(channels))
    scores = normal_scores(observations, predictor, channels, params)
    leading = observations.shape[:-1]
    k = len(channels)
    observations = observations.reshape(-1, k)
    predictor = predictor.reshape(-1, k)
    scores = scores.reshape(-1, k)
    observed = ~np.isnan(observations)
    gradient = np.zeros_like(observations)
    diagonal = np.zeros_like(observations)
    first = np.zeros_like(observations)
    second = np.zeros_like(observations)
    for j, channel in enumerate(channels):
        mask = observed[:, j]
        if not np.any(mask):
            continue
        values, location = observations[mask, j], predictor[mask, j]
        sigma = float(params[f"sigma.{channel.name}"])
        sign = float(channel.transform_sign)
        kwargs = {"sigma": sigma, "xi": params.get(f"xi.{channel.name}")}
        gradient[mask, j] = channel.observation.grad_eta(values, location, **kwargs)
        diagonal[mask, j] = channel.observation.hess_eta(values, location, **kwargs)
        if channel.family == "gaussian":
            first[mask, j] = -sign / sigma
            continue
        q = sign * scores[mask, j]
        logpdf = channel.observation.logpdf(values, location, **kwargs)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            density_ratio = np.exp(logpdf + .5 * q**2 + .5 * np.log(2 * np.pi))
            first[mask, j] = -sign * density_ratio
            second[mask, j] = sign * (
                -density_ratio * gradient[mask, j] + q * density_ratio**2)
    hessian = np.zeros((observations.shape[0], k, k))
    coordinate = np.arange(k)
    hessian[:, coordinate, coordinate] = diagonal
    patterns, labels = np.unique(observed, axis=0, return_inverse=True)
    for group, mask in enumerate(patterns):
        if mask.sum() < 2:
            continue
        rows = np.flatnonzero(labels == group)
        columns = np.flatnonzero(mask)
        block = np.ix_(rows, columns)
        submatrix = matrix[np.ix_(columns, columns)]
        precision_difference = cho_solve((np.linalg.cholesky(submatrix), True),
                                         np.eye(columns.size), check_finite=False) - np.eye(columns.size)
        score_gradient = -scores[block] @ precision_difference.T
        jacobian = first[block]
        gradient[block] += jacobian * score_gradient
        addition = (-precision_difference[None, :, :]
                    * jacobian[:, :, None] * jacobian[:, None, :])
        local_diagonal = np.arange(columns.size)
        addition[:, local_diagonal, local_diagonal] += second[block] * score_gradient
        hessian[np.ix_(rows, columns, columns)] += addition
    if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(hessian)):
        raise FloatingPointError("Copula observation derivatives exceed numerical precision.")
    return gradient.reshape((*leading, k)), hessian.reshape((*leading, k, k))


def sample_normal_scores(correlation: Any, size: int | tuple[int, ...] = 1,
                         rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw correlated standard-normal residual scores in original orientation."""
    matrix = validate_correlation(correlation)
    shape = (size,) if isinstance(size, (int, np.integer)) else tuple(size)
    generator = np.random.default_rng() if rng is None else rng
    return generator.normal(size=(*shape, matrix.shape[0])) @ np.linalg.cholesky(matrix).T


def sample_uniforms(correlation: Any, size: int | tuple[int, ...] = 1,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw copula uniforms; use normal scores for extreme-safe inverse margins.

    Floating-point CDF values may equal zero/one in extreme tails. They are
    deliberately not clipped; ``quantiles_from_normal_scores`` avoids this.
    """
    return ndtr(sample_normal_scores(correlation, size=size, rng=rng))


def quantiles_from_normal_scores(scores: Any, eta: Any, channels: Sequence[Any],
                                 params: Mapping[str, float]) -> np.ndarray:
    """Map ORIGINAL-orientation copula scores to INTERNAL observations.

    The final axis is channel. GEV quantiles use log-tail arithmetic without
    converting a large positive normal score into a rounded probability one.
    """
    scores, predictor = np.broadcast_arrays(np.asarray(scores, float), np.asarray(eta, float))
    if scores.ndim < 1 or scores.shape[-1] != len(channels):
        raise ValueError("scores and eta must have channel as their final dimension.")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(predictor)):
        raise ValueError("scores and eta must be finite.")
    output = np.empty_like(scores)
    for j, channel in enumerate(channels):
        sigma = float(params[f"sigma.{channel.name}"])
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Marginal scale must be finite and positive.")
        internal_score = float(channel.transform_sign) * scores[..., j]
        if channel.family == "gaussian":
            standardized = internal_score
        elif channel.family == "gev":
            xi = float(params[f"xi.{channel.name}"])
            if not np.isfinite(xi):
                raise ValueError("GEV xi must be finite.")
            log_survival = log_ndtr(-internal_score)
            tiny_survival = log_survival < -36.0
            log_exponent = np.array(log_survival, copy=True)
            log_exponent[~tiny_survival] = np.log(-log_ndtr(internal_score[~tiny_survival]))
            with np.errstate(over="ignore", invalid="ignore"):
                standardized = -log_exponent if xi == 0 else np.expm1(-xi * log_exponent) / xi
        else:
            raise ValueError(f"Copula margins must be Gaussian or GEV; got {channel.family!r}.")
        output[..., j] = predictor[..., j] + sigma * standardized
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("Marginal copula quantile exceeds numerical precision.")
    return output
