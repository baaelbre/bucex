"""Generalized-extreme-value observations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy.stats import genextreme

from .base import ObsSpec, resolve_observation_parameters
from .scale import SeasonalScale, LogScale, StructuralScale


@dataclass(frozen=True)
class GEV:
    """GEV observations using the EVT shape convention ``xi``.

    SciPy uses ``c=-xi``.  Both legacy ``params={"sigma", "xi"}`` and
    explicit keyword calls are accepted. ``phi`` describes the model for
    ``phi_t = log(sigma_t)``. The paper model uses a stationary baseline
    with optional repeating seasonal effects on the observation scale.
    """

    xi_bounds: tuple[float | None, float | None] | None = None
    phi: str = "stationary"
    scale: SeasonalScale | LogScale | StructuralScale | None = None
    name: str = field(default="gev", init=False)
    spec: ObsSpec = field(default=ObsSpec("gev"), init=False, repr=False)

    def __post_init__(self) -> None:
        bounds = (None, None) if self.xi_bounds is None else self.xi_bounds
        if len(bounds) != 2:
            raise ValueError("GEV xi_bounds needs two endpoints, or None.")
        lo = -np.inf if bounds[0] is None else float(bounds[0])
        hi = np.inf if bounds[1] is None else float(bounds[1])
        if not lo < hi:
            raise ValueError("GEV xi_bounds must satisfy lower < upper.")
        object.__setattr__(self, "xi_bounds", (lo, hi))
        mode = str(self.phi).lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "constant": "stationary",
            "fixed": "stationary",
        }
        mode = aliases.get(mode, mode)
        if mode != "stationary":
            raise ValueError("BUCEX 1.8.5 supports stationary GEV phi only.")
        object.__setattr__(self, "phi", mode)
        if self.scale is not None:
            if not isinstance(self.scale, (SeasonalScale, LogScale)):
                raise TypeError("scale must be SeasonalScale(...), LogScale(...), or None.")
            if mode != "stationary":
                raise ValueError("SeasonalScale currently requires phi='stationary'; dynamic phi remains available without SeasonalScale.")

    @staticmethod
    def _parameters(params, sigma, xi):
        sigma, xi = resolve_observation_parameters(params, sigma=sigma, xi=xi)
        if xi is None:
            raise TypeError("GEV observations require shape parameter 'xi'.")
        return sigma, xi

    def logpdf(self, y: Any, eta: Any, params=None, *, sigma=None, xi=None):
        sigma, xi = self._parameters(params, sigma, xi)
        if np.any(np.asarray(sigma) <= 0.0):
            return np.full(
                np.broadcast_shapes(
                    np.shape(y), np.shape(eta), np.shape(sigma), np.shape(xi)
                ),
                -np.inf,
            )
        value = genextreme.logpdf(y, c=-np.asarray(xi), loc=eta, scale=sigma)
        return float(value) if np.ndim(value) == 0 else value

    def cdf(self, y: Any, eta: Any, params=None, *, sigma=None, xi=None):
        sigma, xi = self._parameters(params, sigma, xi)
        return genextreme.cdf(y, c=-np.asarray(xi), loc=eta, scale=sigma)

    def ppf(self, probability: Any, eta: Any, params=None, *, sigma=None, xi=None):
        sigma, xi = self._parameters(params, sigma, xi)
        return genextreme.ppf(probability, c=-np.asarray(xi), loc=eta, scale=sigma)

    def sample(
        self,
        eta: Any,
        params: Mapping[str, float] | None = None,
        rng: np.random.Generator | None = None,
        *,
        sigma: float | None = None,
        xi: float | None = None,
    ):
        sigma, xi = self._parameters(params, sigma, xi)
        rng = np.random.default_rng() if rng is None else rng
        value = genextreme.rvs(
            c=-xi,
            loc=eta,
            scale=sigma,
            size=np.shape(eta),
            random_state=rng,
        )
        return float(value) if np.ndim(value) == 0 else value

    def grad_eta(self, y: Any, eta: Any, params=None, *, sigma=None, xi=None):
        sigma, xi = self._parameters(params, sigma, xi)
        y_arr, eta_arr, sigma_arr, xi_arr = np.broadcast_arrays(
            np.asarray(y, dtype=float),
            np.asarray(eta, dtype=float),
            np.asarray(sigma, dtype=float),
            np.asarray(xi, dtype=float),
        )
        value = np.full(y_arr.shape, np.nan)
        positive = sigma_arr > 0.0
        z = np.zeros_like(y_arr)
        z[positive] = (y_arr[positive] - eta_arr[positive]) / sigma_arr[positive]
        gumbel = positive & (np.abs(xi_arr) < 1e-7)
        value[gumbel] = (1.0 - np.exp(-z[gumbel])) / sigma_arr[gumbel]
        general = positive & ~gumbel
        support = 1.0 + xi_arr * z
        ok = general & (support > 0.0)
        t = support[ok]
        shape = xi_arr[ok]
        value[ok] = (
            (1.0 + shape) / t - t ** (-1.0 / shape - 1.0)
        ) / sigma_arr[ok]
        return float(value) if np.ndim(value) == 0 else value

    def hess_eta(self, y: Any, eta: Any, params=None, *, sigma=None, xi=None):
        sigma, xi = self._parameters(params, sigma, xi)
        y_arr, eta_arr, sigma_arr, xi_arr = np.broadcast_arrays(
            np.asarray(y, dtype=float),
            np.asarray(eta, dtype=float),
            np.asarray(sigma, dtype=float),
            np.asarray(xi, dtype=float),
        )
        value = np.full(y_arr.shape, np.nan)
        positive = sigma_arr > 0.0
        z = np.zeros_like(y_arr)
        z[positive] = (y_arr[positive] - eta_arr[positive]) / sigma_arr[positive]
        gumbel = positive & (np.abs(xi_arr) < 1e-7)
        value[gumbel] = -np.exp(-z[gumbel]) / sigma_arr[gumbel] ** 2
        general = positive & ~gumbel
        support = 1.0 + xi_arr * z
        ok = general & (support > 0.0)
        t = support[ok]
        shape = xi_arr[ok]
        value[ok] = ((1.0 + shape) / sigma_arr[ok] ** 2) * (
            shape / t**2 - t ** (-1.0 / shape - 2.0)
        )
        return float(value) if np.ndim(value) == 0 else value

    def grad_phi(self, y: Any, eta: Any, params=None, *, sigma=None, xi=None):
        """Derivative of the log density with respect to ``phi=log(sigma)``."""

        sigma, xi = self._parameters(params, sigma, xi)
        y_arr, eta_arr, sigma_arr, xi_arr = np.broadcast_arrays(
            np.asarray(y, dtype=float),
            np.asarray(eta, dtype=float),
            np.asarray(sigma, dtype=float),
            np.asarray(xi, dtype=float),
        )
        value = np.full(y_arr.shape, np.nan)
        positive = sigma_arr > 0.0
        z = np.zeros_like(y_arr)
        z[positive] = (y_arr[positive] - eta_arr[positive]) / sigma_arr[positive]
        gumbel = positive & (np.abs(xi_arr) < 1e-7)
        zg = z[gumbel]
        value[gumbel] = -1.0 + zg * (1.0 - np.exp(-zg))
        general = positive & ~gumbel
        support = 1.0 + xi_arr * z
        ok = general & (support > 0.0)
        t = support[ok]
        shape = xi_arr[ok]
        zz = z[ok]
        score = (1.0 + shape) / t - t ** (-1.0 / shape - 1.0)
        value[ok] = -1.0 + zz * score
        return float(value) if np.ndim(value) == 0 else value

    def hess_phi(self, y: Any, eta: Any, params=None, *, sigma=None, xi=None):
        """Second derivative of the log density with respect to log scale."""

        sigma, xi = self._parameters(params, sigma, xi)
        y_arr, eta_arr, sigma_arr, xi_arr = np.broadcast_arrays(
            np.asarray(y, dtype=float),
            np.asarray(eta, dtype=float),
            np.asarray(sigma, dtype=float),
            np.asarray(xi, dtype=float),
        )
        value = np.full(y_arr.shape, np.nan)
        positive = sigma_arr > 0.0
        z = np.zeros_like(y_arr)
        z[positive] = (y_arr[positive] - eta_arr[positive]) / sigma_arr[positive]
        gumbel = positive & (np.abs(xi_arr) < 1e-7)
        zg = z[gumbel]
        exp_term = np.exp(-zg)
        value[gumbel] = -zg * (1.0 - exp_term) - zg**2 * exp_term
        general = positive & ~gumbel
        support = 1.0 + xi_arr * z
        ok = general & (support > 0.0)
        t = support[ok]
        shape = xi_arr[ok]
        zz = z[ok]
        score = (1.0 + shape) / t - t ** (-1.0 / shape - 1.0)
        score_derivative = (1.0 + shape) * (
            -shape / t**2 + t ** (-1.0 / shape - 2.0)
        )
        value[ok] = -zz * score - zz**2 * score_derivative
        return float(value) if np.ndim(value) == 0 else value

    def support_ok(self, y: Any, eta: Any, params=None, *, sigma=None, xi=None) -> bool:
        sigma, xi = self._parameters(params, sigma, xi)
        if np.any(np.asarray(sigma) <= 0.0):
            return False
        y_arr, eta_arr, sigma_arr, xi_arr = np.broadcast_arrays(
            np.asarray(y, dtype=float),
            np.asarray(eta, dtype=float),
            np.asarray(sigma, dtype=float),
            np.asarray(xi, dtype=float),
        )
        support = 1.0 + xi_arr * (y_arr - eta_arr) / sigma_arr
        return bool(np.all((np.abs(xi_arr) < 1e-12) | (support > 0.0)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.name,
            "xi_bounds": [v if np.isfinite(v) else None for v in self.xi_bounds],
            "phi": self.phi,
            **({"scale": self.scale.to_dict()} if self.scale else {}),
        }


GEVObs = GEV
