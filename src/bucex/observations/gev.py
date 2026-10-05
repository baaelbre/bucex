from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm, genextreme
from .scale import SeasonalScale

@dataclass(frozen=True)
class GEV:
    """xi is the EVT shape, c=-xi in SciPy. Lower tails use reflected maxima.

    Public eta, probabilities, quantiles and paths are always on the original
    response scale. Derivatives include the reflection; inference uses the original scale.
    """
    tail: str = "upper"
    scale: SeasonalScale | None = None
    name = "gev"
    gaussian_location = False

    def __post_init__(self):
        if self.tail not in {"upper", "lower"}:
            raise ValueError("GEV.tail must be upper or lower.")
        if self.scale is not None and not isinstance(self.scale, SeasonalScale):
            raise TypeError("scale must be SeasonalScale or None for constant dispersion.")

    @property
    def sign(self):
        return -1 if self.tail == "lower" else 1

    def logpdf(self, y, eta, params):
        return genextreme.logpdf(self.sign * np.asarray(y), c=-np.asarray(params['xi']),
                                loc=self.sign * np.asarray(eta), scale=params['sigma'])

    def cdf(self, y, eta, params):
        fn = genextreme.sf if self.tail == "lower" else genextreme.cdf
        return fn(self.sign * np.asarray(y), c=-np.asarray(params['xi']),
                  loc=self.sign * np.asarray(eta), scale=params['sigma'])

    def sf(self, y, eta, params):
        fn = genextreme.cdf if self.tail == "lower" else genextreme.sf
        return fn(self.sign * np.asarray(y), c=-np.asarray(params['xi']),
                  loc=self.sign * np.asarray(eta), scale=params['sigma'])

    def ppf(self, p, eta, params):
        fn = genextreme.isf if self.tail == "lower" else genextreme.ppf
        return self.sign * fn(p, c=-np.asarray(params['xi']),
                              loc=self.sign * np.asarray(eta), scale=params['sigma'])

    def grad_eta(self, y, eta, params):
        return self._derivatives(y, eta, params)[0]

    def hess_eta(self, y, eta, params):
        return self._derivatives(y, eta, params)[1]

    def _derivatives(self, y, eta, params):
        y, eta, sigma, xi = np.broadcast_arrays(self.sign * np.asarray(y, float),
            self.sign * np.asarray(eta, float), np.asarray(params['sigma'], float),
            np.asarray(params['xi'], float))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            z = (y - eta) / sigma
            t = 1 + xi*z
            # log1p gives a stable continuation toward the Gumbel limit.
            nonzero = xi != 0
            a = np.where(nonzero, -np.log1p(xi*z) / np.where(nonzero, xi, 1), -z)
            power = np.exp(a)
            grad = (1 + xi - power) / (sigma*t)
            hess = (1 + xi) * (xi - power) / (sigma*t)**2
            ok = (sigma > 0) & (t > 0)
        return np.where(ok, self.sign*grad, np.nan), np.where(ok, hess, np.nan)
