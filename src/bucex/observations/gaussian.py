from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm, genextreme
from .scale import SeasonalScale

@dataclass(frozen=True)
class Gaussian:
    scale: SeasonalScale | None = None
    name = "gaussian"
    gaussian_location = True

    def __post_init__(self):
        if self.scale is not None and not isinstance(self.scale, SeasonalScale):
            raise TypeError("scale must be SeasonalScale or None for constant dispersion.")

    def logpdf(self, y, eta, params):
        return norm.logpdf(y, loc=eta, scale=params['sigma'])

    def cdf(self, y, eta, params):
        return norm.cdf(y, loc=eta, scale=params['sigma'])

    def sf(self, y, eta, params):
        return norm.sf(y, loc=eta, scale=params['sigma'])

    def ppf(self, p, eta, params):
        return norm.ppf(p, loc=eta, scale=params['sigma'])

    def grad_eta(self, y, eta, params):
        return (np.asarray(y) - eta) / np.asarray(params['sigma'])**2

    def hess_eta(self, y, eta, params):
        return np.broadcast_to(-1 / np.asarray(params['sigma'])**2, np.shape(y))


