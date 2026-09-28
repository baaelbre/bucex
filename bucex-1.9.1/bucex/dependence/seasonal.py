"""Pooled periodic effects on Gaussian-copula partial correlations."""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import helmert

from .gaussian import (GaussianCopula, correlation_from_unconstrained,
                       gaussian_copula_logpdf, log_lkj_unconstrained)


@dataclass(frozen=True)
class SeasonalGaussianCopula(GaussianCopula):
    """A periodic Gaussian copula with a common baseline and shrunk effects.

    Fisher partial correlations equal a baseline plus calendar contrasts.
    The baseline matrix has an LKJ(eta) prior. Contrast coefficients have
    independent N(0, prior_sd**2) priors. Individual monthly matrices are
    positive definite but do NOT each have an LKJ prior. The seasonal prior
    depends on channel ordering; declare that order and assess permutations.

    ``structure='harmonic'`` uses one sine/cosine pair; ``'seasons'`` uses
    DJF/MAM/JJA/SON contrasts (period=12 for months, period=4 for seasonal
    blocks); ``'monthly'`` uses period-1 zero-sum
    contrasts. Effects shrink towards a constant, estimated correlation.
    Public phases are 1-based; dated monthly data use calendar months, and
    dated seasonal blocks use DJF=1, MAM=2, JJA=3, SON=4.
    """

    period: int = 12
    structure: str = "harmonic"
    prior_sd: float = 0.25

    def __post_init__(self):
        super().__post_init__()
        if self.correlation is not None:
            raise ValueError("SeasonalGaussianCopula estimates its baseline and effects; use GaussianCopula for fixed R.")
        if int(self.period) != self.period or self.period < 3:
            raise ValueError("period must be an integer >= 3.")
        if self.structure not in {"harmonic", "seasons", "monthly"}:
            raise ValueError("structure must be harmonic, seasons, or monthly.")
        if self.structure == "seasons" and self.period not in {4,12}:
            raise ValueError("Meteorological seasons require period=4 or period=12.")
        if not np.isfinite(self.prior_sd) or self.prior_sd <= 0:
            raise ValueError("prior_sd must be finite and positive.")

    @property
    def seasonal(self):
        return True

    def contrast(self):
        if self.structure == "harmonic":
            angle = 2*np.pi*np.arange(self.period)/self.period
            return np.column_stack((np.cos(angle), np.sin(angle)))
        if self.structure == "seasons":
            if self.period == 4:
                return helmert(4, full=False).T
            return helmert(4, full=False).T[((np.arange(12)+1) % 12)//3]
        return helmert(self.period, full=False).T

    def phases(self, n_time, dates=None, *, start_index=0):
        if self.structure == "seasons" and self.period == 4:
            from ..core.calendar import meteorological_phases
            if dates is None or len(dates) != n_time:
                raise ValueError("Seasonal-block copulas require aligned dates.")
            return meteorological_phases(dates)
        from ..core.calendar import seasonal_phases
        return seasonal_phases(self.period, n_time, dates, start_index=start_index)

    def parameter_names(self, channel_names):
        baseline = GaussianCopula.parameter_names(self, channel_names)
        effects = tuple(f"copula.effect.{i}.{j}.{h}"
                        for i in range(len(channel_names)) for j in range(i)
                        for h in range(self.contrast().shape[1]))
        return baseline + effects

    def sample_parameters(self, channel_names, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        result = GaussianCopula(eta=self.eta).sample_parameters(channel_names, rng)
        result.update({key: float(rng.normal(0, self.prior_sd))
                       for key in self.parameter_names(channel_names) if key.startswith("copula.effect.")})
        return result

    def correlation_matrix(self, params, channel_names, *, phase=None):
        baseline_names = GaussianCopula.parameter_names(self, channel_names)
        baseline = np.array([params[key] for key in baseline_names])
        d = self.contrast().shape[1]
        effects = np.array([params[key] for key in self.parameter_names(channel_names)
                            if key.startswith("copula.effect.")]).reshape(len(baseline), d)
        if phase is not None:
            if int(phase) != phase or not 1 <= phase <= self.period:
                raise ValueError("phase must be an integer in 1,...,period.")
            return correlation_from_unconstrained(baseline + effects @ self.contrast()[int(phase)-1], len(channel_names))
        return np.stack([correlation_from_unconstrained(baseline + effects @ row, len(channel_names))
                         for row in self.contrast()])

    def correlation_path(self, params, channel_names, n_time, dates=None, *, start_index=0):
        return self.correlation_matrix(params, channel_names)[self.phases(n_time, dates, start_index=start_index)-1]

    def logpdf(self, scores, params, channel_names, *, dates=None, start_index=0):
        values = np.asarray(scores)
        if values.ndim != 2:
            raise ValueError("Seasonal copula logpdf expects a time by channel matrix.")
        phases = self.phases(len(values), dates, start_index=start_index)
        matrices = self.correlation_matrix(params, channel_names)
        result = np.empty(len(values))
        for phase in np.unique(phases):
            selected = phases == phase
            result[selected] = gaussian_copula_logpdf(values[selected], matrices[phase-1])
        return result

    def log_prior(self, params, channel_names):
        baseline = [params[key] for key in GaussianCopula.parameter_names(self, channel_names)]
        effects = np.array([params[key] for key in self.parameter_names(channel_names)
                            if key.startswith("copula.effect.")])
        return (log_lkj_unconstrained(baseline, len(channel_names), self.eta)
                - .5*np.sum((effects/self.prior_sd)**2)
                - len(effects)*np.log(self.prior_sd*np.sqrt(2*np.pi)))

    def to_dict(self):
        return {"family": "gaussian", "seasonal": True, "eta": self.eta,
                "period": self.period, "structure": self.structure, "prior_sd": self.prior_sd}
