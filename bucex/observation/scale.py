"""Identifiable calendar effects on the observation scale."""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import helmert


@dataclass(frozen=True)
class SeasonalScale:
    """Static, zero-sum seasonal effects on log observation scale.

    ``log(sigma_t) = log(sigma) + effect[phase_t]``. Orthonormal contrast
    coefficients have independent N(0, prior_sd**2) priors. Thus each effect
    has variance prior_sd**2 * (1 - 1/period), with no reference season.
    For period=12, dated observations use their calendar month. This is
    observation noise, separate from latent location seasonality.
    """

    period: int = 12
    prior_sd: float = 0.3

    def __post_init__(self):
        if int(self.period) != self.period or self.period < 2:
            raise ValueError("SeasonalScale.period must be an integer >= 2.")
        if not np.isfinite(self.prior_sd) or self.prior_sd <= 0:
            raise ValueError("SeasonalScale.prior_sd must be finite and positive.")
        object.__setattr__(self, "period", int(self.period))
        object.__setattr__(self, "prior_sd", float(self.prior_sd))

    def contrast(self):
        """Columns are an orthonormal basis for the zero-sum subspace."""
        return helmert(self.period, full=False).T

    def phases(self, n_time, dates=None, *, start_index=0):
        from ..core.calendar import seasonal_phases
        return seasonal_phases(self.period, n_time, dates, start_index=start_index) - 1

    def to_dict(self):
        return {"period": self.period, "prior_sd": self.prior_sd}
