from dataclasses import dataclass
import numpy as np
from scipy.linalg import helmert

def contrasts(period):
    """Deterministic orthonormal zero-sum contrast matrix, shape (p, p-1)."""
    return helmert(period, full=False).T


@dataclass(frozen=True)
class SeasonalScale:
    """log(sigma[t]) = log(sigma) + C[phase[t]] @ u, u ~ N(0, prior_sd**2 I)."""
    period: int = 4
    prior_sd: float = .3

    def __post_init__(self):
        if isinstance(self.period, bool) or not isinstance(self.period, int) or self.period < 2:
            raise ValueError("SeasonalScale.period must be an integer >= 2.")
        if not np.isfinite(self.prior_sd) or self.prior_sd <= 0:
            raise ValueError("SeasonalScale.prior_sd must be positive and finite.")


