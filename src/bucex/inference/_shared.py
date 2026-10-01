from __future__ import annotations
import numpy as np
from scipy.stats import geninvgauss

def _half_normal_gig_log_multiplier(coefficients, *, anchor, rng):
    """Draw log(tau / A) given s_j ~ N(0, tau**2), tau ~ HalfNormal(A).

    For J active coefficients and S=sum(s_j**2), v=tau**2 has density
    v**((1-J)/2-1) exp(-(v/A**2 + S/v)/2). Thus
    v ~ GIG((1-J)/2, a=A**-2, b=S). SciPy uses a symmetric standardized
    GIG: X ~ geninvgauss((1-J)/2, sqrt(S)/A), v=A*sqrt(S)*X.

    Compute the relative norm and keep the result in log coordinates to avoid
    squaring tiny coefficients. No variance floor or proposal tuning is used.
    """
    values = np.asarray(coefficients, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("GIG shrinkage requires a nonempty vector of finite coefficients.")
    if not np.isfinite(anchor) or anchor <= 0:
        raise ValueError("GIG shrinkage requires a positive finite half-normal scale.")
    if not np.any(values):
        raise ValueError(
            "The half-normal scale conditional is improper when all active "
            "coefficients are exactly zero; initialize an active coefficient "
            "away from zero or declare the component static.")
    with np.errstate(over="ignore", under="ignore"):
        relative_norm = float(np.hypot.reduce(values / anchor))
    if not np.isfinite(relative_norm) or relative_norm <= 0:
        raise FloatingPointError("Coefficient norm relative to the half-normal scale is not representable.")
    log_norm = np.log(relative_norm)
    if values.size > 1 and relative_norm < .1:
        # For small r, SciPy's ratio-of-uniforms bounding calculations can
        # overflow. An exact rejection sampler avoids that issue: with
        # W=S/(2*tau**2), the target is Gamma((J-1)/2, 1) tilted by
        # exp(-r**2/(4*W)) <= 1. Acceptance is high in this small-r branch.
        for _ in range(10000):
            precision = float(rng.gamma((values.size - 1.) / 2.))
            if precision <= 0:
                continue
            exponential = float(rng.exponential())
            if exponential > 0 and np.log(exponential) >= 2*log_norm - np.log(4.) - np.log(precision):
                return log_norm - .5 * (np.log(2.) + np.log(precision))
        raise FloatingPointError("GIG rejection sampling failed to produce a valid shrinkage draw.")
    draw = float(geninvgauss.rvs((1. - values.size) / 2., relative_norm,
                                random_state=rng))
    if not np.isfinite(draw) or draw <= 0:
        raise FloatingPointError("The GIG shrinkage sampler returned a nonpositive or nonfinite draw.")
    return .5 * (log_norm + np.log(draw))
