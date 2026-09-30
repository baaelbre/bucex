"""Conditional updates of shared FS normal-prior scales.

Half-normal scales use direct generalized inverse Gaussian (GIG) Gibbs draws
for their squares. Other hyperprior families retain slice updates in
u=log(scale/anchor). Standardized state paths do not enter either conditional.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import geninvgauss

from ...priors.shrinkage import COEFFICIENT_FIELDS, NORMAL_ABSOLUTE_MEDIAN, IndependentShrinkage
from .fs_utils import _active_scale_names, _slice_sample_real


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


def shared_scale_log_target(u, coefficients, *, anchor, log_sd=float(np.log(2.)),
                            scale_conversion=NORMAL_ABSOLUTE_MEDIAN,
                            hyperprior="lognormal", df=4.):
    """Log density w.r.t. du, including the coefficient-scale normalization."""
    if not np.isfinite(u):
        return -np.inf
    coefficients = np.asarray(coefficients, dtype=float)
    square = float(np.sum((coefficients * scale_conversion / anchor)**2))
    with np.errstate(over="ignore", invalid="ignore"):
        penalty = 0. if square == 0. else square * np.exp(-2*u)
        if hyperprior == "lognormal":
            value = -.5*(u/log_sd)**2 - coefficients.size*u - .5*penalty
        elif hyperprior == "half_normal":
            # d tau / du = tau: -(K-1)u, not -K u.
            value = -(coefficients.size-1)*u - .5*penalty - .5*np.exp(2*u)
        elif hyperprior in {"half_t", "half_cauchy"}:
            nu = 1. if hyperprior == "half_cauchy" else df
            value = (-(coefficients.size-1)*u - .5*penalty
                     - .5*(nu+1)*np.logaddexp(0., 2*u-np.log(nu)))
        else:
            raise ValueError("Unknown shared-scale hyperprior.")
    return float(value) if np.isfinite(value) else -np.inf


@dataclass
class SharedShrinkageState:
    specification: object
    log_multipliers: dict
    members: dict

    @classmethod
    def initialize(cls, specification, states, rng, initial=None):
        independent = isinstance(specification, IndependentShrinkage)
        if independent and len(states) != 1:
            raise ValueError("IndependentShrinkage requires exactly one channel.")
        prefix = "independent" if independent else "shared"
        members = {}
        for component in specification.anchors:
            legacy = COEFFICIENT_FIELDS[component].removeprefix("s_")
            members[component] = [state for state in states if
                (state.layout.has_beta if component == "initial_slope"
                 else legacy in _active_scale_names(state.layout))]
            if len(members[component]) < (1 if independent else 2):
                raise ValueError(f"SharedShrinkage {component} needs at least two channels with that active coefficient; disable pooling for absent components.")
        multipliers = {}
        for component, anchor in specification.anchors.items():
            value = (initial or {}).get(f"shrinkage.{prefix}.{component}")
            if value is None:
                scale = specification.sample_medians(1, rng=rng)[component][0]
                multipliers[component] = float(np.log(max(scale, np.finfo(float).tiny)/anchor))
            elif not np.isfinite(value) or value <= 0:
                raise ValueError("A shared-shrinkage warm start must contain positive finite medians.")
            else:
                multipliers[component] = float(np.log(value/anchor))
        return cls(specification, multipliers, members)

    @property
    def scope(self):
        return "independent" if isinstance(self.specification, IndependentShrinkage) else "shared"

    @property
    def update_method(self):
        return "gig" if self.specification.hyperprior == "half_normal" else "slice"

    @property
    def medians(self):
        return {c: float(anchor*np.exp(self.log_multipliers[c]))
                for c, anchor in self.specification.anchors.items()}

    def update(self, rng):
        metrics = {}
        for component, anchor in self.specification.anchors.items():
            coefficients = [s.params_state[COEFFICIENT_FIELDS[component]] for s in self.members[component]]
            if self.update_method == "gig":
                # Direct-SD units in the reference. Also preserve the optional
                # legacy median-absolute convention for pooled initial slopes.
                normalized = np.asarray(coefficients) / self.specification.coefficient_sd(component, 1.)
                self.log_multipliers[component] = _half_normal_gig_log_multiplier(
                    normalized, anchor=anchor, rng=rng)
                metrics[f"{self.scope}_shrinkage_gig_draws.{component}"] = 1.
                continue
            target = lambda u: shared_scale_log_target(u, coefficients, anchor=anchor,
                log_sd=self.specification.log_sd,
                scale_conversion=1. / self.specification.coefficient_sd(component, 1.),
                hyperprior=self.specification.hyperprior, df=self.specification.df)
            self.log_multipliers[component], evaluations = _slice_sample_real(
                self.log_multipliers[component], target, rng, width=.5)
            metrics[f"{self.scope}_shrinkage_slice_evaluations.{component}"] = evaluations
        return metrics

    def parameter_values(self):
        return {f"shrinkage.{self.scope}.{c}": m for c, m in self.medians.items()}
