"""Exact conditional updates of shared FS normal-prior scales.

The update is on u=log(scale/anchor). In NCP coordinates the scale enters only
the coefficient prior. Its normalization contributes -J*u. Half-family priors
also require the +u change-of-variable term; a lognormal prior is already
Gaussian in u. Standardized paths do not enter this conditional density.
"""
from dataclasses import dataclass

import numpy as np

from ...priors.shrinkage import COEFFICIENT_FIELDS, NORMAL_ABSOLUTE_MEDIAN, IndependentShrinkage
from .fs_utils import _active_scale_names, _slice_sample_real


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
    def medians(self):
        return {c: float(anchor*np.exp(self.log_multipliers[c]))
                for c, anchor in self.specification.anchors.items()}

    def update(self, rng):
        metrics = {}
        for component, anchor in self.specification.anchors.items():
            coefficients = [s.params_state[COEFFICIENT_FIELDS[component]] for s in self.members[component]]
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
