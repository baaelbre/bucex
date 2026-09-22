"""Exact conditional updates of shared FS normal-prior scales.

The update is on u=log(m/anchor), where the hyperprior is Gaussian. In NCP
coordinates m enters only the coefficient prior. Its normalizing constants
contribute -J*u; standardized paths do not enter this conditional density.
"""
from dataclasses import dataclass

import numpy as np

from ...priors.shrinkage import COMPONENT_FIELDS, NORMAL_ABSOLUTE_MEDIAN
from .fs_utils import _active_scale_names, _slice_sample_real


def shared_scale_log_target(u, coefficients, *, anchor, log_sd):
    """Log density w.r.t. du, including the coefficient-scale normalization."""
    if not np.isfinite(u):
        return -np.inf
    coefficients = np.asarray(coefficients, dtype=float)
    square = float(np.sum((coefficients * NORMAL_ABSOLUTE_MEDIAN / anchor)**2))
    with np.errstate(over="ignore", invalid="ignore"):
        penalty = 0. if square == 0. else square * np.exp(-2*u)
        value = -.5*(u/log_sd)**2 - coefficients.size*u - .5*penalty
    return float(value) if np.isfinite(value) else -np.inf


@dataclass
class SharedShrinkageState:
    specification: object
    log_multipliers: dict
    members: dict

    @classmethod
    def initialize(cls, specification, states, rng, initial=None):
        members = {}
        for component in specification.medians:
            legacy = COMPONENT_FIELDS[component].removeprefix("s_")
            members[component] = [state for state in states if legacy in _active_scale_names(state.layout)]
            if len(members[component]) < 2:
                raise ValueError(f"SharedShrinkage {component} needs at least two channels with that active innovation.")
        multipliers = {}
        for component, anchor in specification.medians.items():
            value = (initial or {}).get(f"shrinkage.shared.{component}")
            if value is None:
                multipliers[component] = float(rng.normal(0., specification.log_sd))
            elif not np.isfinite(value) or value <= 0:
                raise ValueError("A shared-shrinkage warm start must contain positive finite medians.")
            else:
                multipliers[component] = float(np.log(value/anchor))
        return cls(specification, multipliers, members)

    @property
    def medians(self):
        return {c: float(anchor*np.exp(self.log_multipliers[c]))
                for c, anchor in self.specification.medians.items()}

    def update(self, rng):
        metrics = {}
        for component, anchor in self.specification.medians.items():
            coefficients = [s.params_state[COMPONENT_FIELDS[component]] for s in self.members[component]]
            target = lambda u: shared_scale_log_target(u, coefficients, anchor=anchor,
                                                       log_sd=self.specification.log_sd)
            self.log_multipliers[component], evaluations = _slice_sample_real(
                self.log_multipliers[component], target, rng, width=.5)
            metrics[f"shared_shrinkage_slice_evaluations.{component}"] = evaluations
        return metrics

    def parameter_values(self):
        return {f"shrinkage.shared.{c}": m for c, m in self.medians.items()}
