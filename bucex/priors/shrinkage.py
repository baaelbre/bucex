"""Shared, uncertain normal-prior scales for private FS innovations."""
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
from scipy.special import ndtri

from .structural import NormalPrior


NORMAL_ABSOLUTE_MEDIAN = float(ndtri(.75))
COMPONENT_FIELDS = {"level": "s_level", "slope": "s_trend", "seasonal": "s_season"}


@dataclass(frozen=True)
class SharedShrinkage:
    """Learn one common normal-prior scale per selected component.

    ``medians`` are anchors for the *hyperprior*: log(m_c) is normal with
    mean log(anchor_c) and standard deviation ``log_sd``. Conditional on m_c,
    each active channel's signed FS coefficient has distribution
    N(0, (m_c / Phi^{-1}(.75))**2). Its absolute value is the physical process
    SD. Marginally the coefficients have a normal scale-mixture prior.

    The channels retain different process SDs and independent standardized
    state innovations. This hierarchy shares regularization, not a latent
    trajectory or residual correlation. Use a copula for the latter.

    Example::

        SharedShrinkage(medians={"level": .0025, "slope": .0000125})

    Components omitted from ``medians`` keep their declared channel priors.
    ``log_sd=log(2)`` places about 95% of each hyperprior between one quarter
    and four times its anchor. All anchors use the model's time/response units.
    """

    medians: Mapping[str, float]
    log_sd: float = float(np.log(2.))

    def __post_init__(self):
        aliases = {"trend": "slope", "season": "seasonal"}
        medians = {}
        for supplied, value in self.medians.items():
            key = aliases.get(supplied, supplied)
            if key not in COMPONENT_FIELDS or key in medians:
                raise ValueError("SharedShrinkage components must be unique level, slope or seasonal names.")
            if not np.isfinite(value) or value <= 0:
                raise ValueError("SharedShrinkage anchors must be positive and finite.")
            medians[key] = float(value)
        if not medians or not np.isfinite(self.log_sd) or self.log_sd <= 0:
            raise ValueError("Declare at least one anchor and a positive finite log_sd.")
        object.__setattr__(self, "medians", medians)
        object.__setattr__(self, "log_sd", float(self.log_sd))

    def validate_prior(self, prior):
        if any(getattr(prior, key, None) is not None
               for key in ("ssvs", "lasso", "pc", "horseshoe", "triple_gamma")):
            raise ValueError("SharedShrinkage requires continuous normal FS priors; omit SSVS and local-mixture priors.")
        for component in self.medians:
            coefficient = getattr(prior, COMPONENT_FIELDS[component], None)
            if not isinstance(coefficient, NormalPrior) or coefficient.mean != 0:
                raise ValueError(f"SharedShrinkage requires a zero-mean normal {component} coefficient prior.")

    def conditional_prior(self, prior, medians):
        """Return a new channel prior conditional on the common medians."""
        if set(medians) != set(self.medians):
            raise ValueError("Conditional median names must match the hierarchy.")
        if any(not np.isfinite(m) or m <= 0 for m in medians.values()):
            raise ValueError("Conditional medians must be positive and finite.")
        return replace(prior, **{COMPONENT_FIELDS[c]: NormalPrior(0., m / NORMAL_ABSOLUTE_MEDIAN)
                                for c, m in medians.items()})

    def sample_medians(self, size, *, rng):
        """Draw shared hyperparameters, once per joint prior draw."""
        return {c: rng.lognormal(np.log(m), self.log_sd, size=size)
                for c, m in self.medians.items()}


__all__ = ["SharedShrinkage"]
