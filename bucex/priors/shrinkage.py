"""Shared normal-prior scales for private FS innovations and initial slopes."""
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
from scipy.special import ndtri

from .structural import NormalPrior


NORMAL_ABSOLUTE_MEDIAN = float(ndtri(.75))
COMPONENT_FIELDS = {"level": "s_level", "slope": "s_trend", "seasonal": "s_season"}
COEFFICIENT_FIELDS = {**COMPONENT_FIELDS, "initial_slope": "beta0"}


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

        SharedShrinkage(medians={"level": .0025, "slope": .0000125, "seasonal": .02})

    Initial slopes also share a learned zero-centred normal-prior SD, whose
    lognormal anchor is ``initial_slope_sd`` (default .0025 per update). This
    is a separate hyperparameter, not the slope innovation scale or a shared
    mean slope. Set ``initial_slope_sd=None`` to retain fixed channel priors.
    Components omitted from ``medians`` keep their declared channel priors.
    ``log_sd=log(2)`` places about 95% of each hyperprior between one quarter
    and four times its anchor. All anchors use the model's time/response units.
    """

    medians: Mapping[str, float]
    log_sd: float = float(np.log(2.))
    initial_slope_sd: float | None = .0025

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
        if (not medians and self.initial_slope_sd is None) or not np.isfinite(self.log_sd) or self.log_sd <= 0:
            raise ValueError("Declare at least one anchor and a positive finite log_sd.")
        if self.initial_slope_sd is not None:
            if not np.isfinite(self.initial_slope_sd) or self.initial_slope_sd <= 0:
                raise ValueError("initial_slope_sd must be positive and finite, or None.")
            object.__setattr__(self, "initial_slope_sd", float(self.initial_slope_sd))
        # Stable scientific and sampling order, including after sorted JSON/archive decoding.
        object.__setattr__(self, "medians", {c: medians[c] for c in COMPONENT_FIELDS if c in medians})
        object.__setattr__(self, "log_sd", float(self.log_sd))

    def validate_prior(self, prior):
        if any(getattr(prior, key, None) is not None
               for key in ("ssvs", "lasso", "pc", "horseshoe", "triple_gamma")):
            raise ValueError("SharedShrinkage requires continuous normal FS priors; omit SSVS and local-mixture priors.")
        for component in self.anchors:
            coefficient = getattr(prior, COEFFICIENT_FIELDS[component], None)
            if not isinstance(coefficient, NormalPrior) or coefficient.mean != 0:
                raise ValueError(f"SharedShrinkage requires a zero-mean normal {component} coefficient prior.")

    def conditional_prior(self, prior, medians):
        """Return a channel prior conditional on the named shared scales.

        Innovation entries are physical-SD medians; initial_slope is a normal
        prior SD. Both use the same lognormal scale-hierarchy machinery.
        """
        if set(medians) != set(self.anchors):
            raise ValueError("Conditional median names must match the hierarchy.")
        if any(not np.isfinite(m) or m <= 0 for m in medians.values()):
            raise ValueError("Conditional medians must be positive and finite.")
        return replace(prior, **{COEFFICIENT_FIELDS[c]: NormalPrior(0., self.coefficient_sd(c, m))
                                for c, m in medians.items()})

    @property
    def anchors(self):
        """All shared scales; initial_slope uses SD rather than absolute median."""
        return {**self.medians, **({"initial_slope": self.initial_slope_sd}
                                  if self.initial_slope_sd is not None else {})}

    @staticmethod
    def coefficient_sd(component, scale):
        return scale if component == "initial_slope" else scale / NORMAL_ABSOLUTE_MEDIAN

    def sample_medians(self, size, *, rng):
        """Draw shared hyperparameters, once per joint prior draw."""
        return {c: rng.lognormal(np.log(m), self.log_sd, size=size)
                for c, m in self.anchors.items()}

    @classmethod
    def from_effects(cls, *, horizon, level_displacement_sd=None,
                    slope_displacement_sd=None, seasonal_displacement_sd=None,
                    initial_slope_sd=.30, slope_time_unit=120, period=12,
                    log_sd=float(np.log(2.)), calibration="anchor"):
        """Declare physical changes instead of per-update coefficient scales.

        Displacements use response units after ``horizon`` updates. Initial
        slope uses response units per ``slope_time_unit`` updates (120 monthly
        updates = a decade). ``anchor`` calibrates conditional prior SDs at
        the hyperprior median. ``marginal`` calibrates RMS SDs after integrating
        the lognormal hyperprior; these are larger by exp(log_sd**2).
        None omits that component; initial_slope_sd=None disables its pooling.
        """
        from .calibration import innovation_response_gains
        if calibration not in {"anchor", "marginal"}:
            raise ValueError("calibration must be 'anchor' or 'marginal'.")
        if not np.isfinite(slope_time_unit) or slope_time_unit <= 0:
            raise ValueError("slope_time_unit must be positive and finite.")
        gains = innovation_response_gains(horizon, period=period)
        inflation = np.exp(float(log_sd)**2) if calibration == "marginal" else 1.
        medians = {}
        for c, effect in {"level": level_displacement_sd, "slope": slope_displacement_sd,
                          "seasonal": seasonal_displacement_sd}.items():
            if effect is None:
                continue
            if c not in gains or gains[c] <= 0:
                raise ValueError(f"Cannot calibrate {c} at this horizon/period.")
            medians[c] = float(effect)*NORMAL_ABSOLUTE_MEDIAN/(gains[c]*inflation)
        initial = None if initial_slope_sd is None else float(initial_slope_sd)/(slope_time_unit*inflation)
        return cls(medians, log_sd=log_sd, initial_slope_sd=initial)

    def calibration(self, *, horizon=360, period=12, slope_time_unit=120, unit="response units"):
        """Auditable hyperprior effects, including unconditional RMS inflation.

        Returns records suitable for pandas.DataFrame; no observed data are
        used. Innovation effects are future contributions conditional on the
        current state. Initial-slope displacement starts at the initial time.
        """
        from .calibration import innovation_response_gains
        gains = {**innovation_response_gains(horizon, period=period), "initial_slope": float(horizon)}
        if not np.isfinite(slope_time_unit) or slope_time_unit <= 0:
            raise ValueError("slope_time_unit must be positive and finite.")
        rows = []
        for c, anchor in self.anchors.items():
            if c not in gains:
                raise ValueError(f"No response gain for {c}; declare its period.")
            sd = self.coefficient_sd(c, anchor)
            rows.append(dict(component=c, anchor=anchor,
                anchor_kind="normal_SD" if c == "initial_slope" else "absolute_coefficient_median",
                horizon_updates=int(horizon), period=period, response_unit=unit,
                log_sd=self.log_sd, hyperprior_lower=anchor*np.exp(ndtri(.025)*self.log_sd),
                hyperprior_upper=anchor*np.exp(ndtri(.975)*self.log_sd),
                coefficient_sd_at_anchor=sd, response_gain=gains[c],
                displacement_sd_at_anchor=sd*gains[c],
                displacement_sd_marginal=sd*gains[c]*np.exp(self.log_sd**2),
                slope_time_unit=slope_time_unit,
                initial_rate_sd_at_anchor=sd*slope_time_unit if c == "initial_slope" else None,
                initial_rate_sd_marginal=sd*slope_time_unit*np.exp(self.log_sd**2) if c == "initial_slope" else None))
        return rows


__all__ = ["SharedShrinkage"]
