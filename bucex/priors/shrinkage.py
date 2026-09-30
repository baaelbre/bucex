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

    For the paper reference, use ``half_normal(anchors)``: signed coefficients
    have N(0,tau_c**2) priors and tau_c has a half-normal hyperprior with scale
    anchors[c]. One tau_c is learned across responses, separately for each
    component. ``half_t`` and ``half_cauchy`` provide sensitivity alternatives.
    Initial slopes keep their separately declared normal priors.

    Half-normal scales are sampled by direct GIG Gibbs updates of tau_c**2.
    This update is selected automatically; other hyperprior families use slice
    sampling. The prior specification and saved parameter names are unchanged.

    The legacy constructor below retains its normal--lognormal meaning.
    ``medians`` are anchors for that *hyperprior*: log(m_c) is normal with
    mean log(anchor_c) and standard deviation ``log_sd``. Conditional on m_c,
    each active channel's signed FS coefficient has distribution
    N(0, (m_c / Phi^{-1}(.75))**2). Its absolute value is the physical process
    SD. Marginally the coefficients have a normal scale-mixture prior.

    The channels retain different process SDs and independent standardized
    state innovations. This hierarchy shares regularization, not a latent
    trajectory or residual correlation. Use a copula for the latter.

    ``from_sd(anchors, log_sd=...)`` instead declares conditional Normal SDs
    directly and leaves initial rates separate. The legacy median convention
    remains the default here so archived specifications keep their meaning.

    Example::

        SharedShrinkage(medians={"level": .0025, "slope": .0000125, "seasonal": .02})

    Initial slopes can share a learned zero-centred normal-prior SD, whose
    lognormal anchor is ``initial_slope_sd`` (default .0025 per update).
    Alternatively, set ``initial_slope_sd=None`` and supply
    ``initial_slope_median`` to use the median absolute initial slope as the
    learned scale, with the same Phi^{-1}(.75) conversion as innovations.
    Both conventions describe a separate hyperparameter, not a shared mean
    slope. Set both to None to retain fixed channel priors.
    Components omitted from ``medians`` keep their declared channel priors.
    ``log_sd=log(2)`` places about 95% of each hyperprior between one quarter
    and four times its anchor. All anchors use the model's time/response units.
    """

    medians: Mapping[str, float]
    log_sd: float = float(np.log(2.))
    initial_slope_sd: float | None = .0025
    initial_slope_median: float | None = None
    scale_parameterization: str = "absolute_median"
    hyperprior: str = "lognormal"
    df: float = 4.0

    def __post_init__(self):
        if self.hyperprior not in {"lognormal", "half_normal", "half_t", "half_cauchy"}:
            raise ValueError("Unknown shared-scale hyperprior family.")
        if not np.isfinite(self.df) or self.df <= 0:
            raise ValueError("Half-t degrees of freedom must be positive and finite.")
        if self.hyperprior != "lognormal" and self.scale_parameterization != "normal_sd":
            raise ValueError("Half-scale hyperpriors use direct normal_sd units.")
        if self.scale_parameterization not in {"absolute_median", "normal_sd"}:
            raise ValueError("scale_parameterization must be absolute_median or normal_sd.")
        aliases = {"trend": "slope", "season": "seasonal"}
        medians = {}
        for supplied, value in self.medians.items():
            key = aliases.get(supplied, supplied)
            if key not in COMPONENT_FIELDS or key in medians:
                raise ValueError("SharedShrinkage components must be unique level, slope or seasonal names.")
            if not np.isfinite(value) or value <= 0:
                raise ValueError("SharedShrinkage anchors must be positive and finite.")
            medians[key] = float(value)
        if self.initial_slope_sd is not None and self.initial_slope_median is not None:
            raise ValueError("Choose initial_slope_sd or initial_slope_median, not both.")
        if (not medians and self.initial_slope_sd is None and self.initial_slope_median is None) or not np.isfinite(self.log_sd) or self.log_sd <= 0:
            raise ValueError("Declare at least one anchor and a positive finite log_sd.")
        if self.initial_slope_sd is not None:
            if not np.isfinite(self.initial_slope_sd) or self.initial_slope_sd <= 0:
                raise ValueError("initial_slope_sd must be positive and finite, or None.")
            object.__setattr__(self, "initial_slope_sd", float(self.initial_slope_sd))
        if self.initial_slope_median is not None:
            if not np.isfinite(self.initial_slope_median) or self.initial_slope_median <= 0:
                raise ValueError("initial_slope_median must be positive and finite, or None.")
            object.__setattr__(self, "initial_slope_median", float(self.initial_slope_median))
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

        Innovation entries are physical-SD medians; initial_slope is either
        a normal prior SD or a median absolute coefficient, as declared.
        """
        if set(medians) != set(self.anchors):
            raise ValueError("Conditional median names must match the hierarchy.")
        if any(not np.isfinite(m) or m <= 0 for m in medians.values()):
            raise ValueError("Conditional medians must be positive and finite.")
        return replace(prior, **{COEFFICIENT_FIELDS[c]: NormalPrior(0., self.coefficient_sd(c, m))
                                for c, m in medians.items()})

    @property
    def anchors(self):
        """All shared scales in their declared units."""
        initial = (self.initial_slope_median if self.initial_slope_median is not None
                   else self.initial_slope_sd)
        return {**self.medians, **({"initial_slope": initial} if initial is not None else {})}

    def coefficient_sd(self, component, scale):
        return (scale if self.uses_normal_sd(component)
                else scale / NORMAL_ABSOLUTE_MEDIAN)

    def uses_normal_sd(self, component):
        return ((component == "initial_slope" and self.initial_slope_median is None)
                or (component != "initial_slope" and self.scale_parameterization == "normal_sd"))

    @classmethod
    def from_sd(cls, anchors, *, log_sd=float(np.log(3.)), hyperprior="lognormal", df=4.):
        """Learn Normal coefficient SDs directly; initial rates stay separate.

        s_cj | tau_c ~ Normal(0, tau_c**2). With the legacy default lognormal
        family, log(tau_c / anchors[c]) ~ Normal(0, log_sd**2). Half-family
        anchors are their distributional scale parameters instead.
        """
        return cls(anchors, log_sd=log_sd, initial_slope_sd=None,
                   scale_parameterization="normal_sd", hyperprior=hyperprior, df=df)

    @classmethod
    def half_normal(cls, anchors):
        """s_cj | tau_c ~ N(0,tau_c^2); tau_c ~ HalfNormal(anchors[c]).

        Each anchor is the SD of the underlying zero-centred normal and the
        marginal RMS signed coefficient. Initial slopes remain separate.
        """
        return cls.from_sd(anchors, hyperprior="half_normal")

    @classmethod
    def half_t(cls, scales, *, df=4.):
        """Half-Student-t hyperpriors; scales are t scales, not t SDs."""
        return cls.from_sd(scales, hyperprior="half_t", df=df)

    @classmethod
    def half_cauchy(cls, scales):
        """Half-Cauchy hyperpriors; no finite marginal coefficient variance."""
        return cls.from_sd(scales, hyperprior="half_cauchy", df=1.)

    @property
    def rms_multiplier(self):
        if self.hyperprior == "lognormal":
            return float(np.exp(self.log_sd**2))
        if self.hyperprior == "half_normal":
            return 1.
        nu = 1. if self.hyperprior == "half_cauchy" else self.df
        return float(np.sqrt(nu/(nu-2))) if nu > 2 else np.inf

    def scale_quantile(self, component, probability):
        """Quantiles of the shared SD, not of a response's innovation SD."""
        from scipy.stats import t
        probability = np.asarray(probability)
        if np.any((probability < 0) | (probability > 1)):
            raise ValueError("Probabilities must lie in [0,1].")
        a = self.anchors[component]
        if self.hyperprior == "lognormal":
            return a*np.exp(self.log_sd*ndtri(probability))
        if self.hyperprior == "half_normal":
            return a*ndtri((1+probability)/2)
        return a*t.ppf((1+probability)/2, 1. if self.hyperprior == "half_cauchy" else self.df)

    def scale_mean(self, component):
        """Hyperprior mean, including an infinite mean for half-Cauchy scales."""
        from scipy.stats import t
        a=self.anchors[component]
        if self.hyperprior=='lognormal':return float(a*np.exp(self.log_sd**2/2))
        if self.hyperprior=='half_normal':return float(a*np.sqrt(2/np.pi))
        nu=1. if self.hyperprior=='half_cauchy' else self.df
        return float(a*2*nu*t.pdf(0,nu)/(nu-1)) if nu>1 else np.inf

    def scale_cdf(self, component, value):
        from scipy.special import ndtr
        from scipy.stats import t
        x = np.asarray(value)/self.anchors[component]
        if self.hyperprior == "lognormal":
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.where(x > 0, ndtr(np.log(x)/self.log_sd), 0.)
        if self.hyperprior == "half_normal":
            return np.where(x > 0, 2*ndtr(x)-1, 0.)
        return np.where(x > 0, 2*t.cdf(x, 1. if self.hyperprior == "half_cauchy" else self.df)-1, 0.)

    def sample_medians(self, size, *, rng):
        """Draw shared hyperparameters, once per joint prior draw."""
        if self.hyperprior == "lognormal":
            return {c: rng.lognormal(np.log(m), self.log_sd, size=size)
                    for c, m in self.anchors.items()}
        if self.hyperprior == "half_normal":
            return {c: a*np.abs(rng.normal(size=size)) for c,a in self.anchors.items()}
        nu = 1. if self.hyperprior == "half_cauchy" else self.df
        return {c: a*np.abs(rng.standard_t(nu, size=size)) for c,a in self.anchors.items()}

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
                anchor_kind=("normal_SD" if self.uses_normal_sd(c)
                             else "absolute_coefficient_median"),
                horizon_updates=int(horizon), period=period, response_unit=unit,
                hyperprior=self.hyperprior, df=self.df if self.hyperprior=='half_t' else None,
                log_sd=self.log_sd if self.hyperprior=='lognormal' else None,
                hyperprior_lower=float(self.scale_quantile(c,.025)),
                hyperprior_upper=float(self.scale_quantile(c,.975)),
                coefficient_sd_at_anchor=sd, response_gain=gains[c],
                displacement_sd_at_anchor=sd*gains[c],
                displacement_sd_marginal=sd*gains[c]*self.rms_multiplier,
                finite_second_moment=bool(np.isfinite(self.rms_multiplier)),
                slope_time_unit=slope_time_unit,
                initial_rate_sd_at_anchor=sd*slope_time_unit if c == "initial_slope" else None,
                initial_rate_sd_marginal=sd*slope_time_unit*self.rms_multiplier if c == "initial_slope" else None))
        return rows


__all__ = ["SharedShrinkage"]


@dataclass(frozen=True)
class IndependentShrinkage(SharedShrinkage):
    """Normal scale-mixture innovation priors for exactly one response.

    The numerical anchors may be reused in other analyses, but each fit draws
    and updates its own hyperparameters. No information crosses responses.
    Initial slopes retain their separately declared Normal priors.
    """
    initial_slope_sd: float | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.initial_slope_sd is not None or self.initial_slope_median is not None:
            raise ValueError("IndependentShrinkage leaves the initial slope prior separate.")
