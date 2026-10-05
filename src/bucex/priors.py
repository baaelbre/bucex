"""Explicit prior distributions; all scale arguments are standard deviations.

Innovation priors act on signed amplitudes s, with innovation variance s**2.
Normal priors do not put point mass on exactly zero.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class Normal:
    mean: float = 0.0
    sd: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.mean) or not math.isfinite(self.sd) or self.sd <= 0:
            raise ValueError("Normal requires a finite mean and a positive finite sd.")


@dataclass(frozen=True)
class HalfNormal:
    sd: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.sd) or self.sd <= 0:
            raise ValueError("HalfNormal requires a positive finite sd (underlying normal scale).")


@dataclass(frozen=True)
class InverseGamma:
    """Density proportional to v**(-shape-1) exp(-scale/v), v > 0."""
    shape: float = 2.0
    scale: float = 2.0

    def __post_init__(self):
        if any(not math.isfinite(x) or x <= 0 for x in (self.shape, self.scale)):
            raise ValueError("InverseGamma requires positive finite shape and scale.")


@dataclass(frozen=True)
class Fixed:
    """Hold a parameter at a known value; this is not an estimated static state."""
    value: float

    def __post_init__(self):
        if not math.isfinite(self.value):
            raise ValueError("Fixed requires a finite value.")


@dataclass(frozen=True)
class Priors:
    """Priors for one response, in observation units and per observation step.

    The defaults are the seasonal temperature paper calibration, not universal
    defaults for arbitrary units. Initial seasonal SD is on orthonormal
    zero-sum contrasts. Variance is the squared baseline observation scale.
    Shape is the EVT xi convention, with no artificial truncation.
    """
    initial_level: Normal | Fixed = field(default_factory=lambda: Normal(0, 10))
    initial_slope: Normal | Fixed = field(default_factory=lambda: Normal(0, .01))
    initial_seasonal: Normal | Fixed = field(default_factory=lambda: Normal(0, 10))
    level: Normal | Fixed = field(default_factory=lambda: Normal(0, .1))
    slope: Normal | Fixed = field(default_factory=lambda: Normal(0, .002))
    seasonal: Normal | Fixed = field(default_factory=lambda: Normal(0, .1))
    variance: InverseGamma | Fixed = field(default_factory=InverseGamma)
    shape: Normal | Fixed = field(default_factory=lambda: Normal(0, .3))

    def __post_init__(self):
        for name in ("initial_level", "initial_slope", "initial_seasonal", "level", "slope", "seasonal", "shape"):
            if not isinstance(getattr(self, name), (Normal, Fixed)):
                raise TypeError(f"{name} must be Normal or Fixed.")
        for name in ("level", "slope", "seasonal"):
            p = getattr(self, name)
            if isinstance(p, Normal) and p.mean != 0:
                raise ValueError("Innovation Normal priors must be zero centred.")
            if isinstance(p, Fixed) and p.value < 0:
                raise ValueError("Fixed innovation amplitudes must be nonnegative.")
        if not isinstance(self.variance, (InverseGamma, Fixed)):
            raise TypeError("variance must be InverseGamma or Fixed.")
        if isinstance(self.variance, Fixed) and self.variance.value <= 0:
            raise ValueError("Fixed observation variance must be positive.")


@dataclass(frozen=True)
class Pooling:
    """Shared normal-prior SDs for innovations only; None leaves a prior private.

    s[c,j] | tau[c] ~ Normal(0, tau[c]**2), tau[c] ~ HalfNormal(A[c]).
    Initial states, initial slopes, observation scales and shapes are private.
    This does not model residual dependence between observations.
    """
    level: HalfNormal | None = None
    slope: HalfNormal | None = None
    seasonal: HalfNormal | None = None

    groups: dict = field(default_factory=dict)

    @property
    def scales(self):
        return {**{c: getattr(self, c) for c in ("level", "slope", "seasonal") if getattr(self, c) is not None}, **self.groups}

    def __post_init__(self):
        if set(self.groups) & {"level", "slope", "seasonal"}:
            raise ValueError("Use the named level, slope and seasonal fields for these groups.")
        if any(not isinstance(k, str) or not k or not isinstance(v, HalfNormal) for k, v in self.groups.items()):
            raise TypeError("groups maps innovation names to HalfNormal priors.")
        if not self.scales:
            raise ValueError("Specify at least one pooled innovation scale, or use pooling=None.")
        for name in ("level", "slope", "seasonal"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), HalfNormal):
                raise TypeError(f"Pooling.{name} must be HalfNormal or None.")
