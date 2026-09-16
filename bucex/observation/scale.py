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


@dataclass(frozen=True)
class LogScale:
    """Observation scale with optional calendar effects and secular change.

    log(sigma_t) = log(sigma) + seasonal[phase_t] + offset_t.
    ``mode='linear'`` uses offset_t = slope * t / time_unit, with
    slope ~ N(0, slope_sd**2). ``mode='rw'`` uses offset_t = omega*z_t,
    z_0=0, z_t=z_(t-1)+N(0,1), omega ~ N(0, innovation_sd**2).
    Observation t=1 is one step after the anchored initial state.
    ``mode='constant'`` omits the secular term, retaining optional seasons.

    Units are observation steps: time_unit=120 is a decade for monthly data.
    Every channel has its own coefficients/path. These are additional scale
    priors, independent of the location innovation prior in MarginalPriors.
    """

    mode: str = "constant"
    seasonal: SeasonalScale | None = None
    slope_sd: float = 0.1
    innovation_sd: float = 0.01
    time_unit: float = 120.0

    def __post_init__(self):
        if self.mode not in {"constant", "linear", "rw"}:
            raise ValueError("LogScale.mode must be constant, linear, or rw.")
        if self.seasonal is not None and not isinstance(self.seasonal, SeasonalScale):
            raise TypeError("seasonal must be SeasonalScale(...) or None.")
        if any(not np.isfinite(v) or v <= 0 for v in (self.slope_sd, self.innovation_sd, self.time_unit)):
            raise ValueError("Scale prior SDs and time_unit must be finite and positive.")

    @property
    def period(self):
        return self.seasonal.period if self.seasonal else 1

    @property
    def prior_sd(self):
        return self.seasonal.prior_sd if self.seasonal else 1.

    def contrast(self):
        return self.seasonal.contrast() if self.seasonal else np.empty((1, 0))

    def phases(self, n_time, dates=None, *, start_index=0):
        return self.seasonal.phases(n_time, dates, start_index=start_index) if self.seasonal else np.zeros(n_time, int)

    def to_dict(self):
        return {"kind": "log_scale", "mode": self.mode,
                "seasonal": self.seasonal.to_dict() if self.seasonal else None,
                "slope_sd": self.slope_sd, "innovation_sd": self.innovation_sd,
                "time_unit": self.time_unit}


@dataclass(frozen=True)
class StructuralScale:
    """A complete additive structural model for log observation scale.

    log(sigma_t) = log(sigma) + level_t + seasonal_t. The evolving level
    starts at zero; its initial slope and seasonal state are estimated.
    Components use the same time-step convention as location. Separate
    ``EvolutionPriors`` prevent accidental reuse of temperature-scale priors
    for a dimensionless log scale. Inference uses exact-likelihood elliptical
    slices in FS coordinates, including any residual copula contribution.
    """
    components: tuple
    priors: object = None

    def __post_init__(self):
        from ..components import LocalLevel, LocalLinearTrend, DummySeasonal
        from ..priors.evolution import EvolutionPriors
        components = tuple(self.components)
        if any(not isinstance(c, (LocalLevel, LocalLinearTrend, DummySeasonal)) for c in components):
            raise TypeError("StructuralScale supports level, local-linear trend, and dummy seasonality.")
        if sum(isinstance(c, (LocalLevel, LocalLinearTrend)) for c in components) != 1:
            raise ValueError("StructuralScale requires exactly one level or trend component.")
        if sum(isinstance(c, DummySeasonal) for c in components) > 1:
            raise ValueError("StructuralScale supports one seasonal component.")
        # Initial distributions belong to EvolutionPriors; never silently ignore
        # a second initial-state declaration on an ancillary component.
        for c in components:
            fields = ("initial_mean", "initial_sd", "initial_level", "initial_level_sd",
                      "initial_slope_sd")
            if any(getattr(c, k, None) is not None for k in fields) or getattr(c, "initial_slope", 0.) != 0.:
                raise ValueError("Declare scale initial priors in EvolutionPriors, not in components.")
        components = tuple(LocalLinearTrend(level_mode=c.mode, trend_mode="off",
                           level_name=c.name) if isinstance(c, LocalLevel) else c for c in components)
        prior = EvolutionPriors() if self.priors is None else self.priors
        if not isinstance(prior, EvolutionPriors):
            raise TypeError("StructuralScale.priors must be EvolutionPriors(...).")
        names = [n for c in components for n in c.spec.state_names]
        if len(names) != len(set(names)):
            raise ValueError("Scale state names must be unique.")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "priors", prior)

    @property
    def mode(self):
        return "structural"

    @property
    def period(self):
        # Static calendar coefficients are absent; seasonality lives in the
        # structural state, not a duplicate SeasonalScale term.
        return 1

    def contrast(self):
        return np.empty((1, 0))

    def phases(self, n_time, dates=None, *, start_index=0):
        return np.zeros(n_time, int)

    def to_dict(self):
        return {"kind": "structural_scale", "components": [c.to_dict() for c in self.components],
                "priors": self.priors.to_dict()}


def scale_from_dict(value):
    if not value:
        return None
    if value.get("kind") == "structural_scale":
        from ..components import component_from_dict
        from ..priors.evolution import EvolutionPriors
        return StructuralScale(tuple(component_from_dict(c) for c in value["components"]),
                               EvolutionPriors(**value["priors"]))
    if value.get("kind") == "log_scale":
        fields = {k:v for k,v in value.items() if k != "kind"}
        fields["seasonal"] = SeasonalScale(**fields["seasonal"]) if fields.get("seasonal") else None
        return LogScale(**fields)
    return SeasonalScale(**value)
