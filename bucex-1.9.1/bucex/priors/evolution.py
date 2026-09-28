"""Priors for ancillary structural predictors, in their link-scale units."""
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass(frozen=True)
class EvolutionPriors:
    """Continuous FS shrinkage for a log-scale evolution.

    Innovation medians are SDs per observation step, on log-scale units.
    Initial slope is regularized separately; seasonal coefficients have proper
    Gaussian priors. The initial level is anchored to zero because the overall
    log scale is already identified by the observation-scale parameter.
    These defaults are explicit starting assumptions, not tuned to the data.
    """
    innovation: str = "lasso"
    innovation_median: dict = field(default_factory=lambda: {
        "level": .01, "trend": .00001, "season": .01})
    initial_slope_sd: float = .001
    seasonal_initial_sd: float = .3
    spike_shape: float = .5
    tail_shape: float = .5

    def __post_init__(self):
        if self.innovation not in {"normal", "lasso", "triple_gamma"}:
            raise ValueError("innovation must be normal, lasso, or triple_gamma.")
        medians = dict(self.innovation_median)
        if set(medians) != {"level", "trend", "season"}:
            raise ValueError("innovation_median requires level, trend, season.")
        values = [*medians.values(), self.initial_slope_sd, self.seasonal_initial_sd,
                  self.spike_shape, self.tail_shape]
        if any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Evolution prior scales and shapes must be finite and positive.")
        object.__setattr__(self, "innovation_median", medians)

    def resolve(self, period):
        from .fs import fs_priors
        from .structural import NormalPrior
        return fs_priors("gaussian", period=period, innovation=self.innovation,
            innovation_median=self.innovation_median,
            initial_level=NormalPrior(0., 1.),  # excluded from the scale regression
            initial_slope=NormalPrior(0., self.initial_slope_sd),
            seasonal_initial_sd=self.seasonal_initial_sd,
            spike_shape=self.spike_shape, tail_shape=self.tail_shape)

    def to_dict(self):
        return asdict(self)


__all__ = ["EvolutionPriors"]
