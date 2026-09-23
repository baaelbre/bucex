"""Independent structural prior blocks, optionally coupled by a likelihood."""
from dataclasses import dataclass
from typing import Mapping

from .structural import FSGaussianPriors, FSGEVPriors
from .shrinkage import SharedShrinkage


@dataclass(frozen=True)
class MarginalPriors:
    """Named continuous FS priors for private trajectories.

    ``shrinkage=SharedShrinkage(...)`` optionally learns common normal-prior
    scales while retaining private trajectories and individual innovation SDs.
    Without it, the channel priors are independent. A GaussianCopula couples
    observations through the joint likelihood.
    """

    channels: Mapping[str, FSGaussianPriors | FSGEVPriors]
    shrinkage: SharedShrinkage | None = None

    def __post_init__(self):
        if not self.channels:
            raise ValueError("MarginalPriors requires at least one named channel.")
        for name, prior in self.channels.items():
            if not isinstance(prior, (FSGaussianPriors, FSGEVPriors)):
                raise TypeError(f"{name}: supply Gaussian or GEV FS priors.")
            if prior.ssvs is not None:
                raise ValueError(f"{name}: SSVS is outside the BUCEX 1.8.4 paper API; use a continuous prior.")
        if self.shrinkage is not None:
            if not isinstance(self.shrinkage, SharedShrinkage):
                raise TypeError("shrinkage must be SharedShrinkage(...) or None.")
            for prior in self.channels.values():
                self.shrinkage.validate_prior(prior)
        object.__setattr__(self, "channels", dict(self.channels))
