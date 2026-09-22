"""Independent structural prior blocks, optionally coupled by a likelihood."""
from dataclasses import dataclass
from typing import Mapping

from .structural import FSGaussianPriors, FSGEVPriors
from .shrinkage import SharedShrinkage


@dataclass(frozen=True)
class MarginalPriors:
    """Named FS priors for private trajectories, with optional exact SSVS.

    ``shrinkage=SharedShrinkage(...)`` optionally learns common normal-prior
    scales while retaining private trajectories and individual innovation SDs.
    Without it, the channel priors are independent. A GaussianCopula couples
    observations through the joint likelihood. Shared latent components use
    JointPriors instead.
    """

    channels: Mapping[str, FSGaussianPriors | FSGEVPriors]
    shrinkage: SharedShrinkage | None = None

    def __post_init__(self):
        if not self.channels:
            raise ValueError("MarginalPriors requires at least one named channel.")
        for name, prior in self.channels.items():
            if not isinstance(prior, (FSGaussianPriors, FSGEVPriors)):
                raise TypeError(f"{name}: supply Gaussian or GEV FS priors.")
        if self.shrinkage is not None:
            if not isinstance(self.shrinkage, SharedShrinkage):
                raise TypeError("shrinkage must be SharedShrinkage(...) or None.")
            for prior in self.channels.values():
                self.shrinkage.validate_prior(prior)
        object.__setattr__(self, "channels", dict(self.channels))
