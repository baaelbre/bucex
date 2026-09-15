"""Independent structural prior blocks, optionally coupled by a likelihood."""
from dataclasses import dataclass
from typing import Mapping

from .structural import FSGaussianPriors, FSGEVPriors


@dataclass(frozen=True)
class MarginalPriors:
    """Named FS + SSVS priors for private trajectories.

    No hyperparameters are pooled. A model's optional GaussianCopula creates
    full posterior feedback through the likelihood, while these priors stay
    independent. Shared latent components use JointPriors instead.
    """

    channels: Mapping[str, FSGaussianPriors | FSGEVPriors]

    def __post_init__(self):
        if not self.channels:
            raise ValueError("MarginalPriors requires at least one named channel.")
        for name, prior in self.channels.items():
            if not isinstance(prior, (FSGaussianPriors, FSGEVPriors)) or prior.ssvs is None:
                raise TypeError(f"{name}: supply an FS prior with exact SSVS enabled.")
        object.__setattr__(self, "channels", dict(self.channels))
