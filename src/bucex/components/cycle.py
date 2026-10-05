"""A damped stochastic cycle with a fixed period and damping coefficient."""
from dataclasses import dataclass, field
import numpy as np
from .base import ComponentBlock, Innovation, mode
from ..priors import Normal


@dataclass(frozen=True)
class Cycle:
    period: float
    damping: float = 1.
    mode: str = 'dynamic'
    name: str = 'cycle'
    initial_prior: object = field(default_factory=lambda: Normal(0, 1))
    innovation_prior: object = field(default_factory=lambda: Normal(0, .1))

    def __post_init__(self):
        mode(self.mode)
        if not np.isfinite(self.period) or self.period <= 2 or not 0 < self.damping <= 1:
            raise ValueError('Cycle requires period > 2 steps and 0 < damping <= 1.')

    def build(self, steps, priors, exog=None):
        w = 2*np.pi/self.period
        F = self.damping*np.array([[np.cos(w), np.sin(w)], [-np.sin(w), np.cos(w)]])
        H = np.tile([1., 0.], (steps, 1))
        names = (self.name+'.cos', self.name+'.sin')
        g = (Innovation(self.name, np.eye(2), self.innovation_prior),) if self.mode == 'dynamic' else ()
        return ComponentBlock(self.name, F, H, np.eye(2), names, (self.initial_prior,)*2,
            names, g, {self.name: [1., 0.]})
