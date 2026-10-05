"""Level and local linear trend components."""
from dataclasses import dataclass
import numpy as np
from .base import ComponentBlock, Innovation, mode


@dataclass(frozen=True)
class LocalLevel:
    mode: str = 'dynamic'
    name: str = 'level'
    initial_prior: object = None
    innovation_prior: object = None

    def __post_init__(self):
        mode(self.mode)

    def build(self, steps, priors, exog=None):
        g = (Innovation(self.name, np.ones((1, 1)), self.innovation_prior or priors.level),) if self.mode == 'dynamic' else ()
        key = 'initial_level' if self.name == 'level' else self.name + '.initial'
        return ComponentBlock(self.name, np.eye(1), np.ones((steps, 1)), np.eye(1),
            (key,), (self.initial_prior or priors.initial_level,), (self.name,), g,
            {self.name: np.ones(1)})


@dataclass(frozen=True)
class LocalLinearTrend:
    level_mode: str = 'dynamic'
    trend_mode: str = 'dynamic'
    name: str = 'trend'
    level_prior: object = None
    slope_prior: object = None
    initial_level_prior: object = None
    initial_slope_prior: object = None

    def __post_init__(self):
        mode(self.level_mode)
        mode(self.trend_mode, off=True)

    def build(self, steps, priors, exog=None):
        prefix = '' if self.name == 'trend' else self.name + '.'
        level, slope = prefix+'level', prefix+'slope'
        if self.trend_mode == 'off':
            return LocalLevel(self.level_mode, level, self.initial_level_prior, self.level_prior).build(steps, priors, exog)
        F = np.array([[1., 1.], [0., 1.]])
        H = np.tile([1., 0.], (steps, 1))
        groups = []
        if self.level_mode == 'dynamic':
            groups.append(Innovation(level, np.array([[1.], [0.]]), self.level_prior or priors.level))
        if self.trend_mode == 'dynamic':
            groups.append(Innovation(slope, np.array([[0.], [1.]]), self.slope_prior or priors.slope))
        return ComponentBlock(self.name, F, H, np.eye(2),
            (prefix+'initial_level', prefix+'initial_slope'),
            (self.initial_level_prior or priors.initial_level, self.initial_slope_prior or priors.initial_slope),
            (level, slope), tuple(groups), {level: [1., 0.], slope: [0., 1.]})
