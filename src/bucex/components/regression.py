"""Static or random-walk regression coefficients (time-varying parameters)."""
from dataclasses import dataclass, field
import numpy as np
from .base import ComponentBlock, Innovation, mode, features
from ..priors import Normal


@dataclass(frozen=True)
class Regression:
    features: tuple[str, ...]
    mode: str = 'static'
    name: str = 'regression'
    prior: object = field(default_factory=lambda: Normal(0, 1))
    innovation_prior: object = field(default_factory=lambda: Normal(0, .1))

    def __post_init__(self):
        object.__setattr__(self, 'features', (self.features,) if isinstance(self.features, str) else tuple(self.features))
        mode(self.mode)
        if not self.features or len(set(self.features)) != len(self.features) or any(not isinstance(x, str) or not x for x in self.features):
            raise ValueError('Supply distinct, nonempty feature names.')

    def build(self, steps, priors, exog=None):
        H = features(exog, self.features, steps)
        d = len(self.features)
        names = tuple(f'{self.name}.{f}' for f in self.features)
        groups = tuple(Innovation(name, np.eye(d)[:, i:i+1], self.innovation_prior) for i, name in enumerate(names)) if self.mode == 'dynamic' else ()
        return ComponentBlock(self.name, np.eye(d), H, np.eye(d), names,
            (self.prior,)*d, names, groups, {name: np.eye(d)[i] for i, name in enumerate(names)})
