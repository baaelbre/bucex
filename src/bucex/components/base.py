"""Public linear-state component contract.

A component supplies its matrices, coefficient priors and named outputs.
The compiler and samplers do not dispatch on component classes.
"""
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
import numpy as np
from ..priors import Normal, Fixed


@dataclass(frozen=True)
class Innovation:
    """One signed amplitude multiplying one or more independent innovations."""
    name: str
    loading: np.ndarray
    prior: Normal | Fixed


@dataclass
class ComponentBlock:
    name: str
    transition: np.ndarray
    design: np.ndarray
    initial: np.ndarray
    coefficient_names: tuple
    priors: tuple
    state_names: tuple
    innovations: tuple = ()
    outputs: dict = field(default_factory=dict)
    initial_outputs: dict = field(default_factory=dict)

    def validate(self, steps):
        F, H, B = (np.asarray(a, dtype=float) for a in (self.transition, self.design, self.initial))
        self.transition, self.design, self.initial = F, H, B
        names = (self.name, *self.state_names, *self.coefficient_names, *self.outputs, *self.initial_outputs)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError('Component, coefficient, state and output names must be nonempty strings.')
        d = len(F)
        if F.shape != (d, d) or not d or H.shape != (steps, d):
            raise ValueError(f'{self.name}: invalid transition/design dimensions.')
        if B.shape != (d, len(self.priors)) or len(self.coefficient_names) != len(self.priors):
            raise ValueError(f'{self.name}: initial mapping and priors disagree.')
        if len(self.state_names) != d or not all(np.all(np.isfinite(a)) for a in (F, H, B)):
            raise ValueError(f'{self.name}: invalid state names or nonfinite matrices.')
        if not all(isinstance(x, (Normal, Fixed)) for x in self.priors):
            raise TypeError('Initial coefficients require Normal or Fixed priors.')
        for g in self.innovations:
            R = np.asarray(g.loading)
            if R.ndim != 2 or R.shape[0] != d or not R.shape[1] or not np.all(np.isfinite(R)):
                raise ValueError(f'{g.name}: loading must be a finite state-by-noise matrix.')
            if not isinstance(g.prior, (Normal, Fixed)) or (isinstance(g.prior, Normal) and g.prior.mean != 0):
                raise ValueError('Innovation amplitudes require zero-centred Normal or Fixed priors.')
        for name, row in self.outputs.items():
            if np.asarray(row).shape not in {(d,), (steps, d)} or not np.all(np.isfinite(row)):
                raise ValueError(f'{name}: output must have shape (state,) or (time, state).')
        for name, row in self.initial_outputs.items():
            if np.asarray(row).shape != (len(self.priors),):
                raise ValueError(f'{name}: initial output has the wrong dimension.')
        return self


@runtime_checkable
class Component(Protocol):
    name: str
    def build(self, steps, priors, exog=None) -> ComponentBlock: ...


def mode(value, *, off=False):
    if value not in ({'static', 'dynamic', 'off'} if off else {'static', 'dynamic'}):
        raise ValueError('Mode must be static or dynamic' + (' or off.' if off else '.'))


def features(exog, names, steps):
    import pandas as pd
    if exog is None:
        raise ValueError(f'Covariates required: {names}. Pass exog to fit/predict.')
    if isinstance(exog, pd.DataFrame):
        missing = set(names)-set(exog.columns)
        if missing:
            raise ValueError(f'Missing covariates: {sorted(missing)}')
        values = exog.loc[:, list(names)].to_numpy(dtype=float)
    else:
        values = np.asarray(exog, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
    if values.shape != (steps, len(names)) or not np.all(np.isfinite(values)):
        raise ValueError(f'Expected finite covariates of shape {(steps, len(names))}.')
    return values
