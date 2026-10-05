"""Zero-sum initial cycle with optional dummy-seasonal innovations."""
from dataclasses import dataclass
import numpy as np
from .base import ComponentBlock, Innovation, mode
from ..distributions import contrasts


@dataclass(frozen=True)
class DummySeasonal:
    period: int = 4
    mode: str = 'dynamic'
    name: str = 'seasonal'
    initial_prior: object = None
    innovation_prior: object = None

    def __post_init__(self):
        mode(self.mode, off=True)
        if type(self.period) is not int or self.period < 2:
            raise ValueError('period must be an integer of at least two.')

    def build(self, steps, priors, exog=None):
        d = self.period-1
        F = np.zeros((d, d)); F[0] = -1
        if d > 1:
            F[1:, :-1] = np.eye(d-1)
        C = contrasts(self.period)
        # Phase zero is the first observation, not the unobserved time zero.
        first = C[np.r_[0, np.arange(self.period-1, 1, -1)]]
        B = np.linalg.solve(F, first)
        H = np.zeros((steps, d)); H[:, 0] = 1
        initial = self.initial_prior or priors.initial_seasonal
        groups = (Innovation(self.name, np.eye(d)[:, :1], self.innovation_prior or priors.seasonal),) if self.mode == 'dynamic' else ()
        exports = {f'initial_{self.name}[{i}]': C[i] for i in range(self.period)}
        return ComponentBlock(self.name, F, H, B,
            tuple(f'{self.name}.contrast[{i}]' for i in range(d)), (initial,)*d,
            tuple(f'{self.name}[{i+1}]' for i in range(d)), groups,
            {self.name: np.eye(d)[0]}, exports)
