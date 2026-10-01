from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MCMC:
    draws: int = 1000
    warmup: int = 1000
    chains: int = 4
    workers: int = 1
    seed: int = 1000
    chain_ids: tuple[int, ...] | None = None
    progress: bool = True
    progress_every: int = 100

    def __post_init__(self):
        for name in ("draws", "chains", "workers", "progress_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("warmup", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        ids = tuple(range(self.chains)) if self.chain_ids is None else tuple(self.chain_ids)
        if len(ids) != self.chains or len(set(ids)) != self.chains or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("chain_ids must contain one unique nonnegative integer per chain.")
        object.__setattr__(self, 'chain_ids', ids)


@dataclass(frozen=True)
class Laplace:
    max_iterations: int = 30
    tolerance: float = 1e-5
    mh_steps: int = 1
    curvature_floor: float = 1e-6
    maximum_variance: float = 1e8

    def __post_init__(self):
        for name in ("max_iterations", "mh_steps"):
            x = getattr(self, name)
            if type(x) is not int or x < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("tolerance", "curvature_floor", "maximum_variance"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite.")
