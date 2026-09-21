"""Inference configuration shared by every parameterization."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class MCMC:
    """Retained draws, warmup and independent-chain execution.

    ``chain_workers=1`` runs serially; larger values use spawned processes,
    capped at ``chains``. Seeded chains do not depend on the worker count.
    Put script entry points behind an ``if __name__ == '__main__'`` guard.
    """

    draws: int = 1000
    warmup: int = 1000
    thin: int = 1
    chains: int = 4
    seed: int | None = None
    progress: bool = False
    progress_every: int | None = None
    adapt: bool = True
    chain_workers: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.chain_workers, bool) or not isinstance(self.chain_workers, Integral) or self.chain_workers < 1:
            raise ValueError("chain_workers must be a positive integer.")
        object.__setattr__(self, "chain_workers", int(self.chain_workers))
        if int(self.draws) < 1 or int(self.warmup) < 0 or int(self.thin) < 1:
            raise ValueError("draws and thin must be positive; warmup must be non-negative.")
        if int(self.chains) < 1:
            raise ValueError("chains must be positive.")
        if self.progress_every is not None and int(self.progress_every) < 1:
            raise ValueError("progress_every must be positive when supplied.")

    @property
    def iterations(self) -> int:
        return int(self.warmup) + 1 + (int(self.draws) - 1) * int(self.thin)

    # FS kernels use the original compact names internally.
    @property
    def n_iter(self) -> int:
        return self.iterations

    @property
    def burn(self) -> int:
        return int(self.warmup)


@dataclass(frozen=True)
class GibbsConfig:
    """Deprecated 0.3 spelling, translated once at the public API boundary."""

    n_iter: int = 2000
    burn: int = 1000
    thin: int = 1
    seed: int | None = None
    progress: bool = False
    progress_every: int | None = None

    def __post_init__(self) -> None:
        if int(self.n_iter) <= int(self.burn) or int(self.burn) < 0:
            raise ValueError("Require n_iter > burn >= 0.")
        if int(self.thin) < 1:
            raise ValueError("thin must be positive.")

    def to_mcmc(self, *, chains: int = 1) -> MCMC:
        return MCMC(
            draws=len(range(int(self.burn), int(self.n_iter), int(self.thin))),
            warmup=int(self.burn),
            thin=int(self.thin),
            chains=int(chains),
            seed=self.seed,
            progress=bool(self.progress),
            progress_every=self.progress_every,
        )


@dataclass(frozen=True)
class Laplace:
    """Controls for Laplace approximations and Laplace-MH proposals.

    ``mh_steps`` is the number of full-trajectory independence-MH proposals per
    outer MCMC iteration when ``engine="laplace_mh"``.  ``draw_attempts`` is
    retained only for the explicitly approximate ``engine="laplace"`` path.
    For continuous FS GEV coefficients, ``max_iterations`` and ``tolerance``
    also control the deterministic conditional-mode Gaussian reference.
    That reference is always corrected against the exact target by a slice
    update; optimizer convergence is an efficiency diagnostic, not MCMC
    convergence or a replacement for likelihood correction.
    """

    max_iterations: int = 30
    tolerance: float = 1e-5
    curvature_floor: float = 1e-6
    maximum_variance: float = 1e8
    draw_attempts: int = 30
    mh_steps: int = 1

    def __post_init__(self) -> None:
        if int(self.max_iterations) < 1 or float(self.tolerance) <= 0.0:
            raise ValueError("Laplace iterations and tolerance must be positive.")
        if float(self.curvature_floor) <= 0.0 or float(self.maximum_variance) <= 0.0:
            raise ValueError("Laplace curvature and variance controls must be positive.")
        if int(self.draw_attempts) < 1:
            raise ValueError("Laplace.draw_attempts must be positive.")
        if int(self.mh_steps) < 1:
            raise ValueError("Laplace.mh_steps must be positive.")


@dataclass(frozen=True)
class SharedSampler:
    """Additional exact state updates for mixed shared-component models.

    After each joint Laplace-MH update, ``elliptical_slice_steps`` full block
    sweeps refresh the Gaussian-prior state paths using exact likelihoods.
    Set zero for an explicit sampler ablation. Gaussian FFBS models need no
    supplementary slice sweeps. Exceeding the evaluation limit aborts the fit.
    """

    elliptical_slice_steps: int = 1
    maximum_slice_evaluations: int = 200

    def __post_init__(self) -> None:
        for name, lower in (("elliptical_slice_steps", 0), ("maximum_slice_evaluations", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < lower:
                raise ValueError(f"{name} must be an integer >= {lower}.")
            object.__setattr__(self, name, int(value))


@dataclass(frozen=True)
class HierarchicalSampler:
    """Execution controls for a multi-series hierarchical sampler.

    ``initializer='laplace'`` obtains a fast approximate path for every GEV
    channel before exact Laplace-MH starts. It changes only the chain starting
    point. ``channel_workers`` optionally updates conditionally
    independent channels in parallel within each hierarchical Gibbs sweep.
    """

    initializer: str = "laplace"
    channel_workers: int = 1

    def __post_init__(self) -> None:
        initializer = str(self.initializer).lower().replace("-", "_")
        initializer = {"none": "data", "default": "data"}.get(
            initializer, initializer
        )
        if initializer not in {"data", "laplace"}:
            raise ValueError("initializer must be 'data' or 'laplace'.")
        if int(self.channel_workers) < 1:
            raise ValueError("channel_workers must be at least one.")
        object.__setattr__(self, "initializer", initializer)
        object.__setattr__(self, "channel_workers", int(self.channel_workers))
