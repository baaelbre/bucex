"""The supported sampler: FFBS for Gaussian paths, Laplace--MH for GEV paths."""
from .config import MCMC, Laplace

__all__ = ["MCMC", "Laplace"]

from .plan import InferencePlan, plan
