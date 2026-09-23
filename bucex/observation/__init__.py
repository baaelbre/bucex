"""Canonical observation families."""
from .base import ObservationModel, ObsSpec
from .gaussian import Gaussian, GaussianObs
from .gev import GEV, GEVObs
from .scale import SeasonalScale, LogScale, StructuralScale, scale_from_dict

Observation = Gaussian | GEV


def observation_from_dict(value):
    family = str(value["family"]).lower()
    scale = scale_from_dict(value.get("scale"))
    if family == "gaussian":
        return Gaussian(scale=scale)
    if family == "gev":
        return GEV(
            value.get("xi_bounds"),
            phi=value.get("phi", "stationary"), scale=scale,
        )
    raise ValueError(f"Unknown observation family '{family}'.")


__all__ = [
    "SeasonalScale",
    "LogScale",
    "StructuralScale",
    "ObservationModel",
    "ObsSpec",
    "Observation",
    "Gaussian",
    "GEV",
    "GaussianObs",
    "GEVObs",
    "observation_from_dict",
]
