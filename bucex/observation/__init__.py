"""Canonical observation families."""
from .base import ObservationModel, ObsSpec
from .gaussian import Gaussian, GaussianObs
from .gev import GEV, GEVObs
from .scale import SeasonalScale

Observation = Gaussian | GEV


def observation_from_dict(value):
    family = str(value["family"]).lower()
    scale = SeasonalScale(**value["scale"]) if value.get("scale") else None
    if family == "gaussian":
        return Gaussian(scale=scale)
    if family == "gev":
        return GEV(
            tuple(value.get("xi_bounds", (-0.5, 0.5))),
            phi=value.get("phi", "stationary"), scale=scale,
        )
    raise ValueError(f"Unknown observation family '{family}'.")


__all__ = [
    "SeasonalScale",
    "ObservationModel",
    "ObsSpec",
    "Observation",
    "Gaussian",
    "GEV",
    "GaussianObs",
    "GEVObs",
    "observation_from_dict",
]
