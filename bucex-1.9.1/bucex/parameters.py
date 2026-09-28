"""Readable parameter declarations, independent of the observation family.

``Constant`` means estimated but constant in time. ``Latent`` associates a
parameter with structural components. The current Gaussian/GEV adapters bind
location to the identity link and scale to the log link; unsupported bindings
fail at construction, rather than silently ignoring a declaration.
"""
from dataclasses import dataclass, replace
from collections.abc import Mapping

from .components import LocalLevel, LocalLinearTrend, DummySeasonal


@dataclass(frozen=True)
class Constant:
    """An unknown time-constant parameter, with its prior supplied to fit()."""


@dataclass(frozen=True)
class Latent:
    """An additive structural predictor for a distribution parameter.

    Location priors are the ordinary ``fs_priors`` passed to ``fit``.
    ``link=None`` selects the family's natural identity link.
    """
    components: tuple
    link: str | None = None
    priors: object = None

    def __post_init__(self):
        object.__setattr__(self, "components", tuple(self.components))
        if not self.components:
            raise ValueError("Latent requires structural components.")
        if self.link not in {None, "identity", "log"}:
            raise ValueError("Supported links are identity and log.")


def resolve_parameters(observation, components, parameters):
    """Lower named parameters to the existing executable model representation."""
    components = tuple(components)
    if parameters is None:
        return observation, components
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must map distribution parameter names to declarations.")
    allowed = {"mu", "sigma"} | ({"xi"} if observation.name == "gev" else set())
    unknown = set(parameters) - allowed
    if unknown:
        raise ValueError(f"Unknown {observation.name} parameters: {sorted(unknown)}.")
    location = parameters.get("mu")
    if location is not None:
        if components:
            raise ValueError("Declare location with components= or parameters['mu'], not both.")
        if isinstance(location, Constant):
            components = (LocalLinearTrend(level_mode="static", trend_mode="off"),)
        elif isinstance(location, Latent):
            if location.link not in {None, "identity"}:
                raise ValueError("Location uses the identity link.")
            if location.priors is not None:
                raise ValueError("Pass location priors as fit(priors=fs_priors(...)).")
            components = tuple(LocalLinearTrend(level_mode=c.mode, trend_mode="off",
                               level_name=c.name, initial_level=c.initial_mean, initial_level_sd=c.initial_sd)
                               if isinstance(c, LocalLevel) else c for c in location.components)
        else:
            raise TypeError("mu must be Constant() or Latent(...).")
    scale = parameters.get("sigma")
    if scale is not None:
        if observation.scale is not None:
            raise ValueError("Declare scale on the observation or in parameters, not both.")
        if isinstance(scale, Constant):
            # Select the same private continuous kernel used by joint models.
            from .observation.scale import LogScale
            observation = replace(observation, scale=LogScale())
        elif isinstance(scale, Latent):
            raise ValueError("Scale evolution is outside the BUCEX 1.8.5 paper API; use Constant() or SeasonalScale().")
        else:
            raise TypeError("sigma must be Constant() or Latent(...).")
    if "xi" in parameters and not isinstance(parameters["xi"], Constant):
        raise NotImplementedError("GEV shape is estimated but constant in 1.7; latent shape is not implemented.")
    return observation, components


__all__ = ["Constant", "Latent"]
