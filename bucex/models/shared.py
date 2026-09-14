"""Shared unobserved components with fixed, scientifically identified loadings."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np
from scipy.linalg import null_space

from ..components import DummySeasonal, LocalLevel, LocalLinearTrend, component_from_dict


def _component(component):
    if not isinstance(component, (LocalLevel, LocalLinearTrend, DummySeasonal)):
        raise TypeError("Shared components support LocalLevel, LocalLinearTrend, or DummySeasonal.")
    # Shared levels are changes relative to a fixed initial reference. Channel
    # baselines remain the intercepts; no second random intercept is introduced.
    if isinstance(component, LocalLevel):
        if component.initial_sd not in (None, 0, 0.0):
            raise ValueError("A shared/departure initial level must be fixed; use initial_sd=0.")
        return replace(component, initial_mean=0.0 if component.initial_mean is None else component.initial_mean, initial_sd=0.0)
    if isinstance(component, LocalLinearTrend):
        if component.initial_level_sd not in (None, 0, 0.0):
            raise ValueError("A shared/departure initial level must be fixed; use initial_level_sd=0.")
        return replace(component, initial_level=0.0 if component.initial_level is None else component.initial_level,
                       initial_level_sd=0.0, initial_slope_sd=0.005 if component.initial_slope_sd is None else component.initial_slope_sd)
    return replace(component, initial_sd=1.0 if component.initial_sd is None else component.initial_sd)


def _name(name):
    name = str(name)
    if not name or "." in name:
        raise ValueError("Shared component names must be nonempty and cannot contain '.'.")
    return name


@dataclass(frozen=True)
class Shared:
    """One component contributing to several channels.

    Loadings are fixed constants on the original response orientation. Omit
    them for unit loadings in every channel. A mapping may select a subset of
    channels; omitted channels then have loading zero. Estimated free loadings
    are deliberately outside the additive model's identification contract.
    """

    name: str
    component: Any
    loadings: Mapping[str, float] | None = None

    def __post_init__(self):
        object.__setattr__(self, "name", _name(self.name))
        object.__setattr__(self, "component", _component(self.component))
        if self.loadings is not None:
            values = {str(k): float(v) for k, v in self.loadings.items()}
            if not values or not np.all(np.isfinite(list(values.values()))) or not any(values.values()):
                raise ValueError("Shared loadings must be finite and include a nonzero value.")
            object.__setattr__(self, "loadings", values)

    def loading_matrix(self, channels: tuple[str, ...]) -> np.ndarray:
        if self.loadings is None:
            return np.ones((len(channels), 1))
        extra = set(self.loadings) - set(channels)
        if extra:
            raise ValueError(f"Unknown channels in shared loadings: {sorted(extra)}")
        return np.asarray([self.loadings.get(name, 0.0) for name in channels])[:, None]

    def to_dict(self):
        return {"kind": "shared", "name": self.name, "component": self.component.to_dict(),
                "loadings": None if self.loadings is None else dict(self.loadings)}


@dataclass(frozen=True)
class Departures:
    """Channel departures whose weighted sum is exactly zero at every time.

    K-1 orthonormal contrast processes span the null space of the fixed channel
    weights. Each component innovation SD is shared across contrasts, so its
    induced prior does not depend on the arbitrary orthonormal basis. This is
    group shrinkage, not independent per-channel model selection.
    """

    name: str
    component: Any
    weights: Mapping[str, float] | None = None

    def __post_init__(self):
        object.__setattr__(self, "name", _name(self.name))
        component = _component(self.component)
        if isinstance(component, LocalLevel) and component.initial_mean != 0.0:
            raise ValueError("Departure initial levels must be zero.")
        if isinstance(component, LocalLinearTrend) and (component.initial_level != 0.0 or component.initial_slope != 0.0):
            raise ValueError("Departure contrast initial means must be zero; use initial_slope_sd for uncertain slopes.")
        if isinstance(component, DummySeasonal) and component.initial_mean is not None and np.any(component.initial_mean):
            raise ValueError("Departure contrast initial means must be zero.")
        object.__setattr__(self, "component", component)
        if self.weights is not None:
            values = {str(k): float(v) for k, v in self.weights.items()}
            a = np.asarray(list(values.values()))
            if not values or not np.all(np.isfinite(a)) or np.any(a < 0.0) or a.sum() <= 0.0:
                raise ValueError("Departure weights must be finite, nonnegative, and have positive sum.")
            total = float(a.sum())
            object.__setattr__(self, "weights", {k: v / total for k, v in values.items()})

    def weight_vector(self, channels: tuple[str, ...]) -> np.ndarray:
        if self.weights is None:
            return np.full(len(channels), 1.0 / len(channels))
        if set(self.weights) != set(channels):
            raise ValueError("Departure weights must name every model channel exactly once.")
        return np.asarray([self.weights[name] for name in channels])

    def loading_matrix(self, channels: tuple[str, ...]) -> np.ndarray:
        basis = null_space(self.weight_vector(channels)[None, :])
        # Fix the otherwise irrelevant column signs for readable archives.
        for j in range(basis.shape[1]):
            if basis[np.argmax(np.abs(basis[:, j])), j] < 0:
                basis[:, j] *= -1
        return basis

    def to_dict(self):
        return {"kind": "departures", "name": self.name, "component": self.component.to_dict(),
                "weights": None if self.weights is None else dict(self.weights)}


def shared_from_dict(value):
    kind = value.get("kind")
    cls = {"shared": Shared, "departures": Departures}.get(kind)
    if cls is None:
        raise ValueError(f"Unknown shared component kind {kind!r}.")
    keyword = "loadings" if cls is Shared else "weights"
    return cls(value["name"], component_from_dict(value["component"]), **{keyword: value.get(keyword)})


__all__ = ["Shared", "Departures"]
