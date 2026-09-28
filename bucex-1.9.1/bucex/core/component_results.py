"""Scientific component summaries for private channel models.

All public quantities in this module use the original response orientation.
In particular, a warming contribution to a lower-temperature extreme stays
positive even though that channel's GEV likelihood is fitted to its negative.
"""
from __future__ import annotations

import numpy as np


def _coordinate(component, name: str) -> int | None:
    """Locate a semantic coordinate without assuming user-chosen state names."""
    from ..components import DummySeasonal, LocalLevel, LocalLinearTrend

    if isinstance(component, LocalLevel):
        return 0 if name == "level" else None
    if isinstance(component, LocalLinearTrend):
        if name == "level":
            return 0
        if name == "slope" and component.trend_mode != "off":
            return 1
        return None
    if isinstance(component, DummySeasonal):
        return 0 if name in {"season", "seasonal"} and component.state_dim else None
    return None


def channel_design(compiled, channel: str, component: str, n_time: int) -> np.ndarray:
    """Design for one original-scale scientific component of one channel."""
    if channel not in compiled.channel_names:
        raise KeyError(f"Unknown channel {channel!r}; choose from {compiled.channel_names}.")
    name = "seasonal" if component == "season" else str(component)
    if name not in {"level", "slope", "seasonal"}:
        raise ValueError("component must be 'level', 'slope', or 'seasonal'.")
    index = compiled.channel_names.index(channel)
    specification = compiled.model.channels[index]
    sign = float(specification.transform_sign)
    design = np.zeros((int(n_time), compiled.state_dim))
    block = next(block for block in compiled.blocks if block.name == channel)
    offset = int(block.state_slice.start)
    for local in specification.components:
        coordinate = _coordinate(local, name)
        if coordinate is not None:
            design[:, offset + coordinate] = sign
        offset += local.state_dim
    return design


def univariate_design(compiled, component: str, sign: float = 1.0) -> np.ndarray:
    """A semantic scalar component, including user-defined state names."""
    design = np.zeros(compiled.state_dim)
    for local in compiled.model.components:
        coordinate = _coordinate(local, component)
        if coordinate is not None:
            state_slice = compiled.component_slices[compiled._component_key(local)]
            design[state_slice.start + coordinate] += float(sign)
    return design


class ComponentResultMethods:
    """Public helpers for private channel and univariate components."""

    def _component_projection(self, design, *, combine_chains: bool, include_initial: bool):
        paths = self.state_draws[:, :, 0 if include_initial else 1 :]
        values = np.einsum("cdtm,m->cdt", paths, np.asarray(design))
        return values.reshape((-1, values.shape[-1])) if combine_chains else values

    def channel_component_draws(
        self, channel: str, component: str = "level", *,
        combine_chains: bool = True, include_initial: bool = False,
    ) -> np.ndarray:
        """A channel's level, slope, or seasonality on its original scale.

        Level excludes seasonality and observation noise. Slope is per model step.
        """
        if not self.is_multiseries_model:
            raise ValueError("channel_component_draws requires a multiseries result.")
        design = channel_design(self.compiled, channel, component, 1)[0]
        return self._component_projection(design, combine_chains=combine_chains, include_initial=include_initial)

    def component_draws(
        self, component: str, *, channel: str | None = None,
        combine_chains: bool = True, include_initial: bool = False,
    ) -> np.ndarray:
        """Unified component access for univariate and multiseries results."""
        if self.is_multiseries_model:
            if channel is None:
                raise ValueError(f"Choose channel from {self.channel_names}.")
            return self.channel_component_draws(channel, component, combine_chains=combine_chains, include_initial=include_initial)
        if channel is not None:
            raise ValueError("channel= is only valid for multiseries results.")
        if component in {"level", "slope", "season", "seasonal"}:
            design = univariate_design(self.compiled, component, self.transform_sign)
            return self._component_projection(design, combine_chains=combine_chains, include_initial=include_initial)
        return self.state_original(component, combine_chains=combine_chains, include_initial=include_initial)
