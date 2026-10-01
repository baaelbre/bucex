"""Composable single-response and named-channel models."""
from __future__ import annotations
from dataclasses import dataclass, field
from .components import LocalLevel, LocalLinearTrend, DummySeasonal
from .distributions import Gaussian, GEV
from .priors import Priors, Pooling, Normal


@dataclass(frozen=True)
class Model:
    observation: Gaussian | GEV
    components: tuple = field(default_factory=lambda: (LocalLinearTrend(),))
    priors: Priors = field(default_factory=Priors)

    def __post_init__(self):
        object.__setattr__(self, "components", tuple(self.components))
        if not isinstance(self.observation, (Gaussian, GEV)):
            raise TypeError("observation must be Gaussian or GEV.")
        if not isinstance(self.priors, Priors):
            raise TypeError("priors must be Priors(...).")
        if any(not isinstance(c, (LocalLevel, LocalLinearTrend, DummySeasonal)) for c in self.components):
            raise TypeError("Supported components: LocalLevel, LocalLinearTrend, DummySeasonal.")
        if sum(isinstance(c, (LocalLevel, LocalLinearTrend)) for c in self.components) != 1:
            raise ValueError("Use exactly one LocalLevel or LocalLinearTrend component.")
        seasons = [c for c in self.components if isinstance(c, DummySeasonal)]
        if len(seasons) > 1:
            raise ValueError("Use at most one DummySeasonal component.")
        scale = self.observation.scale
        if scale and seasons and scale.period != seasons[0].period:
            raise ValueError("Seasonal state and observation scale must have the same period.")

    @property
    def trend(self):
        c = next(c for c in self.components if isinstance(c, (LocalLevel, LocalLinearTrend)))
        return LocalLinearTrend(c.mode, "off") if isinstance(c, LocalLevel) else c

    @property
    def season(self):
        return next((c for c in self.components if isinstance(c, DummySeasonal) and c.mode != 'off'), None)

    @property
    def period(self):
        season = next((c for c in self.components if isinstance(c, DummySeasonal)), None)
        return season.period if season else self.observation.scale.period if self.observation.scale else 1

    @property
    def active(self):
        return tuple(c for c, yes in (("level", self.trend.level_mode == "dynamic"),
                    ("slope", self.trend.trend_mode == "dynamic"),
                    ("seasonal", self.season is not None and self.season.mode == "dynamic")) if yes)


@dataclass(frozen=True)
class Channel:
    name: str
    model: Model

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or '/' in self.name or '\\' in self.name:
            raise ValueError("Channel name must be a nonempty string without path separators.")
        if not isinstance(self.model, Model):
            raise TypeError("Channel.model must be a Model.")


@dataclass(frozen=True)
class MultiSeriesModel:
    channels: tuple[Channel, ...]
    pooling: Pooling | None = None

    def __post_init__(self):
        object.__setattr__(self, "channels", tuple(self.channels))
        if not self.channels or any(not isinstance(c, Channel) for c in self.channels):
            raise ValueError("Provide one or more Channel objects.")
        if len({c.name for c in self.channels}) != len(self.channels):
            raise ValueError("Channel names must be unique.")
        if self.pooling is not None and not isinstance(self.pooling, Pooling):
            raise TypeError("pooling must be Pooling or None.")
        if self.pooling:
            for name in ("level", "slope", "seasonal"):
                if getattr(self.pooling, name) is None:
                    continue
                members = [c for c in self.channels if name in c.model.active]
                if not members:
                    raise ValueError(f"No active {name} component to pool.")
                if any(not isinstance(getattr(c.model.priors, name), Normal) for c in members):
                    raise ValueError(f"A pooled {name} amplitude must have a zero-centred Normal prior.")
