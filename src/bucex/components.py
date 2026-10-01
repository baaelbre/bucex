"""Structural components. Static means estimated but without innovations."""
from dataclasses import dataclass


@dataclass(frozen=True)
class LocalLevel:
    mode: str = "dynamic"

    def __post_init__(self):
        if self.mode not in {"dynamic", "static"}:
            raise ValueError("LocalLevel.mode must be dynamic or static.")


@dataclass(frozen=True)
class LocalLinearTrend:
    level_mode: str = "dynamic"
    trend_mode: str = "dynamic"

    def __post_init__(self):
        if self.level_mode not in {"dynamic", "static"}:
            raise ValueError("level_mode must be dynamic or static.")
        if self.trend_mode not in {"dynamic", "static", "off"}:
            raise ValueError("trend_mode must be dynamic, static or off.")


@dataclass(frozen=True)
class DummySeasonal:
    period: int = 4
    mode: str = "dynamic"

    def __post_init__(self):
        if isinstance(self.period, bool) or not isinstance(self.period, int) or self.period < 2:
            raise ValueError("period must be an integer >= 2.")
        if self.mode not in {"dynamic", "static", "off"}:
            raise ValueError("seasonal mode must be dynamic, static or off.")
