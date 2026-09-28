"""Checks of physical ordering on unchanged original-scale predictive draws.

These checks diagnose incompatible outcomes. They never sort, censor or reject
samples, and are not a normalized likelihood for order-constrained observations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class OrderingResult:
    """Aggregate and time-specific probabilities of ordering violations.

    Predictive probabilities are Monte Carlo averages over replicated outcomes,
    integrating parameter/state uncertainty and fresh observation noise. They are
    not posterior credible intervals for conditional event probabilities.
    """
    summary: object
    by_time: object
    constraints: tuple[tuple[str, str], ...]
    predictive_violations: np.ndarray | None
    observed_violations: np.ndarray | None

    def plot(self, **kwargs):
        """Plot probabilities of ordering violations over time."""
        from ..plotting.dependence import plot_ordering

        return plot_ordering(self, **kwargs)


def _channels(channel_names):
    names = tuple(str(name) for name in channel_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("channel_names must contain at least two unique names.")
    return names


def ordering_diagnostics(
    predictive=None,
    *,
    channel_names: Sequence[str],
    constraints: Sequence[tuple[str, str]],
    observed=None,
    dates=None,
    tolerance: float = 0.0,
) -> OrderingResult:
    """Check inequalities ``lower <= upper`` for each supplied channel pair.

    ``predictive`` has shape ``(draws, time, channels)``; ``observed`` has shape
    ``(time, channels)``. Both use original response units, including minima.
    Supply either or both. The ``any`` row is the probability that at least one
    listed constraint fails in a time block, not a sum of pair probabilities.
    """
    import pandas as pd

    names = _channels(channel_names)
    pairs = tuple(tuple(pair) for pair in constraints)
    if not pairs or any(len(pair) != 2 for pair in pairs):
        raise ValueError("constraints must contain (lower, upper) channel pairs.")
    if len(set(pairs)) != len(pairs):
        raise ValueError("constraints must not contain duplicate pairs.")
    for lower, upper in pairs:
        if lower not in names or upper not in names or lower == upper:
            raise ValueError(f"Invalid ordering pair {(lower, upper)!r} for channels {names}.")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite nonnegative value in response units.")
    if predictive is None and observed is None:
        raise ValueError("Supply predictive draws, observed data, or both.")
    prediction = None if predictive is None else np.asarray(predictive, dtype=float)
    actual = None if observed is None else np.asarray(observed, dtype=float)
    if prediction is not None:
        if prediction.ndim != 3 or prediction.shape[0] == 0 or prediction.shape[1] == 0 or prediction.shape[2] != len(names):
            raise ValueError("predictive must have shape (positive draws, positive time, channels).")
        if not np.all(np.isfinite(prediction)):
            raise ValueError("predictive draws must be finite.")
        n_time = prediction.shape[1]
    else:
        n_time = actual.shape[0] if actual.ndim == 2 else 0
    if actual is not None:
        if actual.shape != (n_time, len(names)) or n_time == 0 or not np.all(np.isfinite(actual)):
            raise ValueError("observed must contain finite values with shape (time, channels).")
    time = np.arange(n_time) if dates is None else np.asarray(dates).reshape(-1)
    if time.size != n_time:
        raise ValueError("dates must have length equal to time.")
    indices = [(names.index(lower), names.index(upper)) for lower, upper in pairs]
    pred = None if prediction is None else np.stack([
        prediction[..., left] > prediction[..., right] + tolerance for left, right in indices
    ], axis=-1)
    obs = None if actual is None else np.stack([
        actual[..., left] > actual[..., right] + tolerance for left, right in indices
    ], axis=-1)
    labels = [f"{lower} <= {upper}" for lower, upper in pairs] + ["any"]
    all_pred = None if pred is None else np.concatenate((pred, np.any(pred, axis=-1, keepdims=True)), axis=-1)
    all_obs = None if obs is None else np.concatenate((obs, np.any(obs, axis=-1, keepdims=True)), axis=-1)
    summary, by_time = [], []
    for index, label in enumerate(labels):
        row = {"constraint": label, "n_time": n_time}
        if all_pred is not None:
            probabilities = np.mean(all_pred[..., index], axis=0)
            row.update(predictive_probability=float(np.mean(probabilities)),
                       maximum_time_probability=float(np.max(probabilities)),
                       n_predictive_draws=int(prediction.shape[0]))
        if all_obs is not None:
            row.update(observed_violations=int(np.sum(all_obs[..., index])),
                       observed_fraction=float(np.mean(all_obs[..., index])))
        summary.append(row)
        for t in range(n_time):
            item = {"time": time[t], "constraint": label}
            if all_pred is not None:
                item["predictive_probability"] = float(probabilities[t])
            if all_obs is not None:
                item["observed_violation"] = bool(all_obs[t, index])
            by_time.append(item)
    return OrderingResult(pd.DataFrame(summary), pd.DataFrame(by_time), pairs, pred, obs)


def compound_event_probability(
    predictive,
    *,
    channel_names: Sequence[str],
    events: Mapping[str, tuple[str, float]],
    operation: str = "all",
) -> np.ndarray:
    """Monte Carlo probability of a compound event in each time block.

    Example: ``events={"TXx": (">", 35), "TNn": (">", 20)}`` with
    ``operation="all"``. The joint posterior predictive draws must be retained
    intact: independent resampling of channels would destroy their dependence.
    """
    names = _channels(channel_names)
    values = np.asarray(predictive, dtype=float)
    if values.ndim != 3 or values.shape[-1] != len(names) or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("predictive must have shape (positive draws, positive time, channels).")
    if not np.all(np.isfinite(values)):
        raise ValueError("predictive draws must be finite.")
    if not events or operation not in {"all", "any"}:
        raise ValueError("Supply nonempty events and operation='all' or 'any'.")
    comparisons = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal}
    indicators = []
    for channel, event in events.items():
        if channel not in names or len(event) != 2 or event[0] not in comparisons:
            raise ValueError(f"Invalid event {channel!r}: {event!r}.")
        threshold = float(event[1])
        if not np.isfinite(threshold):
            raise ValueError("Event thresholds must be finite.")
        indicators.append(comparisons[event[0]](values[..., names.index(channel)], threshold))
    stacked = np.stack(indicators, axis=-1)
    combined = np.all(stacked, axis=-1) if operation == "all" else np.any(stacked, axis=-1)
    return np.mean(combined, axis=0)
