"""Compact figures for common trajectories and channel departures."""
from __future__ import annotations

import numpy as np


def _axis(ax, figsize):
    import matplotlib.pyplot as plt

    if ax is None:
        return plt.subplots(figsize=figsize)
    return ax.figure, ax


def _ribbon(ax, fit, values, *, credible_interval, label, color):
    if not 0 < credible_interval < 1:
        raise ValueError("credible_interval must lie in (0, 1).")
    alpha = (1 - credible_interval) / 2
    lower, median, upper = np.quantile(values, [alpha, 0.5, 1 - alpha], axis=0)
    time = np.arange(fit.n_time) if fit.dates is None else fit.dates
    ax.fill_between(time, lower, upper, color=color, alpha=0.18)
    ax.plot(time, median, color=color, label=label)


def plot_shared(fit, *, name="warming", component="level", credible_interval=0.90,
                ax=None, color="C0", title=None, figsize=(9, 4)):
    """Plot posterior median and pointwise interval for a common trajectory."""
    figure, ax = _axis(ax, figsize)
    values = fit.shared_draws(name, component=component)
    _ribbon(ax, fit, values, credible_interval=credible_interval, label=name, color=color)
    ax.set_ylabel(f"shared {component}")
    if title is not None:
        ax.set_title(str(title))
    ax.legend()
    return figure, ax


def plot_departures(fit, *, name="departure", component="level", channels=None,
                    credible_interval=0.90, ax=None, title=None, figsize=(9, 4)):
    """Plot original-scale channel departures; their weighted sum is zero."""
    figure, ax = _axis(ax, figsize)
    selected = fit.channel_names if channels is None else ((channels,) if isinstance(channels, str) else tuple(channels))
    ax.axhline(0, color="0.5", linewidth=0.8)
    for index, channel in enumerate(selected):
        values = fit.departure_draws(channel, name=name, component=component)
        _ribbon(ax, fit, values, credible_interval=credible_interval, label=channel, color=f"C{index % 10}")
    ax.set_ylabel(f"departure {component}")
    if title is not None:
        ax.set_title(str(title))
    ax.legend()
    return figure, ax


def plot_channel_component(fit, *, channel, component="level", credible_interval=0.90,
                           ax=None, color="C0", title=None, figsize=(9, 4)):
    """Plot a full channel level, seasonal component, or per-step slope."""
    figure, ax = _axis(ax, figsize)
    values = fit.channel_component_draws(channel, component)
    _ribbon(ax, fit, values, credible_interval=credible_interval, label=channel, color=color)
    ax.set_ylabel(component)
    if title is not None:
        ax.set_title(str(title))
    ax.legend()
    return figure, ax
