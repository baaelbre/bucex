"""Figures for private structural channel components."""
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
