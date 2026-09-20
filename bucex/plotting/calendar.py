"""Calendar diagnostics shared by reports and publication panels."""
from __future__ import annotations

import calendar
import numpy as np
import pandas as pd

from ..diagnostics.calendar import pit_by_month


def plot_pit_calendar(values, dates, *, bins=10, title=None):
    """Twelve PIT histograms; panel labels retain their case counts."""
    import matplotlib.pyplot as plt
    summary = pit_by_month(values, dates)
    if bins < 2 or int(bins) != bins:
        raise ValueError("bins must be an integer of at least two.")
    values, dates = np.asarray(values), pd.DatetimeIndex(dates)
    figure, axes = plt.subplots(4, 3, figsize=(10.6, 8.8), sharex=True,
                                squeeze=False, layout="constrained")
    for month, ax in enumerate(axes.flat, 1):
        selected = values[dates.month == month]
        ax.hist(selected, bins=np.linspace(0, 1, bins+1), alpha=.65, edgecolor="white")
        ax.axhline(len(selected)/bins, color=".4", ls="--", lw=.8)
        ax.set_title(f"{calendar.month_abbr[month]} (n={len(selected)})", loc="left", fontsize=11)
        ax.set_xlim(0, 1)
    for ax in axes[-1]:
        ax.set_xlabel("PIT")
    for ax in axes[:, 0]:
        ax.set_ylabel("Count")
    if title:
        figure.suptitle(title)
    return figure, axes, summary


def plot_monthly_score_sd(summary, *, ax=None, color=None):
    """Normal-score SD; the horizontal line is a descriptive reference."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
    ax.plot(summary.month, summary.normal_score_sd, "o-", color=color)
    ax.axhline(1, color=".4", ls="--", lw=1)
    ax.set_xticks([1, 3, 5, 7, 9, 11], ["Jan", "Mar", "May", "Jul", "Sep", "Nov"])
    ax.set_ylabel("Normal-score SD")
    return ax.figure, ax


__all__ = ["plot_pit_calendar", "plot_monthly_score_sd"]
