import numpy as np
from scipy.stats import gaussian_kde
from ..results import summarize

COLORS = {'TX': '#24658a', 'TN': '#a44839'}


def _axis(ax):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.4), constrained_layout=True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(direction='out')
    return ax


def _band(ax, index, values, *, color, interval, label=None, axis=(0, 1)):
    s = summarize(values, interval, axis=axis)
    ax.fill_between(index, s['lower'], s['upper'], color=color, alpha=.18, linewidth=0)
    ax.plot(index, s['mean'], color=color, lw=1.5, label=label)


def _density(ax, values, *, color, label, linestyle='-'):
    v = np.asarray(values).ravel()
    v = v[np.isfinite(v)]
    if v.size < 2 or np.ptp(v) == 0:
        ax.axvline(v[0], color=color, label=label, linestyle=linestyle)
        return
    lo, hi = np.quantile(v, [.001, .999])
    x = np.linspace(lo, hi, 250)
    ax.plot(x, gaussian_kde(v)(x), color=color, label=label, linestyle=linestyle)

