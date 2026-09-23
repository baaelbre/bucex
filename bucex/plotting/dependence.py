"""Residual-correlation and physical-order diagnostics."""
from __future__ import annotations

import numpy as np


def plot_copula(fit, *, ax=None, title=None, annotate=True, phase=None, save=None, colorbar=True):
    """Posterior median original-orientation Gaussian score correlations."""
    import matplotlib.pyplot as plt
    from .core import _save_result

    if fit.model.copula.seasonal and phase is None:
        if ax is not None:
            raise ValueError("Specify phase=1,...,period to plot a single seasonal matrix on an axis.")
        n = fit.model.copula.period
        phases=list(range(1,n+1))
        labels=[f'Phase {i}' for i in phases]
        if fit.model.copula.structure=='seasons':
            phases=[1,3,6,9] if n==12 else [1,2,3,4]
            labels=['DJF','MAM','JJA','SON']
        n=len(phases)
        columns = 2 if n==4 else min(3,n)
        figure, axes = plt.subplots(int(np.ceil(n/columns)),columns,
            figsize=(4.3*columns,4.0*int(np.ceil(n/columns))),squeeze=False,layout='constrained')
        for index,axis in enumerate(axes.flat):
            if index >= n:
                axis.set_visible(False)
            else:
                plot_copula(fit,ax=axis,phase=phases[index],annotate=annotate,title=labels[index],colorbar=False)
        if colorbar:
            figure.colorbar(axes.flat[0].images[0],ax=axes.ravel().tolist(),shrink=.75,
                            label='Residual normal-score correlation')
        result = figure,axes
        _save_result(result,save)
        return result
    median = np.median(fit.copula_correlation_draws(phase=phase), axis=0)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    artist = ax.imshow(median, vmin=-1, vmax=1, cmap="RdBu_r")
    positions = np.arange(len(fit.channel_names))
    ax.set_xticks(positions, fit.channel_names, rotation=45, ha="right")
    ax.set_yticks(positions, fit.channel_names)
    if colorbar:
        ax.figure.colorbar(artist, ax=ax, label="Residual normal-score correlation")
    if annotate:
        for i, j in np.ndindex(median.shape):
            ax.text(j, i, f"{median[i, j]:.2f}", ha="center", va="center",
                    fontsize=9,
                    color="white" if abs(median[i, j]) > .65 else "black")
    if title is not None:
        ax.set_title(title)
    result = (ax.figure, ax)
    _save_result(result, save)
    return result


def plot_ordering(result, *, constraints=None, ax=None, title=None, save=None):
    """Time-varying probability of invalid outcomes in unchanged joint draws."""
    import matplotlib.pyplot as plt
    from .core import _save_result

    if "predictive_probability" not in result.by_time:
        raise ValueError("Plotting predictive ordering requires replicated draws.")
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    labels = tuple(result.by_time["constraint"].unique()) if constraints is None else tuple(constraints)
    for label in labels:
        values = result.by_time.loc[result.by_time["constraint"] == label]
        if values.empty:
            raise KeyError(f"Unknown ordering label {label!r}.")
        ax.plot(values["time"], values["predictive_probability"], label=label,
                linewidth=2 if label == "any" else 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability of an ordering violation")
    ax.legend()
    if title is not None:
        ax.set_title(title)
    result = (ax.figure, ax)
    _save_result(result, save)
    return result
