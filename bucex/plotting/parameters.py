"""One plot interface for fitted distribution parameters and their components."""
import numpy as np


def plot_parameter_path(fit, parameter, *, component=None, channel=None,
                        link_scale=False, level=.95, ax=None, ylabel=None):
    """Pointwise posterior bands; scale components are always in log units."""
    import matplotlib.pyplot as plt
    if not 0 < level < 1:
        raise ValueError("level must lie in (0,1).")
    values = (fit.parameter_component_draws(parameter,component,channel=channel)
              if component else fit.parameter_path(parameter,channel=channel,link_scale=link_scale))
    lower,median,upper = np.quantile(values,[(1-level)/2,.5,(1+level)/2],axis=0)
    if ax is None:
        _,ax = plt.subplots(figsize=(8,3))
    ax.fill_between(fit.time,lower,upper,alpha=.2)
    ax.plot(fit.time,median)
    label = ("log " if parameter=='sigma' and (link_scale or component) else "")+parameter
    ax.set(xlabel='time',ylabel=ylabel or (label+(' '+component if component else '')))
    return ax


__all__ = ["plot_parameter_path"]
