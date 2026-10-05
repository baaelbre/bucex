"""Extensible native plotting: handlers consume BUCEX result objects and axes."""
from pathlib import Path
from ..results import FitResult, ChannelResult
from ..prediction import Predictive
from .common import COLORS, _axis
from .trajectories import trajectory, forecast, seasonal_cycle, risk_curve, observation_scale
from .diagnostics import diagnostic
from .parameters import parameter_plot

_HANDLERS = {}


def register_plot(name, handler, *, replace=False):
    """Register handler(channel_or_prediction, ax, **context), with no titles imposed."""
    if not isinstance(name, str) or not name or not callable(handler):
        raise TypeError('A plot needs a nonempty name and callable handler.')
    if name in _HANDLERS and not replace:
        raise ValueError(f'Plot {name!r} is already registered.')
    _HANDLERS[name] = handler


for _name in ('level', 'slope', 'location', 'seasonal', 'component', 'risk', 'return_level'):
    register_plot(_name, trajectory)
for _name in ('normal_qq', 'pit', 'acf'):
    register_plot(_name, diagnostic)
for _name in ('trace', 'posterior', 'prior_posterior'):
    register_plot(_name, parameter_plot)
register_plot('forecast', forecast)
register_plot('risk_curve', risk_curve)
register_plot('observation_scale', observation_scale)
register_plot('seasonal_cycle', seasonal_cycle)
register_plot('cycle', seasonal_cycle)  # compatibility alias; use component='cycle' for a Cycle state


def plot(value, type='level', *, channel=None, ax=None, interval=.95, color=None,
         label=None, path=None, dpi=180, **kwargs):
    """Return a Matplotlib Axes; central curves are means and bands pointwise.

    type='component', component='regression.wind' plots any compiled output.
    A user handler can be added with bucex.plots.register_plot.
    """
    if type not in _HANDLERS:
        raise ValueError(f'Unknown plot type {type!r}. Available: {tuple(_HANDLERS)}')
    if not 0 < interval < 1:
        raise ValueError('interval must lie between zero and one.')
    fit = value if isinstance(value, FitResult) else None
    c = fit.channel(channel) if fit else value
    if isinstance(c, Predictive):
        name = c.channel_name(channel)
    elif isinstance(c, ChannelResult):
        name = c.name
    else:
        raise TypeError('plot accepts a FitResult, ChannelResult or Predictive object.')
    ax = _axis(ax)
    _HANDLERS[type](c, ax, fit=fit, name=name, type=type,
        color=color or COLORS.get(name[:2], '#24658a'), interval=interval, label=label, **kwargs)
    if path is not None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(path, dpi=dpi, bbox_inches='tight')
    return ax

__all__ = ['plot', 'register_plot']
