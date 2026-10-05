"""Posterior and predictive component, risk and seasonal-cycle plots."""
import numpy as np
from ..prediction import Predictive
from ..results import FitResult
from .common import _band


def trajectory(c, ax, *, type, name, color, interval, label, component=None,
               threshold=None, tail='upper', years=20, phase=None, rate_scale=None, **kwargs):
    if type == 'risk' and threshold is None:
        raise ValueError('A risk plot requires threshold=...')
    key = component if type == 'component' else type
    if not key:
        raise ValueError('Specify component= with type="component".')
    predictive = isinstance(c, Predictive)
    if predictive:
        values = (c.risk(threshold, channel=name, tail=tail) if type == 'risk' else
                  c.return_level(years, channel=name, tail=tail) if type == 'return_level' else c.path(key, channel=name))
    else:
        values = (c.risk(threshold, tail=tail) if type == 'risk' else
                  c.return_level(years, tail=tail) if type == 'return_level' else c.path(key))
    if type == 'slope':
        values = values*(10*c.steps_per_year if rate_scale is None else rate_scale)
    ids = np.arange(len(c.index))
    if phase is not None:
        if predictive:
            raise ValueError('For forecasts select seasons by their dates, or omit phase.')
        if type_of(phase) is not int or not 0 <= phase < c.model.period:
            raise ValueError('phase must be an integer in [0, period).')
        ids = ids[ids%c.model.period == phase]
    _band(ax, np.asarray(c.index)[ids], values[..., ids], color=color, interval=interval,
          axis=0 if predictive else (0, 1), label=label)
    ax.set_ylabel('Probability' if type == 'risk' else 'Rate per decade' if type == 'slope' and rate_scale is None else key.replace('_', ' ').capitalize())
    if type == 'risk':
        ax.set_ylim(0, 1)


type_of = type


def forecast(c, ax, *, name, color, interval, history=None, observed=None, **kwargs):
    if not isinstance(c, Predictive):
        raise TypeError('A forecast plot requires a Predictive object.')
    if history is not None:
        h = history.channel(name) if isinstance(history, FitResult) else history
        ax.plot(h.index, h.y, '.', color='.65', ms=3, label='Observed')
    _band(ax, c.index, c.y[name], color=color, interval=interval, axis=0, label='Predictive mean')
    ax.plot(c.index, c.location[name].mean(axis=0), '--', color='.2', lw=1, label='Location')
    if observed is not None:
        ax.plot(c.index, observed, '.', color='black', ms=4, label='Held out')
    ax.set_ylabel('Response')


def seasonal_cycle(c, ax, *, color, interval, phase_labels=None, **kwargs):
    if isinstance(c, Predictive) or not c.model.season:
        raise ValueError('A seasonal-cycle comparison requires a fit with DummySeasonal.')
    name = c.model.season.name
    p = c.model.season.period
    initial = np.stack([c.parameters[f'initial_{name}[{i}]'] for i in range(p)], axis=-1)
    state = c.states[:, :, -1].copy()
    row = c.compiled.outputs[name][-1]
    values = []
    for i in range(p):
        values.append(np.einsum('cdi,i->cd', state, row))
        state = state @ c.compiled.transition.T
    final = np.roll(np.stack(values, axis=-1), (len(c.y)-1)%p, axis=-1)
    prefix = (kwargs.get('label')+' · ') if kwargs.get('label') else ''
    _band(ax, np.arange(p), initial, color='.5', interval=interval, label=prefix+'Initial cycle')
    _band(ax, np.arange(p), final, color=color, interval=interval, label=prefix+'Final cycle')
    ax.set_xticks(np.arange(p), phase_labels or [str(i+1) for i in range(p)])
    ax.set_ylabel('Seasonal component')


def risk_curve(c, ax, *, name, color, interval, thresholds=None, time=-1,
               tail='upper', label=None, **kwargs):
    if thresholds is None:
        raise ValueError('Specify a vector of thresholds.')
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or not np.all(np.isfinite(thresholds)):
        raise ValueError('thresholds must be a finite vector.')
    if isinstance(c, Predictive):
        values = np.stack([c.risk(x, channel=name, tail=tail)[:, time] for x in thresholds], axis=-1)
        axis = 0
    else:
        values = np.stack([c.risk(x, tail=tail)[:, :, time] for x in thresholds], axis=-1)
        axis = (0, 1)
    _band(ax, thresholds, values, color=color, interval=interval, axis=axis, label=label)
    ax.set(xlabel='Threshold', ylabel='Probability')


def observation_scale(c, ax, *, color, interval, phase_labels=None, **kwargs):
    if isinstance(c, Predictive):
        raise TypeError('Use a fitted channel for phase-specific observation scales.')
    p = c.model.period
    values = np.stack([c.parameters.get(f'sigma[{i}]', c.parameters['sigma']) for i in range(p)], axis=-1)
    from ..results import summarize
    s = summarize(values, interval)
    x = np.arange(p)
    ax.vlines(x, s['lower'], s['upper'], color=color)
    ax.plot(x, s['mean'], 'o', color=color)
    ax.set_xticks(x, phase_labels or [str(i+1) for i in range(p)], rotation=45 if p > 4 else 0)
    ax.set_ylabel('Observation scale')
