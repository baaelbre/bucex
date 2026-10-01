"""Posterior replications and forecasts with full state and parameter uncertainty."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .inference._model import physical_system, layout_for
from .results import summarize


@dataclass
class Predictive:
    models: dict
    index: object
    y: dict
    location: dict
    level: dict
    slope: dict
    seasonal: dict
    sigma: dict
    xi: dict
    draw_indices: np.ndarray
    kind: str
    steps_per_year: float

    def channel_name(self, name=None):
        if name is None:
            if len(self.models) != 1:
                raise ValueError('Choose a channel for a multiseries prediction.')
            return next(iter(self.models))
        if name not in self.models:
            raise KeyError(name)
        return name

    def summary(self, channel=None, *, component='y', interval=.95):
        channel = self.channel_name(channel)
        if component not in {'y', 'location', 'level', 'slope', 'seasonal', 'sigma'}:
            raise ValueError('Unknown predictive component.')
        return summarize(getattr(self, component)[channel], interval=interval, axis=0)

    def risk(self, threshold, *, channel=None, tail='upper'):
        channel = self.channel_name(channel)
        if tail not in {'upper', 'lower'}:
            raise ValueError('tail must be upper or lower.')
        obs = self.models[channel].observation
        fn = obs.sf if tail == 'upper' else obs.cdf
        return fn(threshold, self.location[channel], {'sigma': self.sigma[channel], 'xi': self.xi[channel]})

    def return_level(self, years, *, channel=None, tail='upper'):
        channel = self.channel_name(channel)
        if not np.isfinite(years) or years <= 1 or tail not in {'upper', 'lower'}:
            raise ValueError('years must exceed one and tail must be upper or lower.')
        p = 1-1/years if tail == 'upper' else 1/years
        return self.models[channel].observation.ppf(p, self.location[channel],
            {'sigma': self.sigma[channel], 'xi': self.xi[channel]})

    def plot(self, type='forecast', **kwargs):
        from .plotting import plot
        return plot(self, type=type, **kwargs)


def _indices(fit, draws, seed):
    if type(draws) is not int or draws < 1:
        raise ValueError('draws must be a positive integer.')
    rng = np.random.default_rng(seed)
    return rng, rng.integers(fit.chains*fit.draws, size=draws)


def _take(values, ids):
    return values.reshape((-1,)+values.shape[2:])[ids]


def _observation_draws(obs, location, sigma, xi, rng):
    u = rng.uniform(np.nextafter(0., 1.), np.nextafter(1., 0.), size=location.shape)
    y = obs.ppf(u, location, {'sigma': sigma, 'xi': xi})
    if not np.all(np.isfinite(y)):
        raise FloatingPointError('Predictive draws overflowed; inspect tail priors and parameters.')
    return y


def _future_index(index, horizon, dates):
    if dates is not None:
        future = pd.DatetimeIndex(dates)
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError('Supply training dates when using future dates.')
        expected = pd.date_range(index[-1], periods=horizon+1, freq=pd.infer_freq(index))[1:]
        if not future.equals(expected):
            raise ValueError('Future dates must continue the fitted regular calendar exactly.')
        return future
    if isinstance(index, pd.DatetimeIndex):
        return pd.date_range(index[-1], periods=horizon+1, freq=pd.infer_freq(index))[1:]
    return np.arange(len(index), len(index)+horizon)


def predict(fit, horizon, *, draws=2000, seed=2000, dates=None):
    """Forecast horizon observation steps, retaining one posterior draw per path."""
    if type(horizon) is not int or horizon < 1:
        raise ValueError('horizon must be a positive integer number of observation steps.')
    rng, ids = _indices(fit, draws, seed)
    first = next(iter(fit.channels.values()))
    result = Predictive({k: c.model for k, c in fit.channels.items()}, _future_index(first.index, horizon, dates),
        {}, {}, {}, {}, {}, {}, {}, ids, 'forecast', first.steps_per_year)
    for name, c in fit.channels.items():
        transition, layout = physical_system(c.model)
        state = _take(c.states[:, :, -1], ids).copy()
        sd = np.zeros((draws, layout.centered_state_dim))
        positions = {'level': 0, 'slope': layout.idx_beta,
                     'seasonal': layout.season_slice.start if layout.season_dim else None}
        for component in c.model.active:
            sd[:, positions[component]] = np.sqrt(_take(c.parameters['variance.'+component], ids))
        paths = np.empty((draws, horizon, layout.centered_state_dim))
        for t in range(horizon):
            state = state @ transition.T + sd*rng.normal(size=state.shape)
            paths[:, t] = state
        result.level[name] = paths[:, :, 0]
        result.slope[name] = paths[:, :, 1] if layout.has_beta else np.zeros((draws, horizon))
        result.seasonal[name] = paths[:, :, layout.season_slice.start] if layout.season_dim else np.zeros((draws, horizon))
        result.location[name] = result.level[name]+result.seasonal[name]
        if c.model.observation.scale:
            phase = (len(c.y)+np.arange(horizon)) % c.model.period
            sigmas = np.stack([_take(c.parameters[f'sigma[{i}]'], ids) for i in range(c.model.period)], axis=1)
            result.sigma[name] = sigmas[:, phase]
        else:
            result.sigma[name] = np.broadcast_to(_take(c.parameters['sigma'], ids)[:, None], (draws, horizon)).copy()
        result.xi[name] = np.broadcast_to(_take(c.parameters.get('xi', np.zeros((fit.chains, fit.draws))), ids)[:, None], (draws, horizon)).copy()
        result.y[name] = _observation_draws(c.model.observation, result.location[name], result.sigma[name], result.xi[name], rng)
    return result


def replicate(fit, *, draws=2000, seed=2001):
    """Conditional posterior replications, not out-of-sample forecasts."""
    rng, ids = _indices(fit, draws, seed)
    first = next(iter(fit.channels.values()))
    result = Predictive({k: c.model for k, c in fit.channels.items()}, first.index,
        {}, {}, {}, {}, {}, {}, {}, ids, 'replication', first.steps_per_year)
    for name, c in fit.channels.items():
        for key in ('location', 'level', 'slope', 'seasonal'):
            getattr(result, key)[name] = _take(c.path(key), ids)
        result.sigma[name] = _take(c.sigma(), ids)
        result.xi[name] = np.broadcast_to(_take(c.parameters.get('xi', np.zeros((fit.chains, fit.draws))), ids)[:, None], result.location[name].shape).copy()
        result.y[name] = _observation_draws(c.model.observation, result.location[name], result.sigma[name], result.xi[name], rng)
    return result


def coverage(prediction, observed, *, levels=(.90, .95, .99)):
    """Held-out coverage, width and direction of misses for central intervals."""
    rows = []
    if isinstance(observed, pd.DataFrame):
        if isinstance(prediction.index, pd.DatetimeIndex) and not observed.index.equals(prediction.index):
            raise ValueError('Held-out dates must match forecast dates exactly.')
    if not isinstance(observed, (dict, pd.DataFrame)):
        observed = {prediction.channel_name(): np.asarray(observed)}
    for name in prediction.models:
        y = np.asarray(observed[name], dtype=float)
        if y.shape != (len(prediction.index),) or not np.all(np.isfinite(y)):
            raise ValueError('Provide finite held-out observations for every forecast step.')
        for level in levels:
            s = prediction.summary(name, interval=level)
            rows.append(dict(channel=name, interval=level, n=len(y),
                coverage=float(np.mean((y >= s['lower']) & (y <= s['upper']))),
                mean_width=float(np.mean(s['upper']-s['lower'])),
                below=int(np.sum(y < s['lower'])), above=int(np.sum(y > s['upper']))))
    return pd.DataFrame(rows)
