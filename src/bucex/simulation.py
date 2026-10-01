"""Joint prior simulations on scientifically meaningful time horizons."""
from dataclasses import dataclass
import numpy as np
from .models import Model, Channel, MultiSeriesModel
from .priors import Fixed
from .distributions import contrasts
from .prediction import Predictive, _observation_draws
from .inference._model import physical_system
from .inference._state import seasonal_state_from_phase_effects, seasonal_rotation_matrix


def _normal(prior, draws, rng):
    return np.full(draws, prior.value) if isinstance(prior, Fixed) else rng.normal(prior.mean, prior.sd, draws)


def _normalize(model):
    return MultiSeriesModel((Channel('y', model),)) if isinstance(model, Model) else model


def _parameters(model, draws, rng):
    shared = {name: np.abs(rng.normal(0, getattr(model.pooling, name).sd, draws))
              for name in ('level', 'slope', 'seasonal') if model.pooling and getattr(model.pooling, name)}
    result = {}
    for c in model.channels:
        p = c.model.priors
        values = dict(initial_level=_normal(p.initial_level, draws, rng),
                      initial_slope=_normal(p.initial_slope, draws, rng))
        if c.model.trend.trend_mode == 'off':
            values['initial_slope'][:] = 0
        for component in ('level', 'slope', 'seasonal'):
            if component not in c.model.active:
                amplitude = np.zeros(draws)
            elif component in shared:
                amplitude = rng.normal(size=draws)*shared[component]
            else:
                amplitude = _normal(getattr(p, component), draws, rng)
            values['variance.'+component] = amplitude**2
        variance = np.full(draws, p.variance.value) if isinstance(p.variance, Fixed) else p.variance.scale/rng.gamma(p.variance.shape, size=draws)
        values['sigma'] = np.sqrt(variance)
        if c.model.observation.name == 'gev':
            values['xi'] = _normal(p.shape, draws, rng)
        if c.model.season:
            u = np.stack([_normal(p.initial_seasonal, draws, rng) for _ in range(c.model.period-1)], axis=1)
            effects = u@contrasts(c.model.period).T
            for i in range(c.model.period):
                values[f'initial_seasonal[{i}]'] = effects[:, i]
        if c.model.observation.scale:
            effects = rng.normal(0, c.model.observation.scale.prior_sd, (draws, c.model.period-1))@contrasts(c.model.period).T
            for i in range(c.model.period):
                values[f'sigma[{i}]'] = values['sigma']*np.exp(effects[:, i])
        result[c.name] = values
    return result, shared


def prior_samples(model, *, draws=10000, seed=3000):
    """Scalar physical parameters and hyperparameters, without simulating paths."""
    if type(draws) is not int or draws < 1:
        raise ValueError('draws must be a positive integer.')
    model = _normalize(model)
    parameters, shared = _parameters(model, draws, np.random.default_rng(seed))
    result = {f'{name}.{key}': v for name, ps in parameters.items() for key, v in ps.items()}
    result.update({f'tau.{name}': v for name, v in shared.items()})
    return result


@dataclass
class PriorPredictive(Predictive):
    parameters: dict
    shared_scales: dict


def prior_predictive(model, steps, *, draws=1000, seed=3000, steps_per_year=None):
    """Simulate the full prior, including initial states and shared scales.

    For change calibration subtract the initial level/rate, rather than
    confusing uncertainty in the starting temperature with future evolution.
    """
    if type(steps) is not int or steps < 1 or type(draws) is not int or draws < 1:
        raise ValueError('steps and draws must be positive integers.')
    model = _normalize(model)
    rng = np.random.default_rng(seed)
    parameters, shared = _parameters(model, draws, rng)
    steps_per_year = model.channels[0].model.period if steps_per_year is None else steps_per_year
    result = PriorPredictive({c.name: c.model for c in model.channels}, np.arange(1, steps+1),
        {}, {}, {}, {}, {}, {}, {}, np.arange(draws), 'prior', steps_per_year, parameters, shared)
    for c in model.channels:
        values = parameters[c.name]
        G, layout = physical_system(c.model)
        state = np.zeros((draws, layout.centered_state_dim))
        state[:, 0] = values['initial_level']
        if layout.has_beta:
            state[:, 1] = values['initial_slope']
        if layout.season_dim:
            phase = np.stack([values[f'initial_seasonal[{i}]'] for i in range(c.model.period)], axis=1)
            first = np.column_stack([phase[:, 0], phase[:, :1:-1]])
            state[:, layout.season_slice] = np.linalg.solve(seasonal_rotation_matrix(layout.season_dim), first.T).T
        sd = np.zeros_like(state)
        sd[:, 0] = np.sqrt(values['variance.level'])
        if layout.has_beta:
            sd[:, 1] = np.sqrt(values['variance.slope'])
        if layout.season_dim:
            sd[:, layout.season_slice.start] = np.sqrt(values['variance.seasonal'])
        path = np.empty((draws, steps, layout.centered_state_dim))
        for t in range(steps):
            state = state@G.T+sd*rng.normal(size=state.shape)
            path[:, t] = state
        name = c.name
        result.level[name] = path[:, :, 0]
        result.slope[name] = path[:, :, 1] if layout.has_beta else np.zeros((draws, steps))
        result.seasonal[name] = path[:, :, layout.season_slice.start] if layout.season_dim else np.zeros((draws, steps))
        result.location[name] = result.level[name]+result.seasonal[name]
        if c.model.observation.scale:
            sigmas = np.stack([values[f'sigma[{i}]'] for i in range(c.model.period)], axis=1)
            result.sigma[name] = sigmas[:, np.arange(steps)%c.model.period]
        else:
            result.sigma[name] = np.broadcast_to(values['sigma'][:, None], (draws, steps)).copy()
        result.xi[name] = np.broadcast_to(values.get('xi', np.zeros(draws))[:, None], (draws, steps)).copy()
        result.y[name] = _observation_draws(c.model.observation, result.location[name], result.sigma[name], result.xi[name], rng)
    return result


def simulate(model, steps, *, seed=3001, steps_per_year=None):
    """Draw one complete model realization; use Fixed priors for known parameters."""
    return prior_predictive(model, steps, draws=1, seed=seed, steps_per_year=steps_per_year)
