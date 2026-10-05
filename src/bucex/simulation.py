"""Joint prior parameter draws and simulations through the public component compiler."""
from dataclasses import dataclass
import numpy as np
from .models import Model, Channel, MultiSeriesModel, compile_model
from .models.compiler import placeholder_exog
from .priors import Fixed
from .distributions import contrasts
from .prediction import Predictive, _observation_draws
from .fitting import prepare_exog


def _normal(prior, draws, rng):
    return np.full(draws, prior.value) if isinstance(prior, Fixed) else rng.normal(prior.mean, prior.sd, draws)


def _normalize(model):
    return MultiSeriesModel((Channel('y', model),)) if isinstance(model, Model) else model


def _parameters(model, draws, rng):
    shared = {name: np.abs(rng.normal(0, prior.sd, draws)) for name, prior in (model.pooling.scales if model.pooling else {}).items()}
    result = {}
    from .inference.plan import plan
    for c in model.channels:
        plan(c.model)
        b = compile_model(c.model, 1, placeholder_exog(c.model, 1))
        theta = np.column_stack([_normal(p, draws, rng) for p in b.priors])
        for g in b.groups:
            if g.name in shared:
                theta[:, g.coefficient] = rng.normal(size=draws)*shared[g.name]
        values = {name: theta[:, i] for i, name in enumerate(b.coefficient_names)}
        for g in b.groups:
            values['variance.'+g.name] = theta[:, g.coefficient]**2
        for name, row in b.initial_outputs.items():
            values[name] = theta[:, :b.initial_count] @ row
        for name in ('level', 'slope', 'seasonal'):
            values.setdefault('variance.'+name, np.zeros(draws))
        values.setdefault('initial_slope', np.zeros(draws))
        p = c.model.priors
        variance = np.full(draws, p.variance.value) if isinstance(p.variance, Fixed) else p.variance.scale/rng.gamma(p.variance.shape, size=draws)
        values['sigma'] = np.sqrt(variance)
        if c.model.observation.name == 'gev':
            values['xi'] = _normal(p.shape, draws, rng)
        if c.model.observation.scale:
            effects = rng.normal(0, c.model.observation.scale.prior_sd, (draws, c.model.period-1)) @ contrasts(c.model.period).T
            for i in range(c.model.period):
                values[f'sigma[{i}]'] = values['sigma']*np.exp(effects[:, i])
        result[c.name] = values
    return result, shared


def prior_samples(model, *, draws=10000, seed=3000):
    """Scalar parameters and shared scales, including derived innovation variances."""
    if type(draws) is not int or draws < 1:
        raise ValueError('draws must be a positive integer.')
    parameters, shared = _parameters(_normalize(model), draws, np.random.default_rng(seed))
    result = {f'{name}.{key}': v for name, ps in parameters.items() for key, v in ps.items()}
    result.update({f'tau.{name}': v for name, v in shared.items()})
    return result


@dataclass
class PriorPredictive(Predictive):
    parameters: dict
    shared_scales: dict


def prior_predictive(model, steps, *, draws=1000, seed=3000, steps_per_year=None, exog=None):
    if type(steps) is not int or steps < 1 or type(draws) is not int or draws < 1:
        raise ValueError('steps and draws must be positive integers.')
    model = _normalize(model)
    rng = np.random.default_rng(seed)
    exog = prepare_exog(model, steps, exog)
    parameters, shared = _parameters(model, draws, rng)
    steps_per_year = model.channels[0].model.period if steps_per_year is None else steps_per_year
    if not np.isfinite(steps_per_year) or steps_per_year <= 0:
        raise ValueError('steps_per_year must be positive and finite.')
    result = PriorPredictive({c.name: c.model for c in model.channels}, np.arange(1, steps+1),
        {}, {}, {}, {}, {}, {}, {}, np.arange(draws), 'prior', steps_per_year, parameters, shared)
    for c in model.channels:
        values = parameters[c.name]
        b = compile_model(c.model, steps, exog[c.name])
        theta = np.column_stack([values[key] for key in b.coefficient_names])
        state = theta[:, :b.initial_count] @ b.initial.T
        path = np.empty((draws, steps, b.state_dim))
        for t in range(steps):
            state = state @ b.transition.T
            for g in b.groups:
                state += theta[:, g.coefficient, None]*(rng.normal(size=(draws, g.loading.shape[1])) @ g.loading.T)
            path[:, t] = state
        name = c.name
        outputs = b.paths(path)
        result.components[name] = outputs
        for key in ('level', 'slope', 'seasonal', 'location'):
            getattr(result, key)[name] = outputs.get(key, np.zeros((draws, steps)))
        if c.model.observation.scale:
            sigmas = np.stack([values[f'sigma[{i}]'] for i in range(c.model.period)], axis=1)
            result.sigma[name] = sigmas[:, np.arange(steps)%c.model.period]
        else:
            result.sigma[name] = np.broadcast_to(values['sigma'][:, None], (draws, steps)).copy()
        result.xi[name] = np.broadcast_to(values.get('xi', np.zeros(draws))[:, None], (draws, steps)).copy()
        result.y[name] = _observation_draws(c.model.observation, result.location[name], result.sigma[name], result.xi[name], rng)
    return result


def simulate(model, steps, *, seed=3001, steps_per_year=None, exog=None):
    return prior_predictive(model, steps, draws=1, seed=seed, steps_per_year=steps_per_year, exog=exog)
