"""Small, strict JSON configuration layer for the paper, not package internals."""
from copy import deepcopy
import json
from pathlib import Path
import bucex as bx
from .data import ROOT, SERIES

CONFIG = ROOT/'research/config'


def merge(base, change):
    result = deepcopy(base)
    for key, value in change.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def read_config(path, _seen=()):
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError('Circular JSON extends chain.')
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError('Configuration must be a JSON object.')
    parent = values.pop('extends', None)
    return merge(read_config(path.parent/parent, (*_seen, path)), values) if parent else values


def _keys(value, allowed, label):
    if not isinstance(value, dict) or set(value)-set(allowed):
        raise ValueError(f'Unknown or invalid {label} configuration: {set(value)-set(allowed) if isinstance(value, dict) else value}')


def validate(c):
    _keys(c, ('name', 'description', 'data', 'model', 'priors', 'pooling', 'profiles', 'laplace',
              'forecast_years', 'risk_thresholds', 'return_years', 'validation', 'plot', 'seed'), 'top-level')
    _keys(c['data'], ('source', 'frequency', 'series', 'start', 'end'), 'data')
    _keys(c['model'], ('period', 'level', 'slope', 'seasonal', 'seasonal_scale', 'scale_prior_sd', 'steps_per_year'), 'model')
    _keys(c['priors'], ('initial_level_sd', 'initial_slope_sd', 'initial_seasonal_sd', 'level_sd', 'slope_sd', 'seasonal_sd', 'variance_shape', 'variance_scale', 'xi_sd'), 'priors')
    _keys(c['profiles'], ('smoke', 'screen', 'paper'), 'profiles')
    for name, p in c['profiles'].items():
        _keys(p, ('draws', 'warmup', 'chains', 'workers', 'predictive_draws', 'prior_draws', 'check_draws', 'max_blocks'), 'profile '+name)
    _keys(c['laplace'], ('max_iterations', 'tolerance', 'mh_steps', 'curvature_floor', 'maximum_variance'), 'laplace')
    _keys(c['validation'], ('origins', 'recent_origins', 'years', 'intervals'), 'validation')
    _keys(c['plot'], ('format', 'dpi', 'interval'), 'plot')
    if c['pooling'] not in {'shared', 'private'}:
        raise ValueError('pooling must be shared or private (fixed Normal SDs).')
    if any(s not in SERIES for s in c['data']['series']) or len(set(c['data']['series'])) != len(c['data']['series']):
        raise ValueError('Unknown or duplicate temperature summary.')
    if c['model']['period'] != (4 if c['data']['frequency'] == 'seasonal' else 12):
        raise ValueError('The paper calendar and model period must agree.')
    if c['model']['steps_per_year'] != c['model']['period']:
        raise ValueError('Paper steps_per_year must match the aggregation frequency.')
    for s, r in c['risk_thresholds'].items():
        _keys(r, ('threshold', 'tail', 'month'), 'risk '+s)
    if c['forecast_years'] <= 0 or c['validation']['years'] <= 0:
        raise ValueError('Forecast and validation horizons must be positive.')
    return c


def load_config(path=None):
    return validate(read_config(CONFIG/'main.json' if path is None else path))


def build_model(c):
    validate(c)
    p, m = c['priors'], c['model']
    priors = bx.Priors(initial_level=bx.Normal(0, p['initial_level_sd']),
        initial_slope=bx.Normal(0, p['initial_slope_sd']), initial_seasonal=bx.Normal(0, p['initial_seasonal_sd']),
        level=bx.Normal(0, p['level_sd']), slope=bx.Normal(0, p['slope_sd']), seasonal=bx.Normal(0, p['seasonal_sd']),
        variance=bx.InverseGamma(p['variance_shape'], p['variance_scale']), shape=bx.Normal(0, p['xi_sd']))
    components = (bx.LocalLinearTrend(m['level'], m['slope']), bx.DummySeasonal(m['period'], m['seasonal']))
    scale = bx.SeasonalScale(m['period'], m['scale_prior_sd']) if m['seasonal_scale'] else None
    channels = []
    for name in c['data']['series']:
        obs = bx.Gaussian(scale) if name.endswith('m') else bx.GEV('lower' if name.endswith('n') else 'upper', scale)
        channels.append(bx.Channel(name, bx.Model(obs, components, priors)))
    pooling = None
    if c['pooling'] == 'shared':
        shared = {comp: bx.HalfNormal(p[comp+'_sd']) for comp, key in
            [('level', 'level'), ('slope', 'slope'), ('seasonal', 'seasonal')] if m[key] == 'dynamic'}
        pooling = bx.Pooling(**shared) if shared else None
    return bx.MultiSeriesModel(channels, pooling)
