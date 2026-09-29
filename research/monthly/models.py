"""Scientific declarations for the paper, using only the public BUCEX API."""
import numpy as np
import bucex as bx


def shrinkage_specification(config, *, independent=False):
    """One construction for matched private and shared scale priors."""
    p = config['priors']
    key = 'independent_shrinkage' if independent else 'shared_shrinkage'
    settings = p.get(key)
    if settings is None:
        return None
    cls = bx.IndependentShrinkage if independent else bx.SharedShrinkage
    aliases = {'level': 'level', 'slope': 'trend', 'seasonal': 'season'}
    components = settings.get('components', list(aliases))
    components = [c for c in components if config['model'].get(aliases[c] if c != 'seasonal' else 'seasonal', 'dynamic') == 'dynamic']
    if not components:
        return None
    if settings.get('scale_parameterization') == 'normal_sd':
        if settings.get('pool_initial_slope', False):
            raise ValueError('The direct-SD research hierarchy leaves initial rates separate.')
        return cls.from_sd({c: p['innovation_sd'][aliases[c]] for c in components},
                           log_sd=settings.get('log_sd', np.log(3.)),
                           hyperprior=settings.get('hyperprior','lognormal'), df=settings.get('df',4.))
    return cls(medians={c: p['innovation_median'][aliases[c]] for c in components},
        log_sd=settings.get('log_sd', np.log(2.)),
        initial_slope_sd=(p['initial_slope_sd'] if not independent and
                         settings.get('pool_initial_slope', True) and p.get('initial_slope_median') is None else None),
        initial_slope_median=(p.get('initial_slope_median') if not independent and
                             settings.get('pool_initial_slope', True) else None))


def check_scale_convention(p):
    for key in ('shared_shrinkage', 'independent_shrinkage'):
        if p.get('innovation_sd') is not None and p.get(key) is not None:
            if p[key].get('scale_parameterization') != 'normal_sd':
                raise ValueError('Explicit innovation_sd needs normal_sd parameterization with a shrinkage hyperprior.')


def channel(name, data, config):
    """Private location components and a separately declared observation scale."""
    info, settings = bx.UCCLE_INFO[name], config['model']
    seasonal = (bx.SeasonalScale(settings['period'], settings.get('scale_prior_sd', .3),
                    calendar='meteorological' if config['data'].get('frequency')=='seasonal' else None)
                if settings.get('seasonal_scale', False) else None)
    scale = bx.LogScale(seasonal=seasonal)
    obs = (bx.Gaussian(scale=scale) if info['family'] == 'gaussian' else
           bx.GEV(scale=scale, xi_bounds=config['priors'].get('xi_bounds')))
    components = (bx.LocalLinearTrend(level_mode=settings.get('level', 'dynamic'),
                    trend_mode=settings.get('trend', 'dynamic')),
                  bx.DummySeasonal(settings['period'], mode=settings.get('seasonal', 'dynamic')))
    # Both univariate and joint fits use the same named parameter declaration.
    if seasonal is None:
        obs = bx.Gaussian() if info['family'] == 'gaussian' else bx.GEV(xi_bounds=config['priors'].get('xi_bounds'))
        parameters = {'mu': bx.Latent(components), 'sigma': bx.Constant()}
    else:
        parameters = {'mu': bx.Latent(components)}
    return bx.Channel(name, obs, parameters=parameters, tail=info['tail'])


def marginal_prior(item, data, config):
    """Proper continuous priors; no estimated/data-centred hyperparameters."""
    p = config['priors']
    check_scale_convention(p)
    if p.get('shared_shrinkage') is not None and config['analysis'] == 'independent':
        raise ValueError('Shared shrinkage requires a joint fit of the series; use analysis=joint or copula.')
    center = p.get('initial_level_mean', 0.)
    if isinstance(center, dict):
        center = center[item.name]
    bounds = bx.GEV(xi_bounds=p.get('xi_bounds')).xi_bounds
    xi = (bx.UniformPrior(*bounds) if p.get('xi_prior', 'normal') == 'uniform'
          else bx.NormalPrior(p.get('xi_mean', 0.), p.get('xi_sd', .3)))
    return bx.fs_priors(item.family, period=config['model']['period'],
        innovation=p.get('innovation', 'normal'), innovation_median=p.get('innovation_median'),
        innovation_sd=p.get('innovation_sd'),
        initial_level=bx.NormalPrior(item.transform_sign*center, p.get('baseline_sd', 20.)),
        initial_slope=bx.NormalPrior(0., p['initial_slope_sd']),
        seasonal_initial_sd=p['seasonal_initial_sd'],
        seasonal_initial_basis=p.get('seasonal_initial_basis','lags'),
        observation_variance=bx.InverseGammaPrior(*p['observation_variance']),
        xi_prior=xi, xi_max_abs=max(abs(v) for v in bounds),
        spike_shape=p.get('tg_spike_shape', .5), tail_shape=p.get('tg_tail_shape', .5))


def joint_model(data, config):
    """The same private marginal specifications with an optional joint copula."""
    if config['priors'].get('independent_shrinkage') is not None:
        raise ValueError('A private hierarchy needs independent_model and a single response.')
    mode = config['analysis']
    if mode not in {'joint', 'copula'}:
        raise ValueError("Joint analysis is 'joint' (R=I) or 'copula'.")
    channels = tuple(channel(name, data, config) for name in data)
    settings = dict(config.get('copula') or {'eta': 1.})
    structure = settings.pop('structure', 'constant')
    if mode == 'joint':
        copula = None if config.get('copula') is None else bx.GaussianCopula(correlation=np.eye(len(channels)))
    elif structure == 'constant':
        copula = bx.GaussianCopula(eta=settings.get('eta', 1.))
    else:
        copula = bx.SeasonalGaussianCopula(structure=structure,
            period=config['model']['period'], **settings)
    model = bx.MultiSeriesModel(channels, copula=copula)
    hierarchy = shrinkage_specification(config)
    return model, bx.MarginalPriors(
        {c.name: marginal_prior(c, data, config) for c in channels}, shrinkage=hierarchy)


def independent_model(data, config):
    """One response, its own hyperparameters, and no copula of any kind."""
    if config['analysis'] != 'independent' or len(data.columns) != 1:
        raise ValueError('Independent analyses require exactly one response per fit.')
    p = config['priors']
    check_scale_convention(p)
    if p.get('shared_shrinkage') is not None or config.get('copula') is not None:
        raise ValueError('Independent analyses cannot have shared shrinkage or a copula.')
    item = channel(data.columns[0], data, config)
    hierarchy = shrinkage_specification(config, independent=True)
    return (bx.MultiSeriesModel((item,), copula=None),
            bx.MarginalPriors({item.name: marginal_prior(item, data, config)}, shrinkage=hierarchy))


def inference_options(config):
    return {'laplace': bx.Laplace(**config.get('inference', {}).get('laplace', {}))}


def fit_options(config, *, family='mixed', tail=None):
    engine = 'ffbs' if family=='gaussian' else 'laplace_mh' if family in {'gev','mixed'} else 'auto'
    options = dict(engine=engine,
        parameterization='fs', asis=config.get('inference', {}).get('asis', False),
        mcmc=bx.MCMC(**config['mcmc']), **inference_options(config))
    if tail is not None:
        options['tail'] = tail
    if config['analysis'] == 'independent' and family == 'gev' and config['priors'].get('independent_shrinkage') is None and config['priors'].get('innovation_sd') is None:
        options['init'] = {'xi': 0.}
    return options
