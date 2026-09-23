"""Scientific declarations for the paper, using only the public BUCEX API."""
import numpy as np
import bucex as bx


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
    if p.get('shared_shrinkage') is not None and config['analysis'] == 'independent':
        raise ValueError('Shared shrinkage requires a joint fit of the series; use analysis=joint or copula.')
    center = p.get('initial_level_mean', 0.)
    if isinstance(center, dict):
        center = center[item.name]
    bounds = bx.GEV(xi_bounds=p.get('xi_bounds')).xi_bounds
    xi = (bx.UniformPrior(*bounds) if p.get('xi_prior', 'normal') == 'uniform'
          else bx.NormalPrior(p.get('xi_mean', 0.), p.get('xi_sd', .3)))
    return bx.fs_priors(item.family, period=config['model']['period'],
        innovation=p.get('innovation', 'normal'), innovation_median=p['innovation_median'],
        initial_level=bx.NormalPrior(item.transform_sign*center, p.get('baseline_sd', 20.)),
        initial_slope=bx.NormalPrior(0., p['initial_slope_sd']),
        seasonal_initial_sd=p['seasonal_initial_sd'],
        observation_variance=bx.InverseGammaPrior(*p['observation_variance']),
        xi_prior=xi, xi_max_abs=max(abs(v) for v in bounds),
        spike_shape=p.get('tg_spike_shape', .5), tail_shape=p.get('tg_tail_shape', .5))


def joint_model(data, config):
    """The same private marginal specifications with an optional joint copula."""
    mode = config['analysis']
    if mode not in {'joint', 'copula'}:
        raise ValueError("Joint analysis is 'joint' (R=I) or 'copula'.")
    channels = tuple(channel(name, data, config) for name in data)
    settings = dict(config.get('copula', {'eta': 1.}))
    structure = settings.pop('structure', 'constant')
    if mode == 'joint':
        copula = bx.GaussianCopula(correlation=np.eye(len(channels)))
    elif structure == 'constant':
        copula = bx.GaussianCopula(eta=settings.get('eta', 1.))
    else:
        copula = bx.SeasonalGaussianCopula(structure=structure,
            period=config['model']['period'], **settings)
    model = bx.MultiSeriesModel(channels, copula=copula)
    settings = config['priors'].get('shared_shrinkage')
    hierarchy = None
    if settings is not None:
        aliases = {'level': 'level', 'slope': 'trend', 'seasonal': 'season'}
        hierarchy = bx.SharedShrinkage(
            medians={c: config['priors']['innovation_median'][aliases[c]]
                     for c in settings.get('components', ['level', 'slope', 'seasonal'])},
            log_sd=settings.get('log_sd', np.log(2.)),
            initial_slope_sd=(config['priors']['initial_slope_sd']
                              if settings.get('pool_initial_slope', True) else None))
    return model, bx.MarginalPriors(
        {c.name: marginal_prior(c, data, config) for c in channels}, shrinkage=hierarchy)


def inference_options(config):
    return {'laplace': bx.Laplace(**config.get('inference', {}).get('laplace', {}))}


def fit_options(config, *, family='mixed', tail=None):
    engine = 'ffbs' if family=='gaussian' else 'laplace_mh' if family in {'gev','mixed'} else 'auto'
    options = dict(engine=engine,
        parameterization='fs', asis=config.get('inference', {}).get('asis', False),
        mcmc=bx.MCMC(**config['mcmc']), **inference_options(config))
    if tail is not None:
        options['tail'] = tail
    if config['analysis'] == 'independent' and family == 'gev':
        options['init'] = {'xi': 0.}
    return options
