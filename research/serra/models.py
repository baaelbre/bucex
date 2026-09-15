"""The paper's scientific declarations, expressed through the public API."""
import numpy as np
import bucex as bx


def channel(name, data, config):
    """Each summary has its own location components and observation seasonality."""
    info, settings = bx.UCCLE_INFO[name], config['model']
    scale = (bx.SeasonalScale(settings['period'], settings.get('scale_prior_sd', .3))
             if settings.get('seasonal_scale', True) else None)
    observation = (bx.Gaussian(scale=scale) if info['family'] == 'gaussian' else
                   bx.GEV(phi=settings.get('phi', 'stationary'), scale=scale,
                          xi_bounds=tuple(config['priors']['xi_bounds'])))
    return bx.Channel(name, observation,
        (bx.LocalLinearTrend(), bx.DummySeasonal(settings['period'])), tail=info['tail'])


def marginal_prior(item, data, config):
    """Independent, explicit FS/SSVS priors; slope absence has probability .10."""
    p = config['priors']
    # This empirical centering is fixed using an initial reference prefix.
    reference = data[item.name].iloc[:p.get('reference_months', 120)]
    arguments = dict(period=config['model']['period'],
        alpha_mean=item.transform_sign * float(np.median(reference)),
        alpha_sd=p['baseline_sd'], beta_sd=p['initial_slope_sd'],
        seasonal_initial_sd=p['seasonal_initial_sd'],
        innovation_slab_sd=p['innovation_slab_sd'],
        level_dynamic_probability=p.get('level_dynamic_probability', .5),
        trend_probabilities=tuple(p.get('trend_probabilities', (.10, .45, .45))),
        season_probabilities=tuple(p.get('season_probabilities', (0., .5, .5))),
        sigma2_prior=bx.InverseGammaPrior(*p['observation_variance']))
    if item.family == 'gaussian':
        return bx.ssvs_gaussian_priors(**arguments)
    arguments.update(xi_prior=bx.UniformPrior(*p['xi_bounds']),
                     xi_max_abs=max(abs(v) for v in p['xi_bounds']))
    if 'phi' in p:
        phi = p['phi']
        arguments['phi_prior'] = bx.PhiPrior(linear=bx.NormalPrior(**phi['linear']),
            rw_variance=bx.InverseGammaPrior(**phi['rw_variance']),
            model_probabilities=phi['model_probabilities'])
    return bx.ssvs_gev_priors(**arguments)


def joint_model(data, config):
    """Same six priors/trajectories; only the residual likelihood changes."""
    mode = config['analysis']
    if mode not in {'joint', 'copula'}:
        raise ValueError("SERRA joint analysis is 'joint' (R=I) or 'copula'.")
    channels = tuple(channel(name, data, config) for name in data)
    copula = (bx.GaussianCopula(correlation=np.eye(len(channels))) if mode == 'joint'
              else bx.GaussianCopula(**config.get('copula', {'eta': 2.})))
    return (bx.MultiSeriesModel(channels, copula=copula),
            bx.MarginalPriors({item.name: marginal_prior(item, data, config) for item in channels}))


def inference_options(config):
    return {'laplace': bx.Laplace(**config.get('inference', {}).get('laplace', {}))}


def fit_options(config, *, family='mixed', tail=None):
    options = dict(engine='ffbs' if family == 'gaussian' else 'laplace_mh',
                   parameterization='fs', asis=False,
                   mcmc=bx.MCMC(**config['mcmc']), **inference_options(config))
    if tail is not None:
        options['tail'] = tail
    if config['analysis'] == 'independent' and family == 'gev':
        options['init'] = {'xi': 0.}
    return options
