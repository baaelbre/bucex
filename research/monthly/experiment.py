"""Shared setup for small, explicitly configured SERRA experiments."""
from dataclasses import asdict, replace
from copy import deepcopy
import json
import numpy as np
import pandas as pd
import bucex as bx
from research.monthly.models import channel, marginal_prior, inference_options
from research.monthly.report import save_band


def configured_variant(config, variant):
    config = deepcopy(config)
    p, m = config['priors'], config['model']
    old_hierarchy = p.get('shared_shrinkage')
    for key in ('innovation', 'xi_prior', 'xi_sd', 'xi_bounds', 'tg_spike_shape', 'tg_tail_shape',
                'observation_variance', 'shared_shrinkage'):
        if key in variant:
            p[key] = variant[key]
    if variant.get('match_marginal_moments',False):
        new_hierarchy=p.get('shared_shrinkage')
        if old_hierarchy is None or new_hierarchy is None:
            raise ValueError('Moment-matched width sensitivity requires both shared hierarchies.')
        old_d=old_hierarchy.get('log_sd',np.log(2.));new_d=new_hierarchy.get('log_sd',np.log(2.))
        factor=np.exp(old_d**2-new_d**2)
        aliases={'level':'level','slope':'trend','seasonal':'season'}
        for component in new_hierarchy.get('components',['level','slope','seasonal']):
            p['innovation_median'][aliases[component]]*=factor
        if new_hierarchy.get('pool_initial_slope',True):
            p['initial_slope_sd']*=factor
    for key in ('seasonal_scale', 'scale_prior_sd', 'level', 'trend', 'seasonal'):
        if key in variant:
            m[key] = variant[key]
    for key,factor in variant.get('multipliers', {}).items():
        p['innovation_median'][key] *= factor
    p['initial_slope_sd'] *= variant.get('initial_slope_multiplier', 1.)
    p['seasonal_initial_sd'] *= variant.get('seasonal_initial_multiplier', 1.)
    if 'asis' in variant:
        config.setdefault('inference', {})['asis'] = variant['asis']
    if 'copula' in variant:
        config.setdefault('copula', {}).update(variant['copula'])
    if 'analysis' in variant:
        config['analysis'] = variant['analysis']
    return config


def prior_for(item, data, config, variant):
    return marginal_prior(item, data, configured_variant(config,variant))


def fit_case(data, name, config, variant, *, engine='laplace_mh'):
    local = configured_variant(config,variant)
    item = channel(name,data,local)
    model, prior = bx.Model(item.observation,item.components), marginal_prior(item,data,local)
    fit = bx.fit(data[name], model, tail=item.tail, priors=prior,
        engine='ffbs' if item.family == 'gaussian' else engine,
        parameterization='fs', mcmc=bx.MCMC(**local['mcmc']),
        asis=local.get('inference',{}).get('asis', False),
        init={'xi':0.} if item.family == 'gev' else None, **inference_options(local))
    return fit, prior


def save_case(fit, prior, directory, config, *, threshold, event_index=-1):
    with bx.publication_style(style=config.get('figure_style','manuscript'),
                              dpi=config.get('figure_dpi',180)):
        return _save_case(fit, prior, directory, config, threshold=threshold, event_index=event_index)


def _save_case(fit, prior, directory, config, *, threshold, event_index=-1):
    directory.mkdir(parents=True, exist_ok=True)
    if config.get("save_fits", True):
        fit.save(directory / "fit.bucex")
    bx.save_config(config, directory / "config.json")
    bx.save_config(asdict(prior), directory / "declared_priors.json")
    diagnostic = fit.diagnostics()
    bx.save_config(dict(bucex_version=bx.__version__, model=fit.model.to_dict(),
        inference=fit.plan.to_dict(), warnings=diagnostic["warnings"],
        execution=fit.sampler_diagnostics.get('execution'),
        fitted_start=str(fit.time[0]), fitted_end=str(fit.time[-1]), n_months=fit.n_time), directory / "run.json")
    level = config.get("credible_interval", .95)
    diagnostic["parameters"].to_csv(directory / "mcmc.csv")
    (directory / "engine.json").write_text(json.dumps(
        {key: float(value) if np.isfinite(value) else None for key, value in diagnostic["engine"].items()}, indent=2))
    comparison = bx.compare_innovation_priors(fit, prior,
        size=config.get("prior_draws", 2000), seed=config["seed"], level=level)
    comparison.to_csv(directory / "prior_posterior.csv", index=False)
    bx.innovation_prior_diagnostics(comparison).to_csv(directory / 'prior_updates.csv', index=False)
    scientific = bx.scientific_summary(fit, threshold, event_index=event_index, level=level)
    scientific.to_csv(directory / "scientific_targets.csv")
    if config.get('prior_predictive_draws', 100) > 0:
        prior_targets = bx.prior_predictive_targets(fit.model, prior, fit.n_time, threshold,
            tail="lower" if fit.transform_sign < 0 else "upper", size=config.get("prior_predictive_draws", 100),
            seed=config["seed"], event_index=event_index, level=level)
        pd.concat([prior_targets, scientific.assign(distribution="posterior")]).to_csv(
            directory / "prior_posterior_targets.csv")
    paths = {'level': fit.component_draws('level'),
             'slope': 120 * fit.component_draws('slope'),
             'risk': fit.exceedance_probability_draws(threshold, return_labels=False)}
    for label, values in paths.items():
        band = fit.posterior_summary(values, credible_interval=level)
        pd.DataFrame({'time': fit.time, **band}).to_csv(directory / (label+'.csv'), index=False)
    if config.get('trace_exports', True):
        bx.trace_frame(bx.parameter_trace_draws(fit)).to_csv(directory/'parameter_traces.csv.gz', index=False)
    if config.get("figures", True):
        options = dict(image_format=config.get('figure_format','png'), dpi=config.get('figure_dpi',180))
        save_band(fit, fit.component_draws("level"), directory / "level", level=level, **options)
        save_band(fit, fit.exceedance_probability_draws(threshold, return_labels=False),
                  directory / "risk", ylabel=fit.event_label(threshold), level=level, **options)
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 3, figsize=(10, 3))
        for axis, component in zip(axes, ("level", "slope", "seasonal")):
            rows = comparison[(comparison.component == component) & (comparison.scale == "SD")]
            for j, (_, row) in enumerate(rows.iterrows()):
                axis.errorbar(row["median"], j, xerr=[[row["median"]-row["lower"]], [row["upper"]-row["median"]]], fmt="o")
            axis.tick_params(labelsize=11)
            axis.xaxis.label.set_size(12)
            axis.set(yticks=[0, 1], yticklabels=["prior", "posterior"], xlabel=f"{component} innovation SD")
        figure.tight_layout()
        bx.save_figure(figure,directory/'prior_posterior',formats=(options['image_format'],),
                      dpi=options['dpi'],close=True)
    return scientific
