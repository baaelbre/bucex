"""Shared setup for small, explicitly configured SERRA experiments."""
from dataclasses import asdict, replace
from copy import deepcopy
import json
import numpy as np
import pandas as pd
import bucex as bx
from research.serra.models import channel, marginal_prior, inference_options
from research.serra.report import save_band


def configured_variant(config, variant):
    config = deepcopy(config)
    p, m = config['priors'], config['model']
    for key in ('innovation', 'xi_prior', 'xi_sd', 'xi_bounds', 'tg_spike_shape', 'tg_tail_shape',
                'observation_variance'):
        if key in variant:
            p[key] = variant[key]
    for key in ('seasonal_scale', 'scale_mode', 'scale_prior_sd', 'scale_slope_sd',
                'scale_innovation_sd', 'level', 'trend', 'seasonal'):
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
    directory.mkdir(parents=True, exist_ok=True)
    if config.get("save_fits", True):
        fit.save(directory / "fit.bucex")
    bx.save_config(config, directory / "config.json")
    (directory / "declared_priors.json").write_text(json.dumps(asdict(prior),
        default=lambda value: value.tolist(), indent=2) + "\n")
    diagnostic = fit.diagnostics()
    bx.save_config(dict(bucex_version=bx.__version__, model=fit.model.to_dict(),
        inference=fit.plan.to_dict(), warnings=diagnostic["warnings"]), directory / "run.json")
    level = config.get("credible_interval", .90)
    diagnostic["parameters"].to_csv(directory / "mcmc.csv")
    (directory / "engine.json").write_text(json.dumps(
        {key: float(value) if np.isfinite(value) else None for key, value in diagnostic["engine"].items()}, indent=2))
    comparison = bx.compare_innovation_priors(fit, prior,
        size=config.get("prior_draws", 2000), seed=config["seed"], level=level)
    comparison.to_csv(directory / "prior_posterior.csv", index=False)
    scientific = bx.scientific_summary(fit, threshold, event_index=event_index, level=level)
    scientific.to_csv(directory / "scientific_targets.csv")
    prior_targets = bx.prior_predictive_targets(fit.model, prior, fit.n_time, threshold,
        tail="lower" if fit.transform_sign < 0 else "upper", size=config.get("prior_predictive_draws", 100),
        seed=config["seed"], event_index=event_index, level=level)
    pd.concat([prior_targets, scientific.assign(distribution="posterior")]).to_csv(
        directory / "prior_posterior_targets.csv")
    if config.get("figures", True):
        save_band(fit, fit.component_draws("level"), directory / "level", level=level)
        save_band(fit, fit.exceedance_probability_draws(threshold, return_labels=False),
                  directory / "risk", ylabel=fit.event_label(threshold), level=level)
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 3, figsize=(10, 3))
        for axis, component in zip(axes, ("level", "slope", "seasonal")):
            rows = comparison[(comparison.component == component) & (comparison.scale == "SD")]
            for j, (_, row) in enumerate(rows.iterrows()):
                axis.errorbar(row["median"], j, xerr=[[row["median"]-row["lower"]], [row["upper"]-row["median"]]], fmt="o")
            axis.tick_params(labelsize=11)
            axis.xaxis.label.set_size(12)
            axis.set(yticks=[0, 1], yticklabels=["prior", "posterior"], xlabel=f"{component} innovation SD")
        figure.tight_layout(); figure.savefig(directory / "prior_posterior.pdf"); plt.close(figure)
    return scientific
