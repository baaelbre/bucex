"""Small report writer; all scientific calculations use public bucex methods."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import bucex as bx


def new_run(root, name):
    """Create a fresh directory without overwriting an earlier analysis."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = Path(root) / f"{name}_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_band(fit, values, path, *, dates=None, ylabel="temperature / °C", level=0.90):
    """Export one scientific band with readable labels and no default title."""
    with plt.rc_context({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11}):
        return _save_band(fit, values, path, dates=dates, ylabel=ylabel, level=level)


def _save_band(fit, values, path, *, dates=None, ylabel="temperature / °C", level=0.90):
    """Export pointwise posterior summaries and a figure on the same scale."""
    dates = fit.time if dates is None else dates
    band = fit.posterior_summary(values, credible_interval=level)
    pd.DataFrame({"time": dates, **band}).to_csv(path.with_suffix(".csv"), index=False)
    figure, axis = plt.subplots(figsize=(8, 3))
    axis.fill_between(dates, band["lower"], band["upper"], alpha=0.2)
    axis.plot(dates, band["median"])
    axis.set(xlabel="time", ylabel=ylabel)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def scientific_targets(fit):
    """First-to-last observed changes, retaining chain and draw dimensions."""
    targets = {}
    for name in fit.channel_names if fit.is_multiseries_model else (None,):
        values = fit.component_draws("level", channel=name, combine_chains=False)
        targets[f"{name or 'series'}_level_change"] = values[..., -1] - values[..., 0]
    if fit.is_multiseries_model:
        for left, right in (("TXx", "TXm"), ("TNn", "TNm"), ("TXm", "TNm"), ("TXx", "TNx")):
            if {left, right} <= set(fit.channel_names):
                targets[f"{left}_minus_{right}_change"] = targets[f"{left}_level_change"] - targets[f"{right}_level_change"]
    for group in getattr(fit.model, "shared", ()):
        if not isinstance(group.component, (bx.LocalLevel, bx.LocalLinearTrend)):
            continue
        if isinstance(group, bx.Shared):
            values = fit.shared_draws(group.name, combine_chains=False)
            targets[f"{group.name}_change"] = values[..., -1] - values[..., 0]
        else:
            for name in fit.channel_names:
                values = fit.departure_draws(name, name=group.name, combine_chains=False)
                targets[f"{name}_{group.name}_change"] = values[..., -1] - values[..., 0]
    return targets


def write_report(fit, directory, *, config, risks=None, horizon=12, level=.95):
    """Export auditable numerical results; figures are optional, never evidence of convergence."""
    from dataclasses import asdict
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fit.save(directory/'fit.bucex')
    bx.save_config(config,directory/'config.json')
    (directory/'declared_priors.json').write_text(json.dumps(asdict(fit.priors),
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x,indent=2)+'\n')
    diagnostic = fit.diagnostics()
    diagnostic['parameters'].to_csv(directory/'mcmc.csv')
    pd.DataFrame.from_dict(fit.static_summary(level),orient='index').to_csv(directory/'parameters.csv')
    bx.save_config({key:float(v) if np.isfinite(v) else None for key,v in diagnostic['engine'].items()},directory/'engine.json')
    bx.save_config(dict(bucex_version=bx.__version__,model=fit.model.to_dict(),inference=fit.plan.to_dict(),
        warnings=diagnostic['warnings'],interval='pointwise posterior credible interval',
        status='Research output; assess convergence, sensitivity and held-out forecasts before reporting.'),directory/'run.json')
    pd.DataFrame(diagnostic['pit']).to_csv(directory/'in_sample_pit.csv',index=False)
    targets = scientific_targets(fit)
    fit.contrast_diagnostics(targets,credible_interval=level).to_csv(directory/'scientific_targets.csv')
    bx.residual_serial_check(fit,draws=config.get('predictive_check_draws',200),seed=config['seed']).to_csv(
        directory/'residual_serial.csv',index=False)
    metrics = fit.sampler_diagnostics.get('draw_metrics',{})
    if metrics:
        pd.DataFrame({key:np.ravel(v) for key,v in metrics.items()}).assign(
            chain=np.repeat(np.arange(fit.n_chains),fit.draws_per_chain),
            draw=np.tile(np.arange(fit.draws_per_chain),fit.n_chains)).to_csv(directory/'sampler_metrics.csv',index=False)

    def band(values,label,ylabel='temperature / °C',dates=None):
        dates=fit.time if dates is None else dates
        summary=fit.posterior_summary(values,credible_interval=level)
        pd.DataFrame(dict(time=dates,**summary)).to_csv(directory/(label+'.csv'),index=False)
        if config.get('figures',True):
            save_band(fit,values,directory/label,dates=dates,ylabel=ylabel,level=level)

    forecast = fit.forecast(horizon,draws=config.get('forecast_draws'),seed=config['seed'])
    forecast.summary(level=level).to_csv(directory/'forecast.csv',index=False)
    names = fit.channel_names if fit.is_multiseries_model else (None,)
    for name in names:
        label=name or 'series'
        prior=fit.priors.channels[name] if isinstance(fit.priors,bx.MarginalPriors) else fit.priors
        channel=fit.model.channel(name) if name else fit.model
        band(fit.component_draws('level',channel=name),label+'_level')
        if channel.period is not None:
            band(fit.component_draws('seasonal',channel=name),label+'_seasonal')
        band(fit.channel_eta_draws(name,original_scale=True) if name else fit.eta_draws(original_scale=True),label+'_location')
        band(fit.sigma_draws(channel=name),label+'_observation_scale',ylabel='observation scale / °C')
        effects=fit.innovation_effect_draws(120,channel=name,combine_chains=False)
        practical=fit.contrast_diagnostics(effects,credible_interval=level)
        practical['probability_effect_sd_over_0.1']= [np.mean(effects[key]>.1) for key in practical.index]
        practical['horizon_months']=120
        practical.to_csv(directory/(label+'_innovation_effects.csv'))
        if prior.ssvs is not None:
            fit.component_probabilities(channel=name).to_csv(directory/(label+'_structure.csv'))
            fit.component_transition_summary(channel=name).to_csv(directory/(label+'_structure_mixing.csv'))
        if channel.family=='gev':
            band(fit.return_level_draws(100,channel=name),label+'_return_level_100_blocks')
        if risks and label in risks:
            probability=fit.exceedance_probability_draws(risks[label],channel=name,return_labels=False)
            band(probability,label+'_risk',ylabel=fit.event_label(risks[label],channel=name))
        if config.get('figures',True):
            axis=forecast.plot(channel=name) if name else forecast.plot()
            axis.figure.savefig(directory/(label+'_forecast.pdf'),bbox_inches='tight');plt.close(axis.figure)

    if fit.is_multiseries_model:
        bx.residual_dependence_check(fit,draws=config.get('predictive_check_draws',200),seed=config['seed']).to_csv(
            directory/'residual_dependence.csv',index=False)
        if fit.n_time>=36:
            bx.residual_dependence_check(fit,draws=config.get('predictive_check_draws',200),seed=config['seed'],by_phase=True).to_csv(
                directory/'residual_dependence_by_month.csv',index=False)
        if fit.model.copula is not None:
            fit.copula_summary(credible_interval=level).to_csv(directory/'copula_correlations.csv')
            if config.get('figures',True):
                figure,_=fit.plot('copula');figure.savefig(directory/'copula_correlations.pdf',bbox_inches='tight');plt.close(figure)
        constraints=[pair for pair in bx.UCCLE_ORDER_CONSTRAINTS if set(pair)<=set(fit.channel_names)]
        if constraints:
            for tag,prediction in [('in_sample',fit.posterior_predictive(draws=config.get('predictive_check_draws',200),seed=config['seed'])),('forecast',forecast)]:
                check=prediction.ordering_diagnostics(constraints,observed=fit.observed if tag=='in_sample' else None)
                check.summary.to_csv(directory/('ordering_'+tag+'.csv'),index=False)
                check.by_time.to_csv(directory/('ordering_'+tag+'_by_time.csv'),index=False)
        if {'TXx','TNx'}<=set(fit.channel_names):
            probabilities=forecast.compound_probability_draws({'TXx':('>',config['risks']['TXx']),'TNx':('>',config['risks']['TNx'])})
            band(probabilities,'compound_heat_conditional_risk',ylabel='conditional compound probability',dates=forecast.dates)
            pd.DataFrame(dict(time=forecast.dates,posterior_predictive_probability=probabilities.mean(axis=0))).to_csv(
                directory/'compound_heat_forecast.csv',index=False)
    return directory
