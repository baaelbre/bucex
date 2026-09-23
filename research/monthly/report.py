"""Small report writer; all scientific calculations use public bucex methods."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import bucex as bx
from bucex.plotting.style import styled_report


def new_run(root, name):
    """Create a fresh directory without overwriting an earlier analysis."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = Path(root) / f"{name}_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_band(fit, values, path, *, dates=None, ylabel="temperature / °C", level=0.95, image_format="png", dpi=180, color=None):
    """Export one scientific band with readable labels and no default title."""
    return _save_band(fit, values, path, dates=dates, ylabel=ylabel, level=level, image_format=image_format, dpi=dpi, color=color)


def _save_band(fit, values, path, *, dates=None, ylabel="temperature / °C", level=0.95, image_format="png", dpi=180, color=None):
    """Export pointwise posterior summaries and a figure on the same scale."""
    dates = fit.time if dates is None else dates
    band = fit.posterior_summary(values, credible_interval=level)
    pd.DataFrame({"time": dates, **band}).to_csv(path.with_suffix(".csv"), index=False)
    figure, axis = plt.subplots(figsize=(8, 3))
    axis.fill_between(dates, band["lower"], band["upper"], alpha=0.2, color=color)
    axis.plot(dates, band["median"], color=color)
    axis.set(xlabel="time", ylabel=ylabel)
    figure.tight_layout()
    figure.savefig(path.with_suffix("."+image_format), dpi=dpi)
    plt.close(figure)


def convergence_parameters(table, n_chains):
    """Screen scientific parameters; fixed shrinkage hyperparameters remain in mcmc.csv."""
    names = [name for name in table.index if str(name).startswith(
        ('sd.', 'initial.', 'sigma', 'xi', 'copula.', 'scale.', 'scale_slope', 'scale_rw_sd', 'evolution.', 'shrinkage.shared.'))]
    return table.loc[names].assign(chains=n_chains)


def scientific_targets(fit, config=None):
    """Endpoint diagnostics plus explicitly declared climate-period estimands."""
    targets = {}
    rate_multiplier = 10*(config or {}).get('model',{}).get('steps_per_year',12)
    for name in fit.channel_names if fit.is_multiseries_model else (None,):
        values = fit.component_draws("level", channel=name, combine_chains=False)
        targets[f"{name or 'series'}_level_change"] = values[..., -1] - values[..., 0]
        slopes = fit.component_draws("slope", channel=name, combine_chains=False)
        targets[f"{name or 'series'}_end_slope_C_per_decade"] = slopes[..., -1]*rate_multiplier
    if fit.is_multiseries_model:
        for left, right in (("TXx", "TXm"), ("TNn", "TNm"), ("TXm", "TNm"), ("TXx", "TNx")):
            if {left, right} <= set(fit.channel_names):
                targets[f"{left}_minus_{right}_change"] = targets[f"{left}_level_change"] - targets[f"{right}_level_change"]
    periods = (config or {}).get('contrasts')
    if periods:
        pairs = [pair for pair in periods.get('pairs', [])
                 if fit.is_multiseries_model and set(pair) <= set(fit.channel_names)]
        targets.update(bx.period_contrasts(fit, periods['reference'], periods['comparison'],
            pairs=pairs, months=periods.get('months', []),
            rate_multiplier=periods.get('rate_multiplier', 120.)))
    return targets


@styled_report
def write_report(fit, directory, *, config, risks=None, horizon=12, level=.95, save_fit=True):
    """Export auditable numerical results; figures are optional, never evidence of convergence."""
    from dataclasses import asdict
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    image_format = config.get('figure_format', 'png')
    dpi = config.get('figure_dpi', 180)
    seasonal_blocks = config.get('data',{}).get('frequency')=='seasonal'
    rate_multiplier = 10*config.get('model',{}).get('steps_per_year',4 if seasonal_blocks else 12)
    if save_fit and config.get('save_fits', True):
        fit.save(directory/'fit.bucex')
    bx.save_config(config,directory/'config.json')
    bx.save_config(asdict(fit.priors), directory/'declared_priors.json')
    diagnostic = fit.diagnostics()
    diagnostic['parameters'].to_csv(directory/'mcmc.csv')
    pd.DataFrame.from_dict(fit.static_summary(level),orient='index').to_csv(directory/'parameters.csv')
    bx.save_config({key:float(v) if np.isfinite(v) else None for key,v in diagnostic['engine'].items()},directory/'engine.json')
    bx.save_config(dict(bucex_version=bx.__version__,model=fit.model.to_dict(),inference=fit.plan.to_dict(),
        execution=fit.sampler_diagnostics.get('execution'),
        fitted_start=str(fit.time[0]),fitted_end=str(fit.time[-1]),n_blocks=fit.n_time,
        n_months=None if seasonal_blocks else fit.n_time,block_frequency='seasonal' if seasonal_blocks else 'monthly',
        warnings=diagnostic['warnings'],interval='pointwise posterior credible interval',
        credible_interval=level,figure_style=config.get('figure_style','manuscript'),
        status='Research output; assess convergence, sensitivity and held-out forecasts before reporting.'),directory/'run.json')
    pd.DataFrame(diagnostic['pit']).to_csv(directory/'in_sample_pit.csv',index=False)
    if getattr(fit.priors, 'shrinkage', None) is not None:
        bx.save_shared_shrinkage_report(fit, directory, level=level,
            figures=config.get('figures', True), style=config.get('figure_style', 'manuscript'), dpi=dpi,
            seed=config['seed'], horizon=config.get('prior_calibration', {}).get('horizon', 360),
            rate_multiplier=rate_multiplier,
            response_unit='degC', rate_unit='°C per decade')
        pd.DataFrame(fit.priors.shrinkage.calibration(period=config['model']['period'],
            **config.get('prior_calibration', {}))).to_csv(directory/'prior_calibration.csv',index=False)
    targets = scientific_targets(fit, config)
    target_table = fit.contrast_diagnostics(targets, credible_interval=level)
    target_table['probability_positive'] = [float(np.mean(targets[key] > 0)) for key in target_table.index]
    target_table.to_csv(directory/'scientific_targets.csv')
    parameter_table = convergence_parameters(diagnostic['parameters'], fit.n_chains)
    assessment = bx.convergence_assessment({'parameters': parameter_table, 'scientific_targets': target_table},
        **config.get('diagnostic_thresholds', {}))
    bx.save_config(assessment, directory/'convergence.json')
    if config.get('contrasts'):
        bx.save_config(config['contrasts'], directory/'contrast_definitions.json')
        target_table.loc[[key for key in target_table.index if '.' in key]].to_csv(directory/'period_contrasts.csv')
    bx.residual_serial_check(fit,lags=(1,4 if seasonal_blocks else 12),
        draws=config.get('predictive_check_draws',200),seed=config['seed']).to_csv(
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
            save_band(fit,values,directory/label,dates=dates,ylabel=ylabel,level=level,image_format=image_format,dpi=dpi,
                      color=config.get('figure_colors',{}).get(label.split('_')[0]))

    forecast = fit.forecast(horizon,draws=config.get('forecast_draws'),seed=config['seed'])
    forecast.summary(level=level).to_csv(directory/'forecast.csv',index=False)
    predictive = fit.posterior_predictive(draws=config.get('predictive_check_draws',200),seed=config['seed'])
    bx.posterior_predictive_checks(fit,prediction=predictive,seed=config['seed']).to_csv(
        directory/'posterior_predictive_checks.csv',index=False)
    names = fit.channel_names if fit.is_multiseries_model else (None,)
    shape_rows = []
    for name in names:
        label=name or fit.series_name or 'series'
        prior=fit.priors.channels[name] if isinstance(fit.priors,bx.MarginalPriors) else fit.priors
        channel=fit.model.channel(name) if name else fit.model
        if isinstance(fit.priors, bx.MarginalPriors) or not fit.is_multiseries_model:
            comparison = bx.compare_innovation_priors(fit, channel=name,
                size=config.get('prior_draws', 2000), seed=config['seed'], level=level)
            comparison.to_csv(directory/(label+'_prior_posterior.csv'), index=False)
            bx.innovation_prior_diagnostics(comparison).to_csv(directory/(label+'_prior_updates.csv'), index=False)
        band(fit.component_draws('level',channel=name),label+'_level')
        band(rate_multiplier*fit.component_draws('slope',channel=name),label+'_slope_C_per_decade',ylabel='latent slope / °C per decade')
        if channel.period is not None:
            band(fit.component_draws('seasonal',channel=name),label+'_seasonal')
        band(fit.channel_eta_draws(name,original_scale=True) if name else fit.eta_draws(original_scale=True),label+'_location')
        band(fit.sigma_draws(channel=name),label+'_observation_scale',ylabel='observation scale / °C')
        effects=fit.innovation_effect_draws(rate_multiplier,channel=name,combine_chains=False)
        practical=fit.contrast_diagnostics(effects,credible_interval=level)
        practical['probability_effect_sd_over_0.1']= [np.mean(effects[key]>.1) for key in practical.index]
        practical['horizon_months']=120
        practical['horizon_updates']=rate_multiplier
        practical.to_csv(directory/(label+'_innovation_effects.csv'))
        if channel.family=='gev':
            band(fit.return_level_draws(100,channel=name),label+'_return_level_100_blocks')
            lower, upper = channel.observation.xi_bounds
            lower = max(lower, -prior.xi_max_abs, getattr(prior.xi, 'lower', -np.inf))
            upper = min(upper, prior.xi_max_abs, getattr(prior.xi, 'upper', np.inf))
            shape = fit.parameter('xi.'+name if name else 'xi')
            shape_rows.append(dict(channel=label, shape_prior_lower=lower, shape_prior_upper=upper,
                prior_allows_infinite_conditional_mean=upper>1,
                prior_allows_infinite_conditional_variance=upper>.5,
                retained_fraction_xi_ge_1=float(np.mean(shape>=1)),
                retained_fraction_xi_ge_half=float(np.mean(shape>=.5)),
                note='Retained fractions do not establish existence of full predictive moments; use probabilities and quantiles.'))
        for extra_threshold in config.get('additional_risks', {}).get(label, []):
            tag = str(extra_threshold).replace('-', 'minus').replace('.', 'p')
            band(fit.exceedance_probability_draws(extra_threshold,channel=name,return_labels=False),
                 label+'_risk_'+tag,ylabel=fit.event_label(extra_threshold,channel=name))
        threshold = (risks or {}).get(label, (risks or {}).get('series'))
        if threshold is not None:
            probability=fit.exceedance_probability_draws(threshold,channel=name,return_labels=False)
            band(probability,label+'_risk',ylabel=fit.event_label(threshold,channel=name))
            periods = config.get('contrasts')
            if periods:
                paired = probability.reshape(fit.n_chains, fit.draws_per_chain, fit.n_time)
                risk_targets = {}
                for month in periods.get('months', []):
                    early = bx.period_average(paired, fit.time, periods['reference'], month=month)
                    late = bx.period_average(paired, fit.time, periods['comparison'], month=month)
                    risk_targets.update({f'month_{month:02d}.reference':early,
                        f'month_{month:02d}.comparison':late, f'month_{month:02d}.change':late-early})
                if risk_targets:
                    fit.contrast_diagnostics(risk_targets,credible_interval=level).to_csv(directory/(label+'_period_risks.csv'))
        if config.get('figures',True):
            axis=forecast.plot(channel=name,level=level,history=fit.observed,history_dates=fit.time,history_points=120)
            axis.figure.savefig(directory/(label+'_forecast.'+image_format),bbox_inches='tight',dpi=dpi);plt.close(axis.figure)
        prediction_report = bx.save_seasonal_prediction_report if seasonal_blocks else bx.save_prediction_report
        calendar_options = {} if seasonal_blocks else {'months':config.get('forecast_months',list(range(1,13)))}
        prediction_report(fit,directory,channel=name,forecast=forecast,predictive=predictive,
            threshold=threshold,level=level,draws=config.get('predictive_check_draws',200),seed=config['seed'],
            **calendar_options,image_format=image_format,dpi=dpi,
            figures=config.get('figures',True),prefix=label,
            style=config.get('figure_style','manuscript'),
            trace_exports=config.get('trace_exports',True),primary=config.get('figure_colors',{}).get(label))

    if shape_rows:
        pd.DataFrame(shape_rows).to_csv(directory/'shape_support.csv',index=False)
    if fit.is_multiseries_model:
        bx.residual_dependence_check(fit,draws=config.get('predictive_check_draws',200),seed=config['seed']).to_csv(
            directory/'residual_dependence.csv',index=False)
        if fit.n_time>=36:
            bx.residual_dependence_check(fit,draws=config.get('predictive_check_draws',200),seed=config['seed'],by_phase=True).to_csv(
                directory/('residual_dependence_by_season.csv' if seasonal_blocks else 'residual_dependence_by_month.csv'),index=False)
        if fit.model.copula is not None:
            fit.copula_summary(credible_interval=level).to_csv(directory/'copula_correlations.csv')
            if config.get('figures',True):
                figure,_=fit.plot('copula');figure.savefig(directory/('copula_correlations.'+image_format),bbox_inches='tight',dpi=dpi);plt.close(figure)
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


def main():
    """Re-export a saved fit, without refitting or changing its training dates."""
    import argparse
    parser=argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--fit',type=Path,required=True)
    parser.add_argument('--config',type=Path,help='Defaults to config.json beside the saved fit.')
    parser.add_argument('--output',type=Path,default=Path('results/serra_184_monthly'))
    parser.add_argument('--horizon',type=int)
    parser.add_argument('--months',type=int,nargs='+',default=list(range(1,13)))
    parser.add_argument('--format',choices=('png','pdf','svg'),default='png')
    parser.add_argument('--style',choices=('manuscript','default'),default='manuscript')
    parser.add_argument('--dpi',type=int,default=180)
    parser.add_argument('--level',type=float,help='Recompute intervals at this level from saved draws.')
    args=parser.parse_args()
    fit=bx.load_fit(args.fit)
    config=bx.load_config(args.config or args.fit.parent/'config.json')
    config.update(figures=True,figure_format=args.format,forecast_months=args.months,
                  figure_style=args.style,figure_dpi=args.dpi)
    if args.level is not None:
        if not 0 < args.level < 1:
            parser.error('--level must lie in (0,1).')
        config['credible_interval']=args.level
    if args.horizon is not None:
        config['forecast_horizon']=args.horizon
    # Explicitly distinguish fit provenance from the current report settings.
    config['report_source_fit']=str(args.fit.resolve())
    directory=new_run(args.output,'report_'+(fit.series_name or 'joint'))
    print(f'Report only: fitted {fit.time[0]} to {fit.time[-1]}; no refit.',flush=True)
    print(write_report(fit,directory,config=config,risks=config.get('risks'),
        horizon=config.get('forecast_horizon',120),level=config.get('credible_interval',.95),save_fit=False))


if __name__=='__main__':
    main()
