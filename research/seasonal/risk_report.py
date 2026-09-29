"""Marginal risk, stationary-equivalent periods and annual block aggregation."""
from pathlib import Path
import numpy as np
import pandas as pd
import bucex as bx


def bands(values,level=.95):
    lo,med,hi=np.quantile(values,[(1-level)/2,.5,(1+level)/2],axis=0)
    return dict(mean=np.mean(values,axis=0),lower=lo,median=med,upper=hi)


def reciprocal(values):
    values=np.asarray(values,dtype=float)
    return np.divide(1.,values,out=np.full(values.shape,np.inf),where=values>0)


def risk_table(probabilities,level=.95):
    summary=bands(probabilities,level)
    return dict(**summary,stationary_equivalent_period=reciprocal(summary['mean']),
        period_lower=reciprocal(summary['upper']),period_median=reciprocal(summary['median']),
        period_upper=reciprocal(summary['lower']),credible_interval=level)


def save(fit,forecast,predictive,config,directory,level=.95):
    directory=Path(directory)
    periods=config.get('risk_return_periods',[10,20,50,100])
    seasonal=config['data'].get('frequency')=='seasonal'
    names=fit.channel_names if fit.is_multiseries_model else (None,)
    rows=[];historical=[];future=[]
    # Nested annual CDF inversions need fewer paired paths than the marginal
    # predictive fan chart. Subsample whole joint paths, never channels/times.
    from dataclasses import replace
    limit=config.get('annual_risk_draws',5000)
    annual_forecast=forecast
    if forecast.n_draws>limit:
        idx=np.random.default_rng(config.get('seed',196)).choice(forecast.n_draws,limit,replace=False)
        annual_forecast=replace(forecast,observations=forecast.observations[idx],eta=forecast.eta[idx],
            states=forecast.states[idx],parameters={k:v[idx] for k,v in forecast.parameters.items()},_quantile_cache={})
    for name in names:
        label=name or fit.series_name or 'series'
        channel=fit.model.channel(name) if name else fit.model
        threshold=config['risks'][label]
        probabilities=fit.exceedance_probability_draws(threshold,channel=name,return_labels=False)
        dates=pd.DatetimeIndex(fit.time)
        for period,bounds in (config.get('risk_periods') or {}).items():
            for phase in ((12,3,6,9) if seasonal else range(1,13)):
                mask=(dates>=pd.Timestamp(bounds[0])) & (dates<=pd.Timestamp(bounds[1])) & (dates.month==phase)
                if not mask.any():continue
                values=probabilities[:,mask].mean(axis=1)
                rows.append(dict(channel=label,period=period,phase=phase,threshold=threshold,
                    blocks=int(mask.sum()),**risk_table(values,level)))
        pd.DataFrame(dict(time=dates,threshold=threshold,**risk_table(probabilities,level))).to_csv(
            directory/(label+'_return_periods.csv'),index=False)
        forecast.risk_summary(threshold,channel=name,level=level).to_csv(
            directory/(label+'_forecast_risk.csv'),index=False)
        if channel.family=='gev':
            for r in periods:
                values=fit.return_level_draws(r,channel=name)
                historical.append(pd.DataFrame(dict(time=dates,channel=label,return_period=r,
                    credible_interval=level,**bands(values,level))))
                values=forecast.return_level(r,channel=name)
                future.append(pd.DataFrame(dict(time=forecast.dates,channel=label,return_period=r,
                    credible_interval=level,**bands(values,level))))
        if seasonal and config.get('annual_risk',False):
            for tag,prediction in (('historical',predictive),('forecast',annual_forecast)):
                aggregate=prediction.aggregate(frequency='year',channel=name)
                if not aggregate.n_periods:continue
                aggregate.summary(level=level).assign(interval_level=level).to_csv(
                    directory/(label+'_'+tag+'_annual_predictive.csv'),index=False)
                values=aggregate.probability_draws(threshold)
                aggregate.periods.assign(channel=label,threshold=threshold,
                    **risk_table(values,level)).to_csv(directory/(label+'_'+tag+'_annual_risk.csv'),index=False)
                # Annual return levels are useful for maxima/minima, not for means.
                if channel.family=='gev':
                    tables=[]
                    for r in periods:
                        values=aggregate.return_level(r)
                        tables.append(aggregate.periods.assign(channel=label,return_period=r,
                            credible_interval=level,**bands(values,level)))
                    pd.concat(tables).to_csv(directory/(label+'_'+tag+'_annual_return_levels.csv'),index=False)
    if rows:pd.DataFrame(rows).to_csv(directory/'risk_period_summaries.csv',index=False)
    if historical:pd.concat(historical).to_csv(directory/'historical_return_levels.csv',index=False)
    if future:pd.concat(future).to_csv(directory/'forecast_return_levels.csv',index=False)
    if config.get('experiment',{}).get('batch_kind')=='pre2019':
        # Evaluation only: these observations never enter the fit or its prior.
        actual=bx.load_uccle_multiseries(**dict(config['data'],start='2019-06',end='2019-08'))
        table=forecast.summary(level=level)
        table['observed']=[float(actual.loc[pd.Timestamp(row.time),row.channel]) for row in table.itertuples()]
        table['above_upper_95']=table.observed>table.upper
        table['below_lower_95']=table.observed<table.lower
        table.to_csv(directory/'pre2019_observed_predictions.csv',index=False)
        curves=[]
        if 'TXx' in fit.channel_names:
            for threshold in sorted(set([35.,36.6,39.7]+np.linspace(30,42,61).tolist())):
                curves.append(forecast.risk_summary(threshold,channel='TXx',level=level).assign(credible_interval=level))
            pd.concat(curves).to_csv(directory/'pre2019_TXx_risk_curve.csv',index=False)
    bx.save_config(dict(credible_interval=level,
        risk_intervals='Conditional probabilities across posterior parameters and latent paths; their mean is the predictive probability.',
        return_period='Reciprocal local event probability, in seasonal/annual blocks as labelled; not a nonstationary waiting time.',
        return_level='Posterior intervals for conditional quantiles; distinct from quantiles of the posterior predictive mixture.',
        annual_window='Complete meteorological years, previous December through November. Partial years omitted.',
        annual_assumption='Observations independent across time conditional on the full latent path and parameters. CDF products formed within each draw before averaging.',
        historical_annual_draws=predictive.n_draws,forecast_draws=forecast.n_draws,
        forecast_annual_draws=annual_forecast.n_draws),directory/'risk_definitions.json')
