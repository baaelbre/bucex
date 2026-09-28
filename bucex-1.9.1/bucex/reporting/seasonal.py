"""Seasonal-block forecasts and diagnostics using the ordinary BUCEX results API."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd

from ..core.calendar import meteorological_phases
from ..config import save_config
from ..plotting.style import publication_style
from ..plotting.predictive import plot_chain_traces, plot_predictive_diagnostics, plot_calendar_risk_curves
from ..plotting.traces import parameter_trace_draws, trace_frame

SEASONS = ('DJF','MAM','JJA','SON')


def plot_forecast_seasons(forecast, *, channel=None, threshold=None, level=.95,
                          history=None, history_dates=None):
    """Four physical season panels; forecast and history remain on the same calendar."""
    import matplotlib.pyplot as plt
    forecast = replace(forecast,period=4,phases=meteorological_phases(forecast.dates))
    figure,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
    for phase,(season,ax) in enumerate(zip(SEASONS,axes.flat),1):
        ax.set_title(season,fontsize=11)
        if not np.any(forecast.phases==phase):
            ax.text(.5,.5,'Outside forecast window',ha='center',transform=ax.transAxes)
            continue
        if threshold is None:
            forecast.plot(channel=channel,phase=phase,level=level,ax=ax,
                history=history,history_dates=history_dates,history_points=40)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        else:
            table=forecast.risk_summary(threshold,channel=channel,phase=phase,level=level)
            ax.fill_between(table.time,table.lower,table.upper,alpha=.2)
            ax.plot(table.time,table['mean'])
            ax.set(xlabel='season start',ylabel='event probability')
    return figure,axes


def save_seasonal_prediction_report(fit,directory,*,channel=None,forecast=None,predictive=None,
        threshold=None,level=.95,draws=200,seed=None,figures=True,prefix=None,
        image_format='png',dpi=180,style='manuscript',trace_exports=True,primary=None):
    """PIT/Q-Q, scale and trace exports, seasonal and meteorological-year risks.

    Slope units are degrees per decade (40 quarterly updates). Yearly means
    weight each season by its actual day count. Annual blocks run Dec--Nov;
    no January--December quantity is inferred by splitting a DJF observation.
    """
    import matplotlib.pyplot as plt
    path=Path(directory);path.mkdir(parents=True,exist_ok=True)
    label=prefix or channel or fit.series_name or 'series'
    forecast=fit.forecast(40,draws=draws,seed=seed) if forecast is None else forecast
    predictive=fit.posterior_predictive(draws=draws,seed=seed) if predictive is None else predictive
    observed=fit.observed[:,fit.channel_names.index(channel)] if fit.is_multiseries_model else fit.observed
    phase=meteorological_phases(fit.time)
    written=[]
    def table(frame,stem):
        file=path/f'{label}_{stem}.csv';frame.to_csv(file,index=False);written.append(file.name)
    def save(figure,stem):
        file=path/f'{label}_{stem}.{image_format}'
        figure.savefig(file,dpi=dpi,bbox_inches='tight');plt.close(figure);written.append(file.name)
    with publication_style(style=style,dpi=dpi,primary=primary):
        pit=predictive.pit(observed,channel=channel)
        table(pd.DataFrame(dict(time=fit.time,season=np.asarray(SEASONS)[phase-1],pit=pit)),'smoothed_pit')
        if figures:
            save(plot_predictive_diagnostics(predictive,observed,channel=channel,in_sample=True,max_lag=8)[0],
                 'pit_qq_residuals')
            figure,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
            for i,(season,ax) in enumerate(zip(SEASONS,axes.flat),1):
                values=pit[phase==i]
                ax.hist(values,bins=np.linspace(0,1,11),alpha=.7)
                ax.axhline(len(values)/10,color='black',ls='--',lw=.8)
                ax.set(title=f'{season} (n={len(values)})',xlabel='smoothed PIT',ylabel='count')
            save(figure,'pit_by_season')
        sigma=fit.sigma_draws(channel=channel,combine_chains=False)
        scale_targets={};rows=[]
        for i,season in enumerate(SEASONS,1):
            indices=np.flatnonzero(phase==i)
            if not len(indices):
                continue
            j=indices[-1];values=sigma[...,j]
            lo,mid,hi=np.quantile(values,[(1-level)/2,.5,(1+level)/2])
            rows.append(dict(season=season,date=fit.time[j],lower=lo,median=mid,upper=hi))
            scale_targets[season+' observation scale / °C']=values
        scales=pd.DataFrame(rows);table(scales,'scale_by_season')
        targets={'end slope / °C per decade':40*fit.component_draws('slope',channel=channel,combine_chains=False)[...,-1],
                 **scale_targets}
        table(fit.contrast_diagnostics(targets,credible_interval=level).reset_index(),'scale_and_slope_mcmc')
        traces=parameter_trace_draws(fit,channel=channel)
        if trace_exports:
            for stem,values in [('parameter',traces),('target',targets)]:
                trace_frame(values).to_csv(path/f'{label}_{stem}_traces.csv.gz',index=False)
        if figures:
            save(plot_chain_traces(traces)[0],'parameter_traces')
            save(plot_chain_traces(targets)[0],'scale_and_slope_traces')
            figure,ax=plt.subplots(figsize=(7,3),layout='constrained')
            ax.errorbar(scales.season,scales['median'],
                yerr=[scales['median']-scales.lower,scales.upper-scales['median']],fmt='o-',capsize=3)
            ax.set(xlabel='season',ylabel='observation scale / °C')
            save(figure,'scale_by_season')
            save(plot_forecast_seasons(forecast,channel=channel,level=level,
                history=observed,history_dates=fit.time)[0],'forecast_by_season')
            if threshold is not None:
                save(plot_forecast_seasons(forecast,channel=channel,level=level,threshold=threshold)[0],
                     'forecast_risk_by_season')
                save(plot_forecast_seasons(predictive,channel=channel,level=level,threshold=threshold)[0],
                     'smoothed_risk_by_season')
        for frequency in ('season','year'):
            aggregate=forecast.aggregate(frequency=frequency,channel=channel)
            table(aggregate.summary(level),'forecast_'+frequency)
            omitted=forecast.aggregate(frequency=frequency,channel=channel,include_partial=True).periods
            table(omitted[~omitted.complete],'omitted_partial_'+frequency)
            if not aggregate.n_periods:
                continue
            if threshold is not None:
                table(aggregate.risk_summary(threshold,level=level),'forecast_'+frequency+'_risk')
            if figures:
                save(aggregate.plot(level=level).figure,'forecast_'+frequency)
            figure,_,curves=plot_calendar_risk_curves(aggregate,level=level,points=40)
            table(curves,'forecast_'+frequency+'_risk_curves')
            if figures:
                save(figure,'forecast_'+frequency+'_risk_curves')
            else:
                plt.close(figure)
    notes=dict(frequency='seasonal',slope_unit='°C per decade (40 seasonal updates)',
        phase_order=list(SEASONS),fitted_start=str(fit.time[0]),fitted_end=str(fit.time[-1]),
        last_included_day=str((pd.Timestamp(fit.time[-1])+pd.offsets.MonthEnd(3)).date()),
        yearly_target='Previous December through November; daily-weighted means or max/min',
        residual_assumption='Conditional independence between seasonal blocks; contemporaneous copula only',
        calibration='Smoothed PIT is descriptive; held-out seasonal scores assess forecast performance',
        threshold=threshold,interval_level=level,figures=written)
    save_config(notes,path/f'{label}_prediction_notes.json')
    return notes


__all__=['plot_forecast_seasons','save_seasonal_prediction_report']
