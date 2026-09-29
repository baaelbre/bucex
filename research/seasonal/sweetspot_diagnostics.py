"""Paired-draw diagnostics for level/slope calibration; no new statistical model."""
from pathlib import Path
import numpy as np
import pandas as pd
import bucex as bx


def local_derivative(values, tau, anchor):
    """d E[f|y]/d log(A) = Cov(f, tau**2/A**2 | y), for a HN(A) prior.

    values starts with (chain, draw), followed by optional target dimensions.
    This is a sample covariance diagnostic, not a test or a finite-change refit.
    """
    v=np.asarray(values, dtype=float)
    t=np.asarray(tau, dtype=float)
    if v.shape[:2] != t.shape or t.ndim != 2 or t.size < 2 or anchor <= 0:
        raise ValueError('Align chain/draw dimensions and use a positive anchor.')
    x=v.reshape((t.size,)+v.shape[2:]);score=(t.ravel()/anchor)**2
    return np.tensordot(score-score.mean(), x-x.mean(axis=0), axes=(0,0))/(t.size-1)


def variance_parts(q_level, q_slope, horizon):
    h=int(horizon)
    if h<1 or h!=horizon:raise ValueError('Positive integer horizon required.')
    a=h*np.asarray(q_level)
    b=h*(h-1)*(2*h-1)/6*np.asarray(q_slope)
    fraction=np.divide(b,a+b,out=np.full_like(a+b,np.nan,dtype=float),where=(a+b)>0)
    return a,b,fraction


def _band(values):
    flat=np.asarray(values).reshape(-1)
    lo,med,hi=np.quantile(flat,[.025,.5,.975])
    return dict(mean=float(flat.mean()),lower=float(lo),median=float(med),upper=float(hi))


def write(fit, forecast, config, directory):
    """Retain joint traces, slope sensitivity and forecast allocation on physical scales."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if not fit.is_multiseries_model or fit.priors.shrinkage.hyperprior!='half_normal':
        raise ValueError('This study requires pooled half-normal innovation scales.')
    dates=pd.DatetimeIndex(fit.time)
    multiplier=10*config['model']['steps_per_year']
    if multiplier!=40:raise ValueError('This study is in seasonal units.')
    anchor=config['priors']['innovation_sd']
    tau={c:fit.parameter('shrinkage.shared.'+c,combine_chains=False)
         for c in ('level','slope','seasonal')}
    anchor={c:anchor[k] for c,k in [('level','level'),('slope','trend'),('seasonal','season')]}
    traces={'tau_'+c:x for c,x in tau.items()}
    targets={};paths=[];sensitivity=[];allocations=[];forecast_rows=[];correlations=[]
    # Check yearly historical states and every state in the final 30 years.
    check_indices=np.unique(np.r_[np.arange(0,len(dates),4),np.arange(max(0,len(dates)-120),len(dates))])
    observation_summary=forecast.summary(level=.95)
    for c,x in tau.items():targets['hyperprior_score_'+c]=(x/anchor[c])**2
    for name in fit.channel_names:
        level=fit.component_draws('level',channel=name,combine_chains=False)
        rate=multiplier*fit.component_draws('slope',channel=name,combine_chains=False)
        q={c:fit.parameter(f'sd.channel.{name}.{c}',combine_chains=False)**2
           for c in ('level','slope','seasonal')}
        for label,values in [('level',level),('rate',rate)]:
            lo,med,hi=np.quantile(values.reshape(-1,len(dates)),[.025,.5,.975],axis=0)
            paths.append(pd.DataFrame(dict(channel=name,target=label,time=dates,
                lower=lo,median=med,upper=hi,mean=values.mean(axis=(0,1)))))
            for i in check_indices:targets[f'{name}.{label}.{dates[i]:%Y-%m}']=values[...,i]
            for c,x in tau.items():
                derivative=local_derivative(values,x,anchor[c])
                sd=values.reshape(-1,len(dates)).std(axis=0,ddof=1)
                data=dict(channel=name,target=label,time=dates,anchor_component=c,
                    derivative_per_log_anchor=derivative,
                    derivative_per_posterior_SD=np.divide(derivative,sd,out=np.full_like(sd,np.nan),where=sd>0))
                for chain in range(fit.n_chains):
                    data[f'chain_{chain+1}_derivative']=local_derivative(values[chain:chain+1],x[chain:chain+1],anchor[c])
                sensitivity.append(pd.DataFrame(data))
        traces[name+'_end_level']=level[...,-1]
        traces[name+'_end_rate']=rate[...,-1]
        for c in q:traces[name+'_q_'+c]=q[c]
        for years in (10,30):
            h=4*years
            va,vb,fraction=variance_parts(q['level'],q['slope'],h)
            expected=level[...,-1]+h*rate[...,-1]/multiplier
            traces[f'{name}_slope_fraction_{years}y']=fraction
            traces[f'{name}_conditional_level_mean_{years}y']=expected
            targets[f'{name}.slope_fraction_{years}y']=fraction
            targets[f'{name}.conditional_level_mean_{years}y']=expected
            for label,values in [('level_innovation_SD',np.sqrt(va)),('slope_innovation_SD',np.sqrt(vb)),
                                  ('slope_variance_fraction',fraction)]:
                allocations.append(dict(channel=name,horizon_years=years,target=label,**_band(values)))
            for c,x in tau.items():
                allocations.append(dict(channel=name,horizon_years=years,
                    target='conditional_level_mean_sensitivity_'+c,
                    mean=float(local_derivative(expected,x,anchor[c])),lower=np.nan,median=np.nan,upper=np.nan))
            state_var=float(expected.var(ddof=0))
            allocations.append(dict(channel=name,horizon_years=years,target='forecast_variance_decomposition',
                state_uncertainty=state_var,new_level_variance=float(va.mean()),
                new_slope_variance=float(vb.mean()),total_level_variance=state_var+float(va.mean()+vb.mean())))
        for left,right,x,y in [('log_tau_level','log_tau_slope',np.log(tau['level']),np.log(tau['slope'])),
            ('log_q_level','log_q_slope',np.log(q['level']),np.log(q['slope'])),
            ('end_level','end_rate',level[...,-1],rate[...,-1])]:
            correlations.append(dict(channel=name,left=left,right=right,
                correlation=float(np.corrcoef(x.ravel(),y.ravel())[0,1])))
        # Future level draws preserve the posterior level/slope covariance.
        flevel=forecast.component_draws('level',channel=name)
        for years in (10,30):
            h=4*years
            if h>forecast.horizon:continue
            forecast_rows.append(dict(channel=name,horizon_years=years,horizon=h,
                time=forecast.dates[h-1],target='level',**_band(flevel[:,h-1])))
        obs=observation_summary
        if 'channel' in obs:obs=obs[obs.channel==name].reset_index(drop=True)
        for years in (10,30):
            h=4*years
            if h>len(obs):continue
            r=obs.iloc[h-1]
            forecast_rows.append(dict(channel=name,horizon_years=years,horizon=h,time=r['time'],
                target='observation',lower=r['lower'],median=r['median'],upper=r['upper']))
    pd.concat(paths,ignore_index=True).to_csv(directory/'sweetspot_paths.csv',index=False)
    pd.concat(sensitivity,ignore_index=True).to_csv(directory/'sweetspot_local_sensitivity.csv',index=False)
    pd.DataFrame(allocations).to_csv(directory/'sweetspot_allocation.csv',index=False)
    pd.DataFrame(correlations).to_csv(directory/'sweetspot_correlations.csv',index=False)
    pd.DataFrame(forecast_rows,columns=['channel','horizon_years','horizon','time','target','mean','lower','median','upper']).to_csv(directory/'sweetspot_forecasts.csv',index=False)
    bx.trace_frame(traces).to_csv(directory/'sweetspot_draws.csv.gz',index=False)
    diagnostics=fit.contrast_diagnostics(targets,credible_interval=.95)
    diagnostics.to_csv(directory/'sweetspot_target_diagnostics.csv')
    assessment=bx.convergence_assessment({'sweetspot_targets':diagnostics},**config['diagnostic_thresholds'])
    bx.save_config(assessment,directory/'sweetspot_convergence.json')
    bx.save_config(dict(version=bx.__version__,cell=config.get('sweetspot_cell'),
        initial_rate_sd=config['priors']['initial_slope_sd'],
        interpretation='Pointwise 95% intervals. Local derivatives concern posterior means per natural-log hyperprior scale. Joint traces retain chain/draw pairing. Monte Carlo estimates require convergence; local sensitivity does not replace finite-change refits.',
        allocation='Within each draw: h*q_level + h*(h-1)*(2*h-1)/6*q_slope. Fractions concern NEW trend innovations only, excluding current-state uncertainty, seasonality and observations. Forecast total-level variance includes current-state covariance.',
        diagnostic_times=dates[check_indices].strftime('%Y-%m-%d').tolist()),directory/'sweetspot_definitions.json')
    main=directory/'convergence.json'
    if main.exists():
        combined=bx.load_config(main)
        combined['checked']+=assessment['checked']
        combined['issues']+=assessment['issues']
        if assessment['status']!='passed_numerical_checks':combined['status']=assessment['status']
        bx.save_config(combined,main)
    return assessment
