"""Standalone monthly reference/sensitivity overview, including incomplete runs."""
from __future__ import annotations
import argparse
import base64
import calendar
from html import escape
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtri
import bucex as bx
from research.monthly.study_plan import BATCHES, calibration_rows, specification
from research.seasonal.jobs import tasks, result_directory, task_config, fingerprint

CHANNELS=('TXm','TXx','TXn','TNm','TNx','TNn')
MONTH={c:1 if c in ('TXn','TNn') else 7 for c in CHANNELS}
COLORS={c:'#24658a' if c.startswith('TX') else '#a44839' for c in CHANNELS}


def read(path):
    try:return pd.read_csv(path)
    except (FileNotFoundError,pd.errors.EmptyDataError):return pd.DataFrame()


def build(root, *, tier='screen', batch='monthly_all'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);out=root/tier/'collected'/batch;out.mkdir(parents=True,exist_ok=True)
    images=out/'figures';images.mkdir(exist_ok=True)
    all_selected=tasks(batch,tier=tier)
    # Sensitivity-only batches can display a reference already completed in the
    # same root, without counting it as a missing sensitivity task.
    lookup={t.id:t for t in all_selected+tasks('monthly_reference',tier=tier)}
    directories={};rows=[];issues=[];available={};identities={}
    for t in lookup.values():
        folder=root/tier/t.id;marker=folder/'task.json'
        meta=bx.load_config(marker) if marker.exists() else {}
        expected=t.id in {x.id for x in all_selected}
        setting=t.variant.removeprefix('monthly_fixed_')
        row=dict(task=t.id,setting=setting,channel=t.channel,expected=expected,
                 status=meta.get('status','pending'),numerical_status=meta.get('numerical_status','not_assessed'))
        if meta.get('status')=='completed':
            config=task_config(t,tier)
            # Foreign configurations are not silently combined in sensitivity plots.
            identity=dict(bucex_version=bx.__version__,**fingerprint(config))
            if any(meta.get(k)!=v for k,v in identity.items()):
                row.update(status='provenance_mismatch',numerical_status='not_assessed')
            else:
                directory=result_directory(folder,t)
                check=bx.load_config(directory/'convergence.json')
                row['numerical_status']=check['status']
                directories[(setting,t.channel)]=directory
                available.setdefault(setting,{})[t.channel]=directory
                engine=bx.load_config(directory/'engine.json')
                row['state_acceptance']=engine.get('conditional_state_mh_acceptance',engine.get('state_acceptance'))
                mcmc=read(directory/'mcmc.csv')
                if not mcmc.empty:
                    sampled=mcmc[~mcmc['constant'].astype(bool)] if 'constant' in mcmc else mcmc
                    row['max_rhat']=sampled.rhat.max();row['min_bulk_ess']=sampled.ess_bulk.min();row['min_tail_ess']=sampled.ess_tail.min()
                for x in check.get('issues',[]):issues.append(dict(task=t.id,setting=setting,channel=t.channel,**x))
                for target,file in [('end_level_C',t.channel+'_level.csv'),('end_rate_C_per_decade',t.channel+'_slope_C_per_decade.csv')]:
                    table=read(directory/file)
                    if not table.empty:
                        for q in ('lower','median','upper'):row[target+'_'+q]=table.iloc[-1][q]
                shape=read(directory/'parameters.csv')
                if not shape.empty:
                    first=shape.columns[0];shape=shape[shape[first]=='xi.'+t.channel]
                    if not shape.empty:
                        for key in ('lower','median','upper'):row['xi_'+key]=shape.iloc[0].get(key)
        rows.append(row)
    status=pd.DataFrame(rows);status.to_csv(out/'task_status.csv',index=False)
    selected=status[status.expected]
    info=dict(version=bx.__version__,tier=tier,batch=batch,expected=len(selected),
        completed=int((selected.status=='completed').sum()),
        numerically_passed=int(((selected.status=='completed')&(selected.numerical_status=='passed_numerical_checks')).sum()),
        interpretation='Screen diagnostics and sensitivity, not evidence of out-of-sample forecast calibration. Failed numerical checks remain visible; no best setting is selected.')
    bx.save_config(info,out/'status.json')
    pd.DataFrame(issues).to_csv(out/'numerical_issues.csv',index=False)
    pd.DataFrame(calibration_rows()).to_csv(out/'matched_calibration.csv',index=False)
    galleries=[]
    def save(fig,name,caption):
        fig.savefig(images/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
        galleries.append((name,caption))
    def panels(ylabel):
        fig,axes=plt.subplots(2,3,figsize=(12,6.8),layout='constrained')
        for ch,ax in zip(CHANNELS,axes.flat):
            ax.set_title(ch,loc='left',fontweight='bold');ax.set_ylabel(ylabel)
            ax.grid(axis='y',alpha=.15);ax.tick_params(axis='x',labelrotation=45)
        return fig,axes
    def draw(ax,frame,*,label=None,color=None,band=True):
        if frame.empty:return
        x=pd.to_datetime(frame.time) if 'time' in frame else frame.month.to_numpy()
        ax.plot(x,frame['median'].to_numpy(),label=label,color=color,lw=1.4)
        if band:ax.fill_between(x,frame.lower.to_numpy(),frame.upper.to_numpy(),color=color,alpha=.15)
    ref=available.get('reference',{})
    if ref:
        for suffix,title in [('level','Latent level / °C'),('slope_C_per_decade','Rate / °C per decade')]:
            fig,axes=panels(title)
            for ch,ax in zip(CHANNELS,axes.flat):
                frame=read(ref[ch]/(ch+'_'+suffix+'.csv')) if ch in ref else pd.DataFrame()
                if frame.empty:ax.text(.5,.5,'Reference pending',transform=ax.transAxes,ha='center')
                else:draw(ax,frame,color=COLORS[ch]);ax.axhline(0,color='.5',lw=.5) if suffix.startswith('slope') else None
            save(fig,'reference_'+suffix,'Monthly reference: posterior medians and pointwise 95% credible intervals. Inspect numerical status before interpreting the paths.')
        fig,ax=plt.subplots(figsize=(10,4.3),layout='constrained')
        for i,ch in enumerate(CHANNELS):
            f=read(ref[ch]/(ch+'_scale_by_month.csv')) if ch in ref else pd.DataFrame()
            if not f.empty:draw(ax,f,label=ch,color=plt.get_cmap('tab10')(i))
        ax.set(xticks=range(1,13),xticklabels=calendar.month_abbr[1:],ylabel='Observation scale / °C')
        ax.legend(ncol=6);ax.grid(axis='y',alpha=.15)
        save(fig,'reference_monthly_scales','All six repeating monthly observation scales, with 95% credible intervals. Scales are residual SDs for means and GEV scales for extrema.')
        fig,axes=plt.subplots(6,2,figsize=(10,15),layout='constrained')
        pit_tables=[];ppcs=[]
        for ch,row in zip(CHANNELS,axes):
            f=read(ref[ch]/(ch+'_smoothed_pit.csv')) if ch in ref else pd.DataFrame()
            row[0].set_title(ch,loc='left',fontweight='bold')
            if f.empty:continue
            pit=f.pit.to_numpy();z=ndtri(np.clip(pit,1e-10,1-1e-10));theory=ndtri((np.arange(len(z))+.5)/len(z))
            row[0].hist(pit,bins=np.linspace(0,1,11),color=COLORS[ch],alpha=.7)
            row[0].axhline(len(pit)/10,color='.3',ls='--');row[0].set(xlabel='Smoothed PIT',ylabel='Count')
            row[1].plot(theory,np.sort(z),'.',ms=2,color=COLORS[ch]);lim=[min(theory.min(),z.min()),max(theory.max(),z.max())]
            row[1].plot(lim,lim,color='.3',lw=.8);row[1].set(xlabel='Standard-normal quantile',ylabel='Normal-score quantile')
            f['channel']=ch;pit_tables.append(f)
            p=read(ref[ch]/'posterior_predictive_checks.csv')
            if not p.empty:ppcs.append(p)
        save(fig,'reference_pit_qq','In-sample smoothed PIT histograms and normal-score QQ plots. The fitted observations helped estimate the states, so these are descriptive residual diagnostics, not held-out calibration tests.')
        if pit_tables:pd.concat(pit_tables).to_csv(out/'reference_pit.csv',index=False)
        if ppcs:pd.concat(ppcs).to_csv(out/'reference_posterior_predictive_checks.csv',index=False)
        # Calendar QQ curves expose cancellations hidden by a pooled check.
        fig,axes=panels('Normal-score quantile')
        for ch,ax in zip(CHANNELS,axes.flat):
            f=read(ref[ch]/(ch+'_smoothed_pit.csv')) if ch in ref else pd.DataFrame()
            if f.empty:continue
            f['month']=pd.to_datetime(f.time).dt.month
            for month,g in f.groupby('month'):
                z=np.sort(ndtri(np.clip(g.pit.to_numpy(),1e-10,1-1e-10)))
                ax.plot(ndtri((np.arange(len(z))+.5)/len(z)),z,label=calendar.month_abbr[month],lw=.8,color=plt.get_cmap('tab20')(month-1))
            ax.plot([-3,3],[-3,3],color='.3',lw=.8);ax.set_xlabel('Standard-normal quantile')
        legend_items={label:handle for ax in axes.flat for handle,label in zip(*ax.get_legend_handles_labels())}
        handles,labels=list(legend_items.values()),list(legend_items)
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.075),ncol=12,fontsize=8)
        save(fig,'reference_calendar_qq','Supplementary QQ curves by calendar month. No reference envelope or goodness-of-fit p-value is implied.')
        fig,axes=panels('Monthly event probability')
        for ch,ax in zip(CHANNELS,axes.flat):
            f=read(ref[ch]/(ch+'_risk.csv')) if ch in ref else pd.DataFrame()
            if f.empty:continue
            f=f[pd.to_datetime(f.time).dt.month==MONTH[ch]];draw(ax,f,color=COLORS[ch]);ax.set_title(ch+' — '+calendar.month_name[MONTH[ch]],loc='left')
        save(fig,'reference_historical_risk','Historical conditional-risk medians and 95% credible intervals for the declared thresholds, shown in July for warm summaries and January for cold minima. All months are retained in the CSV tables.')
        for widths in (False,True):
            fig,axes=panels('95% interval width / °C' if widths else 'Temperature / °C')
            for ch,ax in zip(CHANNELS,axes.flat):
                f=read(ref[ch]/(ch+'_forecast_uncertainty.csv')) if ch in ref else pd.DataFrame()
                if f.empty:continue
                f=f[(pd.to_datetime(f.time).dt.month==MONTH[ch])&np.isclose(f.nominal,.95)]
                for target,color,label in [('observation','#24658a','Observation'),('location','#a44839','Latent location')]:
                    g=f[f.target==target]
                    if widths:ax.plot(pd.to_datetime(g.time),g.interval_width,color=color,label=label)
                    else:draw(ax,g,label=label,color=color)
                ax.set_title(ch+' — '+calendar.month_name[MONTH[ch]],loc='left');ax.legend(fontsize=8)
            save(fig,'reference_forecast_widths' if widths else 'reference_forecast',
                'Thirty-year forecasts: '+('central 95% interval widths.' if widths else 'latent location and observations with central 95% intervals.')+' Monthly predictive quantiles use the averaged conditional CDF; future state innovations are simulated. These are not validation results.')
        fig,axes=panels('Change in monthly seasonal component / °C')
        for ch,ax in zip(CHANNELS,axes.flat):
            f=read(ref[ch]/(ch+'_seasonal_change.csv')) if ch in ref else pd.DataFrame()
            if f.empty:continue
            f['month']=pd.to_datetime(f.time).dt.month
            for month,g in f.groupby('month'):
                draw(ax,g,label=calendar.month_abbr[month],color=plt.get_cmap('tab20')(month-1))
        legend_items={label:handle for ax in axes.flat for handle,label in zip(*ax.get_legend_handles_labels())}
        handles,labels=list(legend_items.values()),list(legend_items)
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.075),ncol=12,fontsize=8)
        save(fig,'reference_seasonal_changes','Evolution of each calendar-month seasonal contribution relative to its first observed occurrence. Lines and bands are posterior medians and 95% intervals.')
        fig,axes=plt.subplots(6,4,figsize=(15,12),layout='constrained')
        for ch,row in zip(CHANNELS,axes):
            f=read(ref[ch]/(ch+'_parameter_traces.csv.gz')) if ch in ref else pd.DataFrame()
            keys=['sd.channel.'+ch+'.'+component for component in ('level','slope','seasonal')]
            keys+=['xi.'+ch if ch not in ('TXm','TNm') else 'initial.channel.'+ch+'.slope']
            for key,ax in zip(keys,row):
                ax.set_title(ch+' '+key.split('.')[-1],fontsize=9)
                if key not in f:continue
                for chain,g in f.groupby('chain'):
                    ax.plot(g.draw,g[key],lw=.6,label='Chain '+str(chain))
                ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
        axes[0,0].legend(fontsize=7)
        save(fig,'reference_parameter_traces','Two-chain screening traces for physical innovation SDs and GEV shape (initial slope for Gaussian responses). Trace appearance and high state acceptance do not replace ESS and R-hat checks.')
        # Physical innovation amplitudes and initial-rate learning.
        fig,axes=plt.subplots(1,4,figsize=(14,4.2),layout='constrained')
        for j,component in enumerate(('level','slope','seasonal','initial_slope')):
            for i,ch in enumerate(CHANNELS):
                if ch not in ref:continue
                f=read(ref[ch]/('initial_slope_prior_posterior.csv' if component=='initial_slope' else ch+'_prior_posterior.csv'))
                if f.empty:continue
                f=f[(f.component==component)&(f.scale==('signed_rate' if component=='initial_slope' else 'SD'))]
                for dist,offset,color in [('prior',-.13,'.6'),('posterior',.13,COLORS[ch])]:
                    a=f[f.distribution==dist]
                    if a.empty:continue
                    r=a.iloc[0];axes[j].errorbar(r['median'],i+offset,xerr=[[r['median']-r.lower],[r.upper-r['median']]],fmt='o',ms=3,color=color)
            axes[j].set(yticks=range(6),yticklabels=CHANNELS,title=component.replace('_',' '),xlabel='°C per decade' if component=='initial_slope' else 'Innovation SD / monthly units')
        save(fig,'reference_prior_posterior','Fixed-prior (grey) and posterior (response colour) medians and central 95% intervals. Initial rates use °C per decade; innovation SDs use monthly transition units. There are no fitted hyperparameters.')
    # Compare related settings in readable panels; keep the reference band.
    families=[('level_slope',['reference']+[v['setting'] for v in specification()['variants'] if 'sd_multipliers' in v and set(v['sd_multipliers'])=={'level','trend'}]),
        ('seasonal',['reference','half_season','double_season','fixed_location_seasonality']),
        ('initial_state',['reference','half_initial_slope','double_initial_slope','initialization_bridge']),
        ('observation',['reference','xi_narrow','xi_wide','constant_dispersion','narrow_monthly_log_scale','wide_monthly_log_scale','observation_scale_half','observation_scale_double'])]
    comparisons=[]
    for family,settings in families:
        if not any(s in available for s in settings if s!='reference'):continue
        for suffix,ylabel in [('level','Latent level / °C'),('slope_C_per_decade','Rate / °C per decade'),('risk','Monthly event probability')]:
            fig,axes=panels(ylabel)
            for ch,ax in zip(CHANNELS,axes.flat):
                for i,setting in enumerate(settings):
                    d=directories.get((setting,ch));f=read(d/(ch+'_'+suffix+'.csv')) if d else pd.DataFrame()
                    if f.empty:continue
                    if suffix=='risk':f=f[pd.to_datetime(f.time).dt.month==MONTH[ch]]
                    draw(ax,f,label=setting.replace('_',' '),color='black' if setting=='reference' else plt.get_cmap('tab10')(i%10),band=True)
                    base=read(ref[ch]/(ch+'_'+suffix+'.csv')) if ch in ref else pd.DataFrame()
                    if suffix=='risk' and not base.empty:base=base[pd.to_datetime(base.time).dt.month==MONTH[ch]]
                    if setting!='reference' and not base.empty:
                        m=f.merge(base,on='time',suffixes=('_case','_ref'),validate='one_to_one')
                        if len(m)==len(f)==len(base):
                            comparisons.append(dict(setting=setting,channel=ch,target=suffix,
                                max_median_difference=float(np.max(np.abs(m.median_case-m.median_ref))),
                                max_interval_endpoint_difference=float(max(np.max(np.abs(m.lower_case-m.lower_ref)),np.max(np.abs(m.upper_case-m.upper_ref))))))
            legend_items={label:handle for ax in axes.flat for handle,label in zip(*ax.get_legend_handles_labels())}
            handles,labels=list(legend_items.values()),list(legend_items)
            fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.075),ncol=3,fontsize=8)
            save(fig,'sensitivity_'+family+'_'+suffix,'Monthly sensitivity: '+family.replace('_',' ')+'. Lines show posterior medians; shading shows pointwise 95% intervals. Use task_status.csv to identify fits needing numerical review; differences are not automatically attributed to the prior.')
    pd.DataFrame(comparisons).to_csv(out/'trajectory_sensitivity.csv',index=False)
    for name in ('posterior_predictive_checks.csv','risk_period_summaries.csv','initial_slope_prior_posterior.csv','fixed_prior_settings.csv','scientific_targets.csv'):
        frames=[]
        for (setting,ch),d in directories.items():
            f=read(d/name)
            if not f.empty:frames.append(f.assign(setting=setting,response=ch))
        if frames:pd.concat(frames,ignore_index=True).to_csv(out/('combined_'+name),index=False)
    body=['<!doctype html><html><head><meta charset="utf-8"><title>BUCEX monthly screen</title><style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:0 18px;color:#233345}img{width:100%;height:auto}table{border-collapse:collapse;font-size:12px}td,th{padding:5px;border:1px solid #ddd}.notice{background:#fff0cb;padding:15px}figure{margin:30px 0}figcaption{margin:10px 0}details{overflow:auto}</style></head><body>',
        '<h1>Monthly reference and sensitivity — BUCEX 1.9.8.2</h1>',
        '<p class="notice">'+escape(f"{info['completed']}/{info['expected']} fits completed; {info['numerically_passed']} passed the declared numerical screen. "+info['interpretation'])+'</p>',
        '<p>Six separate Gaussian/GEV structural models with fixed normal innovation priors. Twelve repeating observation scales, except in the constant-dispersion sensitivity. No hyperpriors or residual copula. All displayed intervals are 95%; the forecast tables also contain the 99% interval and one-sided 95%/99% quantiles.</p>',
        '<p>Reference monthly SDs: level 0.057735; slope 0.000383292; seasonal 0.1; initial rate 0.00333333 °C/month. The corresponding seasonal SDs are 0.1, 0.002, 0.1 and 0.01 °C/season. The 30-year effects match; the full discretized processes need not.</p>',
        '<details open><summary>Run status, endpoints and numerical checks</summary>'+status.to_html(index=False,escape=True,float_format=lambda x:f'{x:.4g}')+'</details>']
    for name,caption in galleries:
        data=base64.b64encode((images/(name+'.png')).read_bytes()).decode()
        body.append('<figure><img alt="'+escape(name)+'" src="data:image/png;base64,'+data+'"><figcaption>'+escape(caption)+'</figcaption></figure>')
    body.append('<p>Saved fit.bucex archives remain in the individual result directories for later monthly-versus-seasonal forecast comparisons and longer-chain follow-up. Compact export ZIPs omit these large posterior archives.</p></body></html>')
    (out/'index.html').write_text('\n'.join(body),encoding='utf-8')
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--tier',choices=('screen','paper'),default='screen');p.add_argument('--batch',choices=BATCHES,default='monthly_all')
    a=p.parse_args();print(build(a.root,tier=a.tier,batch=a.batch))
