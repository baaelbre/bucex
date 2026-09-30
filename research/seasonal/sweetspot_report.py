"""Review a joint calibration grid, including partial or combined host outputs.

The declared tolerances identify screening candidates, never an optimal prior.
No MCMC archives are needed: the compact evidence exports contain the inputs.
"""
from __future__ import annotations
import argparse
import base64
from html import escape
from itertools import combinations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import bucex as bx
from research.seasonal.jobs import ROOT, tasks, task_config, fingerprint, result_directory
from research.seasonal.sweetspot_plan import BATCHES, cells, study_variants, specification, calibration_rows

CHANNELS = ('TXm','TNm','TXx','TXn','TNx','TNn')


def read(path):
    try: return pd.read_csv(path)
    except (FileNotFoundError,pd.errors.EmptyDataError): return pd.DataFrame()


def aligned_difference(left, right, keys, columns):
    """Maximum absolute differences; missing/mismatched cases cannot pass."""
    if left.empty or right.empty: return np.nan
    if not set(keys+columns) <= set(left) or not set(keys+columns) <= set(right): return np.nan
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError('Repeated comparison keys: '+str(keys))
    joined=left[keys+columns].merge(right[keys+columns],on=keys,how='outer',
        suffixes=('_a','_b'),indicator=True,validate='one_to_one')
    if not joined['_merge'].eq('both').all(): return np.nan
    delta=np.column_stack([joined[c+'_a']-joined[c+'_b'] for c in columns])
    return float(np.max(np.abs(delta))) if np.isfinite(delta).all() else np.nan


def compare_pair(left, right, *, tolerances=None):
    """Compare two full-record cells, separately for each response."""
    tol=tolerances or specification()['tolerances'];rows=[]
    for channel in CHANNELS:
        row=dict(channel=channel)
        limits={}
        for target in ('level','rate'):
            a=left['paths'];b=right['paths']
            if not a.empty:a=a[(a.channel==channel)&(a.target==target)]
            if not b.empty:b=b[(b.channel==channel)&(b.target==target)]
            for window in ('history','recent'):
                aa,bb=a,b
                if window=='recent' and not a.empty and not b.empty:
                    end=max(pd.to_datetime(a.time).max(),pd.to_datetime(b.time).max())
                    cutoff=end-pd.DateOffset(years=specification()['recent_years'])
                    aa=a[pd.to_datetime(a.time)>cutoff];bb=b[pd.to_datetime(b.time)>cutoff]
                for band,columns in [('median',['median']),('interval',['lower','upper'])]:
                    key=f'{window}_{target}_{band}'
                    row[key]=aligned_difference(aa,bb,['time'],columns)
                    limits[key]=tol[f'{target}_{band}_'+('C' if target=='level' else 'C_per_decade')]
        for target in ('level','observation'):
            a,b=left['forecasts'],right['forecasts']
            if not a.empty:a=a[(a.channel==channel)&(a.target==target)]
            if not b.empty:b=b[(b.channel==channel)&(b.target==target)]
            # Both 10- and 30-year forecasts must be present.
            complete=all('horizon_years' in x and set(x.horizon_years)=={10,30} for x in (a,b))
            for band,columns in [('median',['median']),('interval',['lower','upper'])]:
                key=f'forecast_{target}_{band}'
                row[key]=aligned_difference(a,b,['horizon_years','time'],columns) if complete else np.nan
                limits[key]=tol[key+'_C']
        a,b=left['allocation'],right['allocation']
        if not a.empty:a=a[(a.channel==channel)&(a.target=='slope_variance_fraction')]
        if not b.empty:b=b[(b.channel==channel)&(b.target=='slope_variance_fraction')]
        for band,columns in [('median',['median']),('interval',['lower','upper'])]:
            key='allocation_'+band
            row[key]=aligned_difference(a,b,['horizon_years'],columns)
            limits[key]=tol['slope_variance_fraction']
        a,b=left['risk'],right['risk']
        if not a.empty:a=a[a.channel==channel]
        if not b.empty:b=b[b.channel==channel]
        row['risk_mean']=aligned_difference(a,b,['time','threshold','direction'],['mean'])
        limits['risk_mean']=tol['forecast_risk_probability']
        numerical=left['passed'] and right['passed']
        row['numerically_passed']=numerical
        row['complete_metrics']=all(np.isfinite(row[k]) for k in limits)
        for window in ('history','recent'):
            selected=[k for k in limits if not k.startswith(('history_','recent_')) or k.startswith(window+'_')]
            row[window+'_stable']=numerical and all(np.isfinite(row[k]) and row[k]<=limits[k] for k in selected)
        row['failed_criteria']=';'.join(k for k in limits if not np.isfinite(row[k]) or row[k]>limits[k])
        rows.append(row)
    return rows


def rectangles(pair_rows, scope="shared"):
    """Check all six pairs in each 2x2 rectangle, including both diagonals."""
    if pair_rows.empty:return pd.DataFrame()
    spec=specification();mapping={(c['A_level'],c['A_slope']):c['name'] for c in study_variants() if c['scope']==scope and c['calibration_kind']=='grid'}
    lookup={tuple(sorted((a,b))):g for (a,b),g in pair_rows.groupby(['left','right'])}
    rows=[]
    for a0,a1 in zip(spec['level_scales'][:-1],spec['level_scales'][1:]):
        for b0,b1 in zip(spec['slope_scales'][:-1],spec['slope_scales'][1:]):
            variants=[mapping[a,b] for a in (a0,a1) for b in (b0,b1)]
            groups=[lookup.get(tuple(sorted(p))) for p in combinations(variants,2)]
            complete=all(g is not None and set(g.channel)==set(CHANNELS) and len(g)==6 for g in groups)
            row=dict(scope=scope,A_level_lower=a0,A_level_upper=a1,A_slope_lower=b0,A_slope_upper=b1,
                cells=';'.join(variants),complete=complete)
            for window in ('history','recent'):
                row[window+'_candidate']=bool(complete and all(g[window+'_stable'].all() for g in groups))
            rows.append(row)
    return pd.DataFrame(rows)


def _tables(directory):
    result={name:read(directory/('sweetspot_'+file+'.csv')) for name,file in
        [('paths','paths'),('forecasts','forecasts'),('allocation','allocation'),('local','local_sensitivity')]}
    risks=[]
    for channel in CHANNELS:
        # This common, fixed threshold is present for each response in every cell.
        frame=read(directory/(channel+'_forecast_risk.csv'))
        if not frame.empty:risks.append(frame.assign(channel=channel))
    result['risk']=pd.concat(risks,ignore_index=True) if risks else pd.DataFrame()
    return result


def scale_learning(posterior):
    from scipy.stats import halfnorm
    rows=[];correlations=[];mapping={c['name']:c for c in study_variants()}
    for name,e in posterior.items():
        cell=mapping[name]
        for channel,directory in e['directories']:
            draws=read(directory/'sweetspot_draws.csv.gz')
            metadata=dict(variant=name,scope=cell['scope'],scale_owner=channel,
                A_level=cell['A_level'],A_slope=cell['A_slope'],A_season=cell['A_season'],
                numerically_passed=e['passed'])
            for component,key in [('level','A_level'),('slope','A_slope'),('seasonal','A_season')]:
                column='tau_'+component
                if column not in draws:continue
                lo,med,hi=draws[column].quantile([.025,.5,.975])
                plo,pmed,phi=halfnorm.ppf([.025,.5,.975],scale=cell[key])
                rows.append(dict(**metadata,component=component,
                    prior_lower=plo,prior_median=pmed,prior_upper=phi,posterior_lower=lo,
                    posterior_median=med,posterior_upper=hi,width_ratio=(hi-lo)/(phi-plo)))
            corr=read(directory/'sweetspot_correlations.csv')
            if not corr.empty:correlations.append(corr.assign(**metadata))
    return pd.DataFrame(rows),pd.concat(correlations,ignore_index=True) if correlations else pd.DataFrame()


def aggregate_posterior(parts):
    """Keep all six private reports; an incomplete bundle cannot pass a gate."""
    posterior={}
    for name,items in parts.items():
        owners=[owner for owner,e in items]
        if len(owners)!=len(set(owners)):raise ValueError('Duplicate posterior response: '+name)
        expected={'joint'} if items[0][1]['scope']=='shared' else set(CHANNELS)
        entry={}
        for key in ('paths','forecasts','allocation','local','risk'):
            frames=[e[key] for owner,e in items if not e[key].empty]
            entry[key]=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
        entry.update(passed=set(owners)==expected and all(e['passed'] for owner,e in items),
            complete=set(owners)==expected,scope=items[0][1]['scope'],
            directories=[(owner,e['directory']) for owner,e in items])
        posterior[name]=entry
    return posterior


def load_evidence(roots,tier,batch):
    records=[];parts={};validation=[]
    for task in tasks(batch,tier=tier):
        identity=dict(bucex_version=bx.__version__,**fingerprint(task_config(task,tier)))
        found=[]
        for root in roots:
            directory=Path(root)/tier/task.id;path=directory/'task.json'
            if path.exists():found.append((directory,bx.load_config(path)))
        if len(found)>1:
            raise ValueError(f'{task.id} occurs in multiple roots. Supply disjoint HPC/biobot outputs; remove the redundant root.')
        status='missing';numerical='unknown';directory=None;record={}
        if found:
            directory,record=found[0];status=record.get('status','unknown')
            if any(record.get(k)!=v for k,v in identity.items()):status='provenance_mismatch'
            numerical=record.get('numerical_status','unknown')
        passed=numerical=='passed_numerical_checks' and record.get('final_check')!='failed'
        item=dict(task_id=task.id,kind=task.kind,variant=task.variant,scope=task.scope,channel=task.channel,origin=task.origin,
            status=status,numerical_status=numerical,numerically_passed=passed,
            root=str(directory.parent.parent) if directory else '')
        if status=='completed':
            report=result_directory(directory,task)
            if task.kind=='posterior':
                entry=_tables(report)
                diagnostics=report/'sweetspot_convergence.json'
                extra=bx.load_config(diagnostics).get('status') if diagnostics.exists() else 'missing'
                passed=passed and extra=='passed_numerical_checks'
                item['numerically_passed']=passed;item['sweetspot_numerical_status']=extra
                entry.update(passed=passed,directory=report,scope=task.scope)
                parts.setdefault(task.variant,[]).append((task.channel,entry))
            else:
                validation.append(dict(task=task,report=report,passed=passed))
        records.append(item)
    return pd.DataFrame(records),aggregate_posterior(parts),validation


def validation_tables(entries):
    """Per-origin, per-response evidence; paired CRPS uses identical held-out cases."""
    scores=[];coverage=[];pits=[]
    for e in entries:
        metadata=dict(variant=e['task'].variant,training_end=e['task'].origin,numerically_passed=e['passed'])
        for filename,store in [('scores',scores),('coverage_by_case',coverage),('held_out_pit',pits)]:
            frame=read(e['report']/(filename+'.csv'))
            if not frame.empty:
                frame=frame.assign(**metadata)
                frame['horizon_band']=pd.cut(frame.horizon,[0,20,40,80,120],labels=['0–5y','5–10y','10–20y','20–30y'])
                store.append(frame)
    summaries={}
    keys=['variant','training_end','channel','horizon_band','numerically_passed']
    if scores:
        s=pd.concat(scores,ignore_index=True);s=s[s.score=='crps'].copy()
        summaries['crps']=s.groupby(keys,observed=True).agg(crps=('value','mean'),cases=('value','size'),distinct_dates=('time','nunique')).reset_index()
        ref=s[s.variant=='reference'];other=s[s.variant!='reference']
        merge=['training_end','channel','horizon','time']
        paired=other.merge(ref[merge+['value','numerically_passed']],on=merge,how='inner',validate='many_to_one',suffixes=('','_reference'))
        if not paired.empty:
            paired['difference']=paired.value-paired.value_reference
            paired['both_numerically_passed']=paired.numerically_passed & paired.numerically_passed_reference
            summaries['paired_crps']=paired.groupby(keys[:-1]+['both_numerically_passed'],observed=True).agg(
                crps=('value','mean'),reference_crps=('value_reference','mean'),difference=('difference','mean'),
                matched_cases=('value','size'),distinct_dates=('time','nunique')).reset_index()
            summaries['paired_crps']['ratio']=summaries['paired_crps'].crps/summaries['paired_crps'].reference_crps
    if coverage:
        c=pd.concat(coverage,ignore_index=True);c=c[(c.kind=='central_interval')&c.nominal.isin([.9,.95,.99])].copy()
        if not c.empty:
            c['covered_numeric']=c.covered.astype(str).str.lower().map({'true':1.,'false':0.,'1':1.,'0':0.})
            if c.covered_numeric.isna().any():raise ValueError('Unrecognized coverage indicator.')
            c['below']=c.observed<c.lower;c['above']=c.observed>c.upper
            summaries['coverage']=c.groupby(keys+['nominal'],observed=True).agg(coverage=('covered_numeric','mean'),
                width=('width','mean'),below=('below','mean'),above=('above','mean'),cases=('covered_numeric','size')).reset_index()
    if pits:summaries['pit_cases']=pd.concat(pits,ignore_index=True)
    return summaries


def figures(posterior,out,scope="shared"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    spec=specification();grid=[c for c in study_variants() if c['scope']==scope and c['calibration_kind']=='grid'];result=[]
    def save(fig,name,caption):
        name=scope+'_'+name
        fig.savefig(out/name,bbox_inches='tight',dpi=180);plt.close(fig);result.append((out/name,scope.capitalize()+': '+caption))
    with bx.publication_style(style='manuscript',dpi=180):
        for target,unit in [('level','°C'),('rate','°C/decade')]:
            for a in spec['level_scales']:
                selected=[c for c in grid if c['A_level']==a and c['name'] in posterior]
                if not selected:continue
                fig,axes=plt.subplots(3,2,figsize=(10,8),sharex=True)
                palette=plt.cm.viridis(np.linspace(.08,.9,len(spec['slope_scales'])))
                for ax,channel in zip(axes.flat,CHANNELS):
                    for c in selected:
                        entry=posterior[c['name']];f=entry['paths']
                        if f.empty:continue
                        f=f[(f.channel==channel)&(f.target==target)].sort_values('time')
                        x=pd.to_datetime(f.time);color=palette[spec['slope_scales'].index(c['A_slope'])]
                        ax.plot(x,f['median'],color=color,lw=1.3,ls='-' if entry['passed'] else '--',label=f"{c['A_slope']:g}")
                        ax.fill_between(x,f.lower,f.upper,color=color,alpha=.07,linewidth=0)
                    ax.axhline(0,color='.65',lw=.6);ax.set_ylabel(channel+' ('+unit+')')
                for ax in axes[-1]:ax.set_xlabel('Year')
                handles,labels=axes.flat[0].get_legend_handles_labels()
                if handles:fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.5,-.01))
                fig.tight_layout(rect=(0,.035,1,1))
                save(fig,f"{target}_Alevel_{a:g}.png",f"{target.capitalize()} trajectories, Aα={a:g}; colours identify Aβ in the legend. Medians and pointwise 95% intervals. Dashed curves have numerical flags; these are diagnostic plots, not final inferential figures.")
        for field,label in [('end_rate','Terminal rate (°C/decade)'),('rate_width','Terminal rate: 95% interval width (°C/decade)'),
                            ('slope_fraction','Slope fraction of new 30-year trend variance')]:
            matrices=[]
            for channel in CHANNELS:
                matrix=np.full((len(spec['level_scales']),len(spec['slope_scales'])),np.nan)
                for c in grid:
                    e=posterior.get(c['name'])
                    if e is None:continue
                    f=e['allocation'] if field=='slope_fraction' else e['paths']
                    if f.empty:continue
                    if field=='slope_fraction':f=f[(f.channel==channel)&(f.target=='slope_variance_fraction')&(f.horizon_years==30)]
                    else:f=f[(f.channel==channel)&(f.target=='rate')].sort_values('time').tail(1)
                    if f.empty:continue
                    r=f.iloc[-1];value=r.upper-r.lower if field=='rate_width' else r['median']
                    matrix[spec['level_scales'].index(c['A_level']),spec['slope_scales'].index(c['A_slope'])]=value
                matrices.append(matrix)
            finite=np.concatenate([m.ravel() for m in matrices]);finite=finite[np.isfinite(finite)]
            if not len(finite):continue
            lo,hi=(0,1) if field=='slope_fraction' else (float(finite.min()),float(finite.max()))
            if lo==hi:hi=lo+1e-6
            fig,axes=plt.subplots(3,2,figsize=(9,9),layout='constrained')
            for ax,channel,matrix in zip(axes.flat,CHANNELS,matrices):
                im=ax.imshow(matrix,origin='lower',aspect='auto',vmin=lo,vmax=hi,cmap='viridis')
                ax.set_xticks(range(len(spec['slope_scales'])),[f'{v:g}' for v in spec['slope_scales']],rotation=30)
                ax.set_yticks(range(len(spec['level_scales'])),[f'{v:g}' for v in spec['level_scales']])
                ax.set_xlabel('Aβ');ax.set_ylabel(channel+' · Aα')
                for i,j in np.ndindex(matrix.shape):
                    if np.isfinite(matrix[i,j]):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8,
                        color='white' if (matrix[i,j]-lo)/(hi-lo)<.48 else 'black')
            fig.colorbar(im,ax=list(axes.flat),shrink=.7,label=label)
            save(fig,field+'.png',label+'. Missing cells are blank. Completed cells are shown even when numerically flagged; see the task table before interpretation.')
    return result


def comparison_figures(posterior,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    variants={c['name']:c for c in study_variants()};plots=[]
    panels=[('pooling_reference',['reference','independent_reference'])]
    for scope in ('shared','independent'):
        panels.append(('seasonal_'+scope,[c['name'] for c in study_variants() if c['scope']==scope and
            (c['calibration_kind']=='seasonal' or c['setting']=='reference')]))
    with bx.publication_style(style='manuscript',dpi=180):
        for label,names in panels:
            names=[name for name in names if name in posterior]
            if not names:continue
            for target,unit in [('level','°C'),('rate','°C/decade')]:
                fig,axes=plt.subplots(3,2,figsize=(10,8),sharex=True)
                for ax,ch in zip(axes.flat,CHANNELS):
                    for i,name in enumerate(names):
                        e=posterior[name];f=e['paths']
                        if f.empty:continue
                        f=f[(f.channel==ch)&(f.target==target)].sort_values('time')
                        if f.empty:continue
                        legend=variants[name]['scope'] if label.startswith('pooling') else f"Aγ={variants[name]['A_season']:g}"
                        color=plt.cm.viridis(.15+.7*i/max(1,len(names)-1));x=pd.to_datetime(f.time)
                        ax.plot(x,f['median'],color=color,lw=1.3,ls='-' if e['passed'] else '--',label=legend)
                        ax.fill_between(x,f.lower,f.upper,color=color,alpha=.09,linewidth=0)
                    ax.set_ylabel(ch+' ('+unit+')');ax.axhline(0,lw=.5,color='.65')
                for ax in axes[-1]:ax.set_xlabel('Year')
                handles,labels=axes.flat[0].get_legend_handles_labels()
                if handles:fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
                fig.tight_layout(rect=(0,.04,1,1));path=out/(label+'_'+target+'.png')
                fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig)
                caption=('Matched pooled/private reference' if label.startswith('pooling') else 'Seasonal calibration at Aα=0.1, Aβ=0.002: '+label.split('_')[1])
                plots.append((path,caption+'. Medians and pointwise 95% intervals; dashed curves carry numerical flags.'))
    return plots


def build(root=ROOT,*,tier='screen',batch='sweetspot',other_roots=(),output=None):
    out=Path(output) if output else Path(root)/tier/'collected'/batch
    out.mkdir(parents=True,exist_ok=True)
    records,posterior,validation=load_evidence([root,*other_roots],tier,batch)
    records.to_csv(out/'tasks.csv',index=False)
    pd.DataFrame(calibration_rows()).to_csv(out/'grid_calibration.csv',index=False)
    allpairs=[]
    variant_map={c['name']:c for c in study_variants()}
    for left,right in combinations(sorted(posterior),2):
        a,b=variant_map[left],variant_map[right]
        if a['scope']!=b['scope'] and a['setting']!=b['setting']:continue
        kind='pooling' if a['scope']!=b['scope'] else 'seasonal' if a['calibration_kind']=='seasonal' or b['calibration_kind']=='seasonal' else 'grid'
        for row in compare_pair(posterior[left],posterior[right]):allpairs.append(dict(left=left,right=right,comparison=kind,**row))
    pairs=pd.DataFrame(allpairs);pairs.to_csv(out/'all_pairwise_stability.csv',index=False)
    regions=pd.concat([rectangles(pairs,scope) for scope in ('shared','independent')],ignore_index=True);regions.to_csv(out/'stability_regions.csv',index=False)
    pooling=pairs[pairs.comparison=='pooling'] if not pairs.empty else pd.DataFrame()
    seasonal=pairs[pairs.comparison=='seasonal'] if not pairs.empty else pd.DataFrame()
    pooling.to_csv(out/'matched_pooling_comparisons.csv',index=False)
    seasonal.to_csv(out/'seasonal_comparisons.csv',index=False)
    summaries=validation_tables(validation)
    for name,frame in summaries.items():frame.to_csv(out/(name+'.csv'),index=False)
    learning,correlations=scale_learning(posterior)
    learning.to_csv(out/'scale_learning.csv',index=False)
    correlations.to_csv(out/'allocation_correlations.csv',index=False)
    plots=figures(posterior,out,'shared')+figures(posterior,out,'independent')+comparison_figures(posterior,out)
    status=dict(version=bx.__version__,tier=tier,batch=batch,expected=len(records),completed=int(records.status.eq('completed').sum()),
        numerically_passed=int((records.status.eq('completed')&records.numerically_passed).sum()),
        posterior_cells=len(posterior),validation_fits=len(validation),
        history_candidate_rectangles=int(regions.history_candidate.sum()) if not regions.empty else 0,
        recent_candidate_rectangles=int(regions.recent_candidate.sum()) if not regions.empty else 0,
        decision='Descriptive candidates only; convergence, finite-change stability and predictive checks must be reviewed together.',
        tolerances=specification()['tolerances'])
    bx.save_config(status,out/'status.json')
    def table(f):return '<p>No results available yet.</p>' if f.empty else f.to_html(index=False,border=0,float_format=lambda x:f'{x:.4g}')
    text=[f'<h1>BUCEX {escape(bx.__version__)} · joint level–slope calibration</h1>',
        f'<p>{status["completed"]}/{len(records)} fits completed; {status["numerically_passed"]} passed the declared numerical checks. '
        f'{len(posterior)}/22 full-record calibration/scope combinations have at least one response available. Tier: <strong>{escape(tier)}</strong>.</p>',
        '<p>This study keeps both level and slope innovations. It asks whether a region of half-normal scale settings yields stable trajectories, uncertainty and forecasts. '
        'It does not select the smallest CRPS or prove that estimates are independent of priors. Hyperpriors remain proper and calibrated; widening them is an experiment, not an assumed remedy.</p>',
        '<h2>What would count as a stable region?</h2>',
        '<p>A candidate comprises four adjacent grid cells. Every one of their six pairwise comparisons must meet all tolerances for all six responses, '
        'including both diagonals. We compare trajectory medians and the two 95% interval limits; 10- and 30-year forecast medians and limits; '
        'fixed-threshold risk means over the forecast; and medians and limits of the slope contribution to new trend variance. '
        'Missing quantities or numerical flags prevent a pass. Historical and final-30-year stability are reported separately. '
        'A recent-only pass does not establish stability of the complete slope history. Even a paper-tier pass remains a candidate for scientific review.</p>',
        table(pd.DataFrame(list(status['tolerances'].items()),columns=['Criterion','Maximum difference'])),table(regions),
        '<p>If there is no candidate region, this is evidence to inspect—not a reason to keep expanding the grid automatically. '
        'Check Monte Carlo error, inspect posterior scale/allocation correlations and local sensitivities, and distinguish a stable total trend from an uncertain level/slope decomposition. '
        'Longer chains can resolve Monte Carlo uncertainty; they cannot supply identification missing from the data.</p>',
        '<h2>Prior calibration</h2>',
        '<p>Entries below integrate both hierarchy levels: s|τ ~ Normal(0,τ²), τ ~ HalfNormal(A). Thus E[s²]=A². '
        'These are marginal prior RMS effects of future innovations, not central 95% bounds or the full prior-predictive spread. '
        'At h seasonal steps, new level variance is h qα and new slope variance is h(h−1)(2h−1)qβ/6. '
        'The allocation fraction excludes current-state uncertainty, seasonality and observation noise. Exported total-level variances include the posterior covariance of terminal level and slope.</p>',
        table(pd.DataFrame(calibration_rows())), '<h2>Trajectories and decomposition</h2>']
    for path,caption in plots:
        encoded=base64.b64encode(path.read_bytes()).decode()
        text.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{escape(caption)}"><figcaption>{escape(caption)}</figcaption></figure>')
    if not plots:text.append('<p>Trajectory figures will appear as full-record fits complete.</p>')
    text.extend(['<h2>Matched pooling comparisons</h2>', '<p>The full record uses all 11 calibrations in both scopes. Private fits retain half-normal shrinkage with separate scales for each response; they are not unregularized fits. One-response priors match exactly. Private validation covers the reference at all five origins. A partial private bundle cannot pass a stability check.</p>',table(pooling),
        '<h2>Seasonal scale checks</h2>', '<p>The 3×3 level–slope grid fixes Aγ=0.1. Checks at Aγ=0.05 and 0.2 use Aα=0.1 and Aβ=0.002. This is 11 settings, not a 3×3×3 factorial. Grid heatmaps exclude the two seasonal checks; their comparisons are shown separately.</p>',table(seasonal),
        '<h2>Shared and private scales and allocation correlations</h2>',
        '<p>The scale table contrasts analytic half-normal prior intervals with posterior intervals. '
        'A width ratio below one means the posterior 95% interval is narrower; this alone does not establish robustness or identify the level/slope decomposition. '
        'Compare posterior movement across cells and the trajectory/forecast checks. Correlations use paired draws; a small linear correlation does not rule out nonlinear dependence.</p>',
        table(learning),table(correlations),'<p>Each fit also exports paired parameter draws and local sensitivity of posterior means: '
        'd E[f|y]/d log(Ac) = Cov(f, τc²/Ac² | y). These covariances describe small local changes; '
        'they do not replace the refits. Compare chain-specific estimates before interpreting them.</p>',
        '<h2>Matched hindcasts</h2>',
        '<p>Every grid cell is refitted at the same origins. Hyperparameters are learned using training data only. '
        'The requested horizon is 30 years, truncated where observations end; there is no observed 30-year assessment from 2020. '
        'CRPS comparisons use the same response, origin, date and horizon. Positive differences and ratios above one favour the reference; '
        'negative differences and ratios below one favour the alternative. Coverage and interval width are assessed together, with above/below misses shown. '
        'Horizon bands are disjoint within each origin, but windows from different origins overlap. Counts are forecast cases, not independent replications; '
        'no significance claims are made. These folds are used for calibration, so they are not an untouched final test set.</p>'])
    for name in ('paired_crps','crps','coverage'):
        text.extend([f'<h3>{escape(name.replace("_"," "))}</h3>',table(summaries.get(name,pd.DataFrame()))])
    text.extend(['<h2>Completion and numerical checks</h2>',table(records),
        '<p>Screen: two chains, 1,000 warmup and 2,000 retained draws per chain. These are screening budgets, not a promise of adequate effective sample sizes. '
        'Rerun promising regions with the paper budget and assess slope-specific diagnostics before drawing conclusions. '
        'The full-record initial seasonal contrast SD stays at the 1.9.6 value of 10; initial-rate SD stays 0.01 per season. '
        'The seasonal shrinkage hyperprior scale is 0.1 in the grid, with 0.05 and 0.2 checks at its centre. Initial rates remain separate. Residual independence is unchanged.</p>'])
    html='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+\
        '<title>BUCEX 1.9.8 calibration review</title><style>body{font:16px/1.6 system-ui,sans-serif;color:#253441;max-width:1180px;margin:40px auto;padding:0 24px}h1,h2{color:#24658a}table{display:block;overflow:auto;border-collapse:collapse;font-size:12px;margin:20px 0;max-height:620px}td,th{padding:6px 9px;border-bottom:1px solid #dce3e7;text-align:left;white-space:nowrap}th{background:#edf3f6;position:sticky;top:0}img{width:100%;height:auto}figure{margin:30px 0}figcaption{font-size:14px;color:#53616a}p{max-width:1000px}</style><body>'+''.join(text)+'</body></html>'
    (out/'sweetspot_review.html').write_text(html)
    print(json.dumps(status,indent=2),flush=True)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--other-root',type=Path,action='append',default=[])
    p.add_argument('--tier',choices=('screen','paper'),default='screen');p.add_argument('--batch',choices=BATCHES,default='sweetspot')
    p.add_argument('--output',type=Path)
    a=p.parse_args();build(a.root,tier=a.tier,batch=a.batch,other_roots=a.other_root,output=a.output)
