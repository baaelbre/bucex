"""Collect origin-by-horizon validation without treating overlapping cases as independent."""
import base64
from datetime import datetime, timezone
from html import escape
from itertools import combinations
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import bucex as bx
from research.seasonal.horizon_plan import specification
from research.seasonal.horizon_diagnostics import annotate, windows


def metrics(frame):
    error = frame.predictive_mean - frame.observed
    result = dict(n_cases=len(frame), n_unique_times=frame.time.nunique(),
        crps=float(frame.crps.mean()), bias=float(error.mean()),
        mae=float(error.abs().mean()), rmse=float(np.sqrt((error**2).mean())))
    for percent, low, high in ((90,'q050','q950'),(95,'q025','q975'),(99,'q005','q995')):
        lower = frame.observed < frame[low]
        upper = frame.observed > frame[high]
        result.update({f'coverage{percent}':float((~(lower | upper)).mean()),
            f'width{percent}':float((frame[high]-frame[low]).mean()),
            f'lower_misses{percent}':int(lower.sum()), f'upper_misses{percent}':int(upper.sum())})
    return result


def summarize(cases, spec):
    rows = []
    for (variant, origin, name), frame in cases.groupby(['variant','origin_date','channel']):
        for kind, label, mask, expected in windows(frame, spec):
            for season in ('ALL','DJF','MAM','JJA','SON'):
                selected = frame.loc[mask & (True if season == 'ALL' else frame.season.eq(season))]
                if selected.empty:
                    continue
                wanted = len(selected) if kind == 'available' else expected if season == 'ALL' else expected//4
                rows.append(dict(variant=variant,origin_date=origin,channel=name,window_kind=kind,
                    window=label,season=season,n_requested=wanted,complete=len(selected)==wanted,
                    first_lead=int(selected.horizon.min()),last_lead=int(selected.horizon.max()),
                    numerical_status=','.join(sorted(selected.numerical_status.unique())),**metrics(selected)))
    return pd.DataFrame(rows)


def paired_models(cases, spec):
    """Compare models only on the same origin, response and held-out observations."""
    rows = []
    ref = cases.loc[cases.variant.eq('reference')]
    fixed = cases.loc[cases.variant.eq('fixed_reference')]
    keys = ['origin_date','channel','time','horizon']
    pairs = ref.merge(fixed,on=keys,suffixes=('_reference','_fixed'),validate='one_to_one')
    if pairs.empty:
        return pd.DataFrame(columns=['origin_date','channel','window_kind','window','season','n_cases',
            'crps_fixed_minus_pooled','bias_pooled','bias_fixed'])
    if not np.allclose(pairs.observed_reference,pairs.observed_fixed):
        raise ValueError('Model comparison observations do not match.')
    pairs = annotate(pairs)
    for (origin,name), frame in pairs.groupby(['origin_date','channel']):
        for kind,label,mask,expected in windows(frame,spec):
            for season in ('ALL','DJF','MAM','JJA','SON'):
                f = frame.loc[mask & (True if season == 'ALL' else frame.season.eq(season))]
                if f.empty:
                    continue
                rows.append(dict(origin_date=origin,channel=name,window_kind=kind,window=label,
                    season=season,n_cases=len(f),
                    crps_fixed_minus_pooled=float((f.crps_fixed-f.crps_reference).mean()),
                    bias_pooled=float((f.predictive_mean_reference-f.observed_reference).mean()),
                    bias_fixed=float((f.predictive_mean_fixed-f.observed_fixed).mean())))
    return pd.DataFrame(rows)


def paired_origins(cases, spec):
    """Separate calendar conditions from lead time, using paired verification cases."""
    rows = []
    for (variant,name), data in cases.groupby(['variant','channel']):
        for earlier,later in combinations(sorted(data.origin_date.unique()),2):
            pair = data.loc[data.origin_date.eq(earlier)].merge(
                data.loc[data.origin_date.eq(later)],on='time',suffixes=('_early','_late'),validate='one_to_one')
            if pair.empty:
                continue
            if not np.allclose(pair.observed_early,pair.observed_late):
                raise ValueError('Origin comparison observations do not match.')
            for lo,hi in spec['calendar_windows']:
                in_year = pair.meteorological_year_early.between(lo,hi)
                for season in ('ALL','DJF','MAM','JJA','SON'):
                    f = pair.loc[in_year & (True if season == 'ALL' else pair.season_early.eq(season))]
                    if f.empty:
                        continue
                    rows.append(dict(variant=variant,channel=name,earlier_origin=earlier,later_origin=later,
                        calendar_window=f'{lo}-{hi}',season=season,n_cases=len(f),
                        complete=len(f)==(hi-lo+1)*(4 if season=='ALL' else 1),
                        earlier_first_lead=int(f.horizon_early.min()),earlier_last_lead=int(f.horizon_early.max()),
                        later_first_lead=int(f.horizon_late.min()),later_last_lead=int(f.horizon_late.max()),
                        crps_later_minus_earlier=float((f.crps_late-f.crps_early).mean()),
                        bias_earlier=float((f.predictive_mean_early-f.observed_early).mean()),
                        bias_later=float((f.predictive_mean_late-f.observed_late).mean())))
    return pd.DataFrame(rows,columns=['variant','channel','earlier_origin','later_origin',
        'calendar_window','season','n_cases','complete','earlier_first_lead','earlier_last_lead',
        'later_first_lead','later_last_lead','crps_later_minus_earlier','bias_earlier','bias_later'])


def figures(cases, summary, counts, out, spec):
    out.mkdir(parents=True,exist_ok=True)
    generated = []
    plt.rcParams.update({'font.size':9, 'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name,caption):
        fig.savefig(out/(name+'.png'),dpi=150,bbox_inches='tight')
        fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
        plt.close(fig)
        generated.append((out/(name+'.png'),caption))
    for variant,data in cases.groupby('variant'):
        origins = sorted(data.origin_date.unique())
        shown = summary.loc[summary.variant.eq(variant)&summary.window_kind.eq('lead_band')&summary.season.eq('ALL')]
        labels = [f'{lo:02d}-{hi:02d} years' for lo,hi in spec['lead_bands_years']]
        for field,title in (('crps','CRPS (smaller is better)'),('bias','Forecast mean minus observation (degrees C)'),('coverage95','Central 95% coverage')):
            fig,axes = plt.subplots(2,3,figsize=(13,max(6,len(origins)*.60)),layout='constrained')
            for name,ax in zip(spec['series'],axes.flat):
                part = shown.loc[shown.channel.eq(name)]
                matrix = part.pivot(index='origin_date',columns='window',values=field).reindex(index=origins,columns=labels)
                finite = np.abs(matrix.to_numpy()[np.isfinite(matrix.to_numpy())])
                limit = float(finite.max()) if len(finite) else 1.
                kwargs = dict(vmin=-max(limit,.1),vmax=max(limit,.1),cmap='RdBu_r') if field=='bias' else dict(vmin=0,vmax=1,cmap='viridis') if field=='coverage95' else dict(vmin=0,vmax=max(limit,.1),cmap='YlOrRd')
                im = ax.imshow(matrix,aspect='auto',**kwargs)
                ax.set_title(name,fontweight='bold');ax.set_xticks(range(3),['1–10','11–20','21–30'])
                ax.set_yticks(range(len(origins)),[o[:4] for o in origins]);ax.set_xlabel('Forecast years')
                lookup = part.set_index(['origin_date','window'])
                for i,origin in enumerate(origins):
                    for j,label in enumerate(labels):
                        if (origin,label) not in lookup.index:
                            continue
                        row = lookup.loc[(origin,label)]
                        value = f'{row[field]:.0%}' if field=='coverage95' else f'{row[field]:.2f}'
                        ax.text(j,i,value+(' *' if not row.complete else '')+f'\nn={int(row.n_cases)}',ha='center',va='center',fontsize=8,
                            bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1))
                fig.colorbar(im,ax=ax,shrink=.7)
            fig.suptitle(variant+': '+title)
            save(fig,variant+'_'+field,title+'. Each cell uses its own origin and lead band. * denotes an incomplete band; n is the number of seasonal verification cases.')
        for name,season in (('TXx','JJA'),('TNn','DJF')):
            if name not in set(data.channel):
                continue
            selected_origins = [o for o in spec['priority_origins']+['2010-11','2015-11','2020-11'] if o in origins]
            if not selected_origins:
                selected_origins=origins[:6]
            ncols=min(3,len(selected_origins));nrows=int(np.ceil(len(selected_origins)/ncols))
            fig,axes = plt.subplots(nrows,ncols,squeeze=False,
                figsize=(max(7,4.4*ncols),3.6*nrows),layout='constrained')
            for ax,origin in zip(axes.flat,selected_origins):
                f = data.loc[data.channel.eq(name)&data.season.eq(season)&data.origin_date.eq(origin)].sort_values('time')
                if f.empty:
                    continue
                x = f.meteorological_year.to_numpy()
                ax.fill_between(x,f.q025.to_numpy(),f.q975.to_numpy(),color='#21688d',alpha=.17,label='Observation: central 95% PI')
                ax.plot(x,f.predictive_mean,color='#21688d',lw=1.3,label='Predictive mean')
                ax.fill_between(x,f.location_lower95.to_numpy(),f.location_upper95.to_numpy(),color='#ac4837',alpha=.23,label='Location: central 95% interval')
                ax.plot(x,f.location_mean,color='#ac4837',lw=1.2,label='Location mean')
                ax.plot(x,f.q990,color='#333333',lw=1,ls=':',label='99th predictive percentile')
                ax.scatter(x,f.observed,s=13,color='black',zorder=5,label='Observed')
                ax.set_title(f'{origin}: {name}, {season}');ax.set_xlabel('Meteorological year');ax.set_ylabel('Temperature / °C')
                ax.grid(axis='y',alpha=.2)
            for ax in list(axes.flat)[len(selected_origins):]:ax.set_visible(False)
            handles,labels_ = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles,labels_,loc='upper center',bbox_to_anchor=(.5,0),ncol=min(3,ncols+1),fontsize=8)
            save(fig,variant+'_'+name+'_forecasts','Forecasts remain fixed at the displayed origin; no later observations are assimilated. The location includes seasonality. Central 95% observation intervals and the 99th predictive percentile come from the mixture CDF.')
        for origin,part in data.groupby('origin_date'):
            fig,axes=plt.subplots(2,3,figsize=(11,5),layout='constrained')
            for name,ax in zip(spec['series'],axes.flat):
                f=part.loc[part.channel.eq(name)]
                if f.empty:
                    ax.set_visible(False);continue
                ax.hist(f.pit,bins=np.linspace(0,1,11),density=True,color='#21688d',alpha=.7)
                ax.axhline(1,color='black',lw=1,ls='--');ax.set_title(f'{name}, n={len(f)}');ax.set_xlabel('Forecast PIT')
            fig.suptitle(f'{variant}: origin {origin}')
            save(fig,variant+'_pit_'+origin,'Forecast PITs for this origin over its observed verification window. Horizons and seasons are also retained in the case-level CSV; these histograms are descriptive.')
    if not counts.empty:
        part=counts.loc[counts.channel.eq('TXx')&counts.season.eq('JJA')&counts.threshold.eq(35)&counts.window_kind.eq('available')]
        for variant,f in part.groupby('variant'):
            f=f.sort_values('origin_date');x=np.arange(len(f))
            fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
            ax.vlines(x,f.count_lower95,f.count_upper95,color='#21688d',alpha=.6,lw=3,label='95% predictive count interval')
            ax.scatter(x,f.expected_events,color='#21688d',label='Expected count')
            ax.scatter(x,f.observed_events,color='black',marker='D',label='Observed count')
            ax.set_xticks(x,[f'{o[:4]}\nn={n}' for o,n in zip(f.origin_date,f.n_cases)])
            ax.set_xlabel('Forecast origin; observed summers');ax.set_ylabel('Summers with TXx > 35 °C')
            ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
            save(fig,variant+'_summer_counts','Expected and observed exceedance counts. Each origin has its own verification period and number of summers; these counts are not independent replications.')
    return generated


def finish(root,tier,batch='horizon_reference',require_complete=False):
    from research.seasonal.jobs import tasks,task_config,fingerprint,result_directory
    root=Path(root).resolve();out=root/tier/('validation_report_'+batch);out.mkdir(parents=True,exist_ok=True)
    # Regenerated evidence must not retain stale results if a later collection
    # finds missing outputs or incompatible provenance. Leave other files alone.
    for name in ('figures','diagnostics'):
        if (out/name).exists():shutil.rmtree(out/name)
    for name in ('validation_cases.csv','risk_counts.csv','risk_cases.csv','risk_count_pmf.csv',
                 'folds.csv','origin_horizon_metrics.csv','same_calendar_comparisons.csv',
                 'pooled_vs_fixed_paired.csv','year_by_year_metrics.csv','report.html'):
        (out/name).unlink(missing_ok=True)
    spec=specification();status_rows=[];case_parts=[];risk_parts=[];risk_case_parts=[];pmf_parts=[];fold_parts=[]
    for task in tasks(batch,tier=tier):
        directory=root/tier/task.id;marker=directory/'task.json'
        record=json.loads(marker.read_text()) if marker.exists() else {}
        state=record.get('status','pending');reason=''
        if state=='completed':
            expected=dict(bucex_version=bx.__version__,**fingerprint(task_config(task,tier)))
            if any(record.get(k)!=v for k,v in expected.items()):
                state='provenance_mismatch';reason='Source, data or configuration differs from this release.'
        meta=dict(task_id=task.id,variant=task.variant,origin_date=task.origin,
                  numerical_status=record.get('numerical_status','not_available'))
        if state=='completed':
            target=result_directory(directory,task)
            required=['forecast_paths.csv','scores.csv','held_out_pit.csv','folds.csv','risk_count_windows.csv','risk_count_pmf.csv','risk_cases.csv']
            missing=[name for name in required if not (target/name).is_file()]
            if missing:
                state='missing_outputs';reason=', '.join(missing)
            else:
                paths=pd.read_csv(target/'forecast_paths.csv',parse_dates=['time'])
                score=pd.read_csv(target/'scores.csv',parse_dates=['time'])
                pit=pd.read_csv(target/'held_out_pit.csv',parse_dates=['time'])
                key=['channel','time','horizon']
                paths=paths.merge(score.loc[score.score.eq('crps'),key+['value']].rename(columns={'value':'crps'}),on=key,validate='one_to_one')
                paths=paths.merge(pit[key+['pit']],on=key,validate='one_to_one')
                if len(paths)!=len(pd.read_csv(target/'forecast_paths.csv')):
                    raise ValueError('Missing CRPS/PIT cases for '+task.id)
                case_parts.append(annotate(paths.assign(**meta)))
                risk_parts.append(pd.read_csv(target/'risk_count_windows.csv').assign(**meta))
                risk_case_parts.append(pd.read_csv(target/'risk_cases.csv').assign(**meta))
                pmf_parts.append(pd.read_csv(target/'risk_count_pmf.csv').assign(**meta))
                fold_parts.append(pd.read_csv(target/'folds.csv').assign(**meta))
                diagnostic_dir=out/'diagnostics'/task.id;diagnostic_dir.mkdir(parents=True,exist_ok=True)
                for pattern in ('mcmc_*.csv','targets_*.csv','convergence_*.json','execution_*.json'):
                    for path in target.glob(pattern):shutil.copy2(path,diagnostic_dir/path.name)
                for filename in ('task.json','resolved_config.json'):
                    if (directory/filename).is_file():shutil.copy2(directory/filename,diagnostic_dir/filename)
        status_rows.append(dict(**meta,status=state,reason=reason))
    statuses=pd.DataFrame(status_rows);statuses.to_csv(out/'task_status.csv',index=False)
    status=dict(version=bx.__version__,tier=tier,batch=batch,expected=len(statuses),
        completed=int(statuses.status.eq('completed').sum()),
        numerically_passed=int((statuses.status.eq('completed')&statuses.numerical_status.eq('passed_numerical_checks')).sum()))
    bx.save_config(status,out/'status.json');bx.save_config(spec,out/'validation_design.json')
    images=[];summaries=pd.DataFrame();counts=pd.DataFrame();pairs=pd.DataFrame()
    if case_parts:
        cases=pd.concat(case_parts,ignore_index=True);cases.to_csv(out/'validation_cases.csv',index=False)
        counts=pd.concat(risk_parts,ignore_index=True);counts.to_csv(out/'risk_counts.csv',index=False)
        pd.concat(risk_case_parts,ignore_index=True).to_csv(out/'risk_cases.csv',index=False)
        pd.concat(pmf_parts,ignore_index=True).to_csv(out/'risk_count_pmf.csv',index=False)
        pd.concat(fold_parts,ignore_index=True).to_csv(out/'folds.csv',index=False)
        summaries=summarize(cases,spec);summaries.to_csv(out/'origin_horizon_metrics.csv',index=False)
        calendar=paired_origins(cases,spec);calendar.to_csv(out/'same_calendar_comparisons.csv',index=False)
        pairs=paired_models(cases,spec);pairs.to_csv(out/'pooled_vs_fixed_paired.csv',index=False)
        year_keys=['variant','channel','origin_date','season','lead_year']
        pd.DataFrame([dict(zip(year_keys,key),**metrics(f)) for key,f in cases.groupby(year_keys)]).to_csv(
            out/'year_by_year_metrics.csv',index=False)
        images=figures(cases,summaries,counts,out/'figures',spec)
    def table(frame):
        return frame.to_html(index=False,float_format=lambda x:f'{x:.4g}',border=0) if not frame.empty else '<p>No completed evidence yet.</p>'
    chunks=['<!doctype html><html><head><meta charset="utf-8"><title>BUCEX 1.9.8.4 validation</title>',
        '<style>body{font:16px/1.5 system-ui;max-width:1200px;margin:36px auto;padding:0 20px;color:#20303e}h1,h2{line-height:1.2}table{border-collapse:collapse;font-size:12px;white-space:nowrap}th,td{padding:5px 8px;border-bottom:1px solid #ddd;text-align:right}th{background:#edf2f6}img{max-width:100%}.scroll{overflow:auto;margin:16px 0}figure{margin:26px 0}figcaption{font-size:14px}code{background:#edf2f6}details{margin:18px 0}</style></head><body>',
        f'<h1>BUCEX 1.9.8.4 — validation across origins and horizons</h1><p>Tier: <b>{escape(tier)}</b>. Completed {status["completed"]}/{status["expected"]} fits; {status["numerically_passed"]} pass this tier’s numerical criteria.</p>',
        '<p>At each November origin the model is refitted to the expanding record starting in 1892. Parameters and states use only the training data. New state innovations and observations are simulated forward, without later updating. One fit supplies every reported horizon.</p>',
        '<p>The fixed calibration is pooled half-normal (level 0.1, slope 0.002, seasonal 0.1), with separate initial-slope SD 0.01 per season. The optional fixed-Normal comparison uses the same coefficient second moments and its own trajectories; it has no learned shared scales.</p>',
        '<p>This design and calibration were chosen after examining the full record. These are retrospective forecast experiments, not untouched prospective validation. Screening diagnostics do not establish publication-quality convergence; fits marked needs_review remain visible.</p>',
        '<p><b>Reading the results.</b> Bias is forecast mean minus observation: negative values mean underprediction. CRPS is smaller when better. Coverage is reported with interval width and lower/upper misses. All central uncertainty intervals are 95%, except explicitly named 90% and 99% coverage checks. Predictive quantiles use the mixture CDF. Forecast and risk point summaries are means.</p>',
        '<p>Each lead band covers forecast years 1–10, 11–20 or 21–30. Cumulative windows cover 1, 5, 10, 20 and 30 years. Missing future observations are never scored. The 1996 origin has 119 of 120 seasons through JJA 2026. Partial windows are flagged and must not be described as complete 30-year validation. Overlapping origins, successive observations and different summaries are not independent validation replicates; no binomial calibration test or effective-sample-size claim is made from their raw counts.</p>',
        '<h2>Run status</h2><div class="scroll">',table(statuses),'</div>']
    if not summaries.empty:
        view=summaries.loc[summaries.window_kind.eq('lead_band')&summaries.season.eq('ALL')]
        fields=['variant','origin_date','channel','window','n_cases','n_requested','complete','crps','bias','coverage90','coverage95','coverage99','width95','lower_misses95','upper_misses95','numerical_status']
        chunks+=['<h2>Origin and horizon summary</h2><div class="scroll">',table(view[fields]),'</div>']
        summer=counts.loc[counts.channel.eq('TXx')&counts.season.eq('JJA')&counts.threshold.eq(35)&counts.window_kind.eq('available')]
        chunks+=['<h2>Summer threshold check</h2><p>Expected counts sum posterior predictive event probabilities. The 95% predictive count interval and the probability of at least the observed count use complete replicated paths, preserving dependence from shared uncertain parameters and state evolution. They are simulation estimates; zero is not proof of zero event probability.</p><div class="scroll">',
            table(summer[['variant','origin_date','n_cases','observed_events','expected_events','count_lower95','count_upper95','probability_at_least_observed','numerical_status']]),'</div>',
            '<h2>Matched comparisons</h2><p><code>same_calendar_comparisons.csv</code> compares earlier and later origins on identical verification seasons. <code>pooled_vs_fixed_paired.csv</code>, when both batches are available, compares the two specifications on identical origin/response/date cases. Negative CRPS differences favour the later origin or fixed-Normal model, respectively.</p>']
    for path,caption in images:
        embedded=base64.b64encode(path.read_bytes()).decode('ascii')
        chunks += ['<details open><summary>'+escape(path.stem)+'</summary><figure><img src="data:image/png;base64,'+embedded+'" alt="'+escape(path.stem)+'"><figcaption>'+escape(caption)+'</figcaption></figure></details>']
    chunks += ['<p>CSV files retain the denominators, season labels, individual cases, complete/partial flags and numerical status. PNG and PDF figures are in the figures folder. Posterior archives remain in each task directory and are not included in the compact review ZIP.</p></body></html>']
    (out/'report.html').write_text(''.join(chunks),encoding='utf-8')
    exports=root/'exports';exports.mkdir(exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    archive=exports/f'bucex1984_{tier}_{batch}_{stamp}.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in out.rglob('*'):
            if path.is_file():bundle.write(path,path.relative_to(out.parent))
    print(f'{out / "report.html"}\n{archive}',flush=True)
    if require_complete and (status['completed']!=status['expected'] or status['numerically_passed']!=status['expected']):
        raise RuntimeError('Incomplete or numerically unchecked validation; inspect task_status.csv.')
    return out
