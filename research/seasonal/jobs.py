"""BUCEX 1.9.3: one response per task, with no cross-response borrowing."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import traceback
import numpy as np
import bucex as bx
from research.monthly.experiment import configured_variant
from research.monthly.run import run as fit_full
from research.monthly.validate import validate

PROJECT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT/'research/seasonal/config'
ROOT = Path('results/serra_193_parallel')
BATCHES = ('posterior','reference','pre2019','experiments','validation','validation10','all')
STUDIES = ('manuscript_sensitivity','physical_sensitivity','adequacy')

@dataclass(frozen=True)
class Task:
    id: str
    kind: str
    study: str
    variant: str
    channel: str
    origin: str = ''
    horizon: int = 20
    design: str = 'calendar_5y'


def settings():
    return bx.load_config(CONFIG/'experiments.json')


def resource_class(task, tier):
    return ('long' if tier == 'paper' and task.variant == 'reference'
            and task.kind in ('posterior','pre2019') else 'standard')


@lru_cache(maxsize=1)
def _fold_dates():
    return bx.load_uccle_multiseries(**bx.load_config(CONFIG/'main.json')['data']).index


def tasks(batch='posterior', *, tier='paper', resource='all'):
    """Stable IDs and ordering; each full experiment contains all six responses."""
    if batch not in BATCHES or tier not in ('screen','paper') or resource not in ('all','standard','long'):
        raise ValueError('Unknown batch, tier or resource class.')
    spec=settings();variants={}
    for study in spec['studies']:
        for variant in bx.load_config(CONFIG/f'{study}.json')['variants']:
            name=variant['name']
            if name=='reference' and name in variants:continue
            if name in variants:raise ValueError(f'Duplicate variant: {name}')
            variants[name]=study
    def channels(variant):
        return list(spec['series'])
    posterior=[Task(f'posterior_{v}_{ch}','posterior',s,v,ch)
               for v,s in variants.items() for ch in channels(v)]
    record=[Task(f'pre2019_{v["name"]}_TXx','pre2019','pre2019_sensitivity',v['name'],'TXx')
            for v in bx.load_config(CONFIG/'pre2019_sensitivity.json')['variants']]
    forecasts=[Task(f'forecast_reference_{ch}_{o}','forecast',variants['reference'],'reference',ch,o)
               for o in spec['reference_validation_origins'] for ch in channels('reference')]
    for v in spec['validation_variants']:
        if v=='reference' or v not in variants:raise ValueError(f'Invalid validation variant: {v}')
        forecasts.extend(Task(f'forecast_{v}_{ch}_{o}','forecast',variants[v],v,ch,o)
                         for o in spec['sensitivity_validation_origins'] for ch in channels(v))
    import pandas as pd
    dates=_fold_dates()
    original=[]
    for fraction in spec['original_validation_fractions']:
        n=int(np.floor(len(dates)*fraction))
        origin=(dates[n-1]+pd.DateOffset(months=3)-pd.Timedelta(days=1)).strftime('%Y-%m')
        original.extend(Task(f'forecast_reference_{ch}_split{round(100*fraction)}_h40',
            'forecast',variants['reference'],'reference',ch,origin,
            spec['original_validation_horizon'],'fraction_10y') for ch in channels('reference'))
    all_tasks=posterior+record+forecasts+original
    if len({t.id for t in all_tasks})!=len(all_tasks):raise ValueError('Duplicate task IDs.')
    chosen={'posterior':posterior,'reference':[t for t in posterior if t.variant=='reference'],
            'pre2019':record,'experiments':posterior+record,'validation':forecasts,
            'validation10':original,'all':all_tasks}[batch]
    return [t for t in chosen if resource=='all' or resource_class(t,tier)==resource]


def task_config(task,tier):
    if tier not in ('screen','paper'):raise ValueError('Unknown tier.')
    source=bx.load_config(CONFIG/f'{task.study}.json')
    entry=next(v for v in source['variants'] if v['name']==task.variant)
    c=configured_variant(source,entry);c.pop('variants',None);c['variant']=deepcopy(entry)
    budget=settings()['tiers'][tier]
    c['mcmc'].update(budget['mcmc']);c['diagnostic_thresholds']=deepcopy(budget['diagnostic_thresholds'])
    for field in ('forecast_draws','predictive_check_draws','save_fits','figures','trace_exports'):c[field]=budget[field]
    c['validation'].update(draws=budget['validation_draws'],save_fits=budget['save_fits'])
    if task.kind=='forecast':c['validation'].update(training_ends=[task.origin],horizon=task.horizon)
    elif task.kind=='pre2019':
        c['forecast_horizon']=1
        if tier=='paper':c['forecast_draws']=10000
    else:c['forecast_horizon']=120
    if resource_class(task,tier)=='long':
        final=bx.load_config(CONFIG/'final.json');c['mcmc']=deepcopy(final['mcmc'])
        if task.kind=='posterior':c['forecast_draws']=final['forecast_draws']
    c['data']['series']=[task.channel]
    # Stable per-response/origin streams; unaffected by process count or job order.
    seed_key=f'{task.channel}/{task.kind}/{task.origin}/{tier}'.encode()
    seed=int.from_bytes(hashlib.sha256(seed_key).digest()[:4],'little')
    c['mcmc']['seed']=seed;c['seed']=(seed+1)%(2**32)
    c['credible_interval']=.95
    c['experiment']=dict(task_id=task.id,tier=tier,batch_kind=task.kind,channel=task.channel,
        design=task.design if task.kind=='forecast' else task.kind,
        reference='1.9.3 independent analyses; calibrated fixed Normal SDs; no shrinkage hyperpriors')
    if (c['analysis']!='independent' or c.get('copula') is not None or
        c['priors'].get('shared_shrinkage') is not None or c['priors'].get('independent_shrinkage') is not None or c['priors'].get('innovation_sd') is None):
        raise ValueError('1.9.3 requires independent analyses, fixed Normal SDs and no shrinkage hyperpriors.')
    c['output']=str(ROOT/tier/task.id/'report')
    return c


def fingerprint(config):
    """Never reuse completed work after changes to scientific settings, source or data."""
    digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    source=hashlib.sha256()
    for folder in ('bucex','research'):
        for path in sorted((PROJECT/folder).rglob('*.py')):
            source.update(str(path.relative_to(PROJECT)).encode());source.update(path.read_bytes())
    data=Path(config['data']['daily_source'])
    if not data.is_absolute():data=PROJECT/data
    return dict(config_sha256=digest,source_sha256=source.hexdigest(),daily_sha256=hashlib.sha256(data.read_bytes()).hexdigest())


def execute(task,*,tier='paper',root=ROOT,retry_failed=False):
    directory=Path(root).resolve()/tier/task.id;directory.mkdir(parents=True,exist_ok=True)
    c=task_config(task,tier);identity=dict(bucex_version=bx.__version__,**fingerprint(c))
    manifest=directory/'task.json'
    if manifest.exists():
        previous=bx.load_config(manifest)
        if any(previous.get(k)!=v for k,v in identity.items()):
            raise ValueError(f'{directory}: configuration, source or data changed; use a new results root')
        if previous.get('status')=='completed':
            print(f'{task.id}: completed ({previous.get("numerical_status")}); skipped',flush=True)
            return directory/'report'/task.channel
        if previous.get('status')!='failed' or not retry_failed:
            raise RuntimeError(f'{directory}: prior attempt is {previous.get("status")}; inspect its log. --retry-failed only retries recorded failures; unfinished running tasks need a fresh root.')
        archive=directory/'attempts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        archive.mkdir(parents=True)
        for old in ('task.json','resolved_config.json','report'):
            if (directory/old).exists():shutil.move(str(directory/old),str(archive/old))
    meta=dict(task=asdict(task),tier=tier,**identity,status='running',started=datetime.now(timezone.utc).isoformat())
    with manifest.open('x') as handle:json.dump(meta,handle,indent=2)
    bx.save_config(c,directory/'resolved_config.json')
    try:
        result=directory/'report'/task.channel
        if task.kind=='forecast':
            validate(c,directory=directory/'report')
            import pandas as pd
            statuses=pd.read_csv(result/'folds.csv').numerical_status
            numerical=('passed_numerical_checks' if len(statuses) and statuses.eq('passed_numerical_checks').all() else 'needs_review')
        else:
            fit_full(c,directory=directory/'report')
            shutil.copy2(directory/'report/data_window.json',result/'data_window.json')
            numerical=bx.load_config(result/'convergence.json')['status']
        meta.update(status='completed',result=str(result),numerical_status=numerical)
        if tier=='paper' and task.kind=='posterior' and task.variant=='reference':
            from research.seasonal.check_final import assess
            check=assess(result,directory/'resolved_config.json')
            bx.save_config(check,result/'final_check.json');meta['final_check']=check['status']
    except BaseException as error:
        meta.update(status='failed',error=str(error),traceback=traceback.format_exc());raise
    finally:
        meta['finished']=datetime.now(timezone.utc).isoformat();bx.save_config(meta,manifest)
    print(f'{task.id}: {meta["status"]}; {numerical}; {result}',flush=True)
    return result


def verify(tier='paper',*,output=None):
    """Compile all distinct specifications and verify all complete folds; no MCMC."""
    import pandas as pd
    from research.monthly.models import independent_model,fit_options
    from research.monthly.validate import validation_splits
    cache={};compiled=set();rows=[];posterior={}
    for t in tasks('all',tier=tier):
        c=task_config(t,tier);window=dict(c['data'],series=settings()['series']);key=json.dumps(window,sort_keys=True)
        if key not in cache:cache[key]=bx.load_uccle_multiseries(**window)
        data=cache[key][[t.channel]];model,prior=independent_model(data,c)
        signature=json.dumps([c['data'],c['model'],c['priors']],sort_keys=True)
        if signature not in compiled:bx.compile_model(model,data);compiled.add(signature)
        fit_options(c,family=model.family)
        if c['credible_interval']!=.95 or not c['save_fits']:raise ValueError('Expected 95% intervals and saved fits.')
        if t.kind=='posterior':
            if len(data)!=538 or c['forecast_horizon']!=120:raise ValueError('Full fits require 538 blocks and 30-year forecasts.')
            posterior[t.variant]=c
        elif t.kind=='pre2019':
            if c['data']['end']!='2019-05' or c['forecast_horizon']!=1:raise ValueError('Record-event fit must be prospective.')
        else:
            splits=validation_splits(data,c['validation'])
            if len(splits)!=1 or len(splits[0][1])!=t.horizon:raise ValueError('Expected one complete declared validation fold.')
        if len(model.channels)!=1 or model.copula is not None or list(prior.channels)!=[t.channel]:
            raise ValueError('A fit must contain precisely its own response and no copula.')
        if prior.shrinkage is not None:
            raise ValueError('All shrinkage priors must have fixed SDs, without hyperpriors.')
        p=c['priors'];rows.append(dict(task_id=t.id,kind=t.kind,variant=t.variant,channel=t.channel,origin=t.origin,
            resource=resource_class(t,tier),n_blocks=len(data),chains=c['mcmc']['chains'],warmup=c['mcmc']['warmup'],draws=c['mcmc']['draws'],
            forecast_horizon=c['forecast_horizon'] if t.kind!='forecast' else t.horizon,
            level_prior_sd=p['innovation_sd']['level'],slope_prior_sd=p['innovation_sd']['trend'],
            seasonal_prior_sd=p['innovation_sd']['season'],initial_slope_prior_sd=p['initial_slope_sd'],
            analysis='independent',pool_initial_slope=False,design=t.design if t.kind=='forecast' else t.kind,
            initial_rate_sd_C_per_decade=40*p['initial_slope_sd'],xi_sd=p['xi_sd'],save_fits=c['save_fits']))
    ref=posterior['reference'];z=.6744897501960817
    for field,median in ref['calibration']['reference_m0'].items():
        actual=ref['priors']['initial_slope_sd'] if field=='initial_slope' else ref['priors']['innovation_sd'][field]
        if not np.isclose(actual,median/z,rtol=1e-14):raise ValueError('Incorrect reference Normal calibration: '+field)
    result=dict(version=bx.__version__,tier=tier,status='configuration_checks_passed',
        counts={b:len(tasks(b,tier=tier)) for b in BATCHES},
        scope='Single-response models, data windows, units, horizons and fixed Normal calibration; no MCMC execution.')
    if output is not None:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        bx.save_config(result,output/'preflight.json');pd.DataFrame(rows).to_csv(output/'resolved_settings.csv',index=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',choices=BATCHES,default='posterior');parser.add_argument('--tier',choices=('screen','paper'),default='screen')
    parser.add_argument('--resource',choices=('all','standard','long'),default='all')
    action=parser.add_mutually_exclusive_group(required=True)
    for flag in ('list','count','verify','resource-info'):action.add_argument('--'+flag,action='store_true')
    action.add_argument('--index',type=int);action.add_argument('--task')
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.tier,output=args.root/args.tier/'plan'),indent=2));return
    if args.resource_info:
        key='paper_long' if args.tier=='paper' and args.resource=='long' else args.tier
        r=settings()['resources'][key];print(r['cpus'],r['memory_gb'],r['hours']);return
    available=tasks(args.batch,tier=args.tier,resource=args.resource)
    if args.count:print(len(available));return
    if args.list:
        for i,t in enumerate(available,1):
            c=task_config(t,args.tier)
            print(f"{i:3d} {t.id:60s} {resource_class(t,args.tier):8s} chains={c['mcmc']['chains']} warmup={c['mcmc']['warmup']} retained={c['mcmc']['draws']}")
        return
    selected=(next((t for t in tasks('all',tier=args.tier) if t.id==args.task),None) if args.task else
              available[args.index-1] if 1<=args.index<=len(available) else None)
    if selected is None:parser.error('Unknown index or task ID')
    execute(selected,tier=args.tier,root=args.root,retry_failed=args.retry_failed)

if __name__=='__main__':main()
