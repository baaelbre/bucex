"""One fit per job. Default: posterior sensitivity; validation is a separate batch."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np
import bucex as bx
from research.monthly.experiment import configured_variant
from research.monthly.run import run as fit_full
from research.monthly.validate import validate

PROJECT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT/'research/seasonal/config'
ROOT = Path('results/serra_191_parallel')
BATCHES = ('posterior', 'reference', 'pre2019', 'validation', 'validation10', 'all')
STUDIES = ('manuscript_sensitivity', 'physical_sensitivity', 'adequacy')

@dataclass(frozen=True)
class Task:
    id: str
    kind: str
    study: str
    variant: str
    origin: str = ''
    horizon: int = 20
    design: str = 'calendar_5y'


def settings():
    return bx.load_config(CONFIG/'experiments.json')


def resource_class(task, tier):
    return ('long' if tier == 'paper' and task.variant == 'reference'
            and task.kind in ('posterior', 'pre2019') else 'standard')


def tasks(batch='posterior', *, tier='paper', resource='all'):
    """Stable one-based ordering within the named batch and resource class."""
    if batch not in BATCHES or tier not in ('screen','paper') or resource not in ('all','standard','long'):
        raise ValueError('Unknown batch, tier or resource class.')
    spec = settings();variants = {}
    for study in spec['studies']:
        for variant in bx.load_config(CONFIG/f'{study}.json')['variants']:
            name=variant['name']
            if name=='reference' and name in variants:continue
            if name in variants:raise ValueError(f'Duplicate variant: {name}')
            variants[name]=study
    posterior=[Task('posterior_'+v,'posterior',s,v) for v,s in variants.items()]
    record=[Task('pre2019_'+v['name'],'pre2019','pre2019_sensitivity',v['name'])
            for v in bx.load_config(CONFIG/'pre2019_sensitivity.json')['variants']]
    forecasts=[Task('forecast_reference_'+o,'forecast',variants['reference'],'reference',o)
               for o in spec['reference_validation_origins']]
    for v in spec['validation_variants']:
        if v=='reference' or v not in variants:raise ValueError(f'Invalid validation variant: {v}')
        forecasts.extend(Task(f'forecast_{v}_{o}','forecast',variants[v],v,o)
                         for o in spec['sensitivity_validation_origins'])
    # Reproduce the original 60/80/90% splits on complete seasonal blocks.
    # Keep these ten-year checks separate from the calendar five-year study.
    import pandas as pd
    data_config=bx.load_config(CONFIG/'main.json')['data']
    dates=bx.load_uccle_multiseries(**data_config).index
    original=[]
    for fraction in spec['original_validation_fractions']:
        n=int(np.floor(len(dates)*fraction))
        origin=(dates[n-1]+pd.DateOffset(months=3)-pd.Timedelta(days=1)).strftime('%Y-%m')
        original.append(Task(f'forecast_reference_split{round(100*fraction)}_h40',
            'forecast',variants['reference'],'reference',origin,
            spec['original_validation_horizon'],'fraction_10y'))
    all_tasks=posterior+record+forecasts+original
    if len({t.id for t in all_tasks})!=len(all_tasks):raise ValueError('Duplicate task IDs.')
    chosen={'posterior':posterior,'reference':posterior[:1],'pre2019':record,
            'validation':forecasts,'validation10':original,'all':all_tasks}[batch]
    return [t for t in chosen if resource=='all' or resource_class(t,tier)==resource]


def task_config(task,tier):
    if tier not in ('screen','paper'):raise ValueError('Unknown tier.')
    source=bx.load_config(CONFIG/f'{task.study}.json')
    entry=next(v for v in source['variants'] if v['name']==task.variant)
    c=configured_variant(source,entry);c.pop('variants',None);c['variant']=deepcopy(entry)
    budget=settings()['tiers'][tier]
    c['mcmc'].update(budget['mcmc']);c['diagnostic_thresholds']=deepcopy(budget['diagnostic_thresholds'])
    for field in ('forecast_draws','predictive_check_draws','save_fits','figures','trace_exports'):
        c[field]=budget[field]
    c['validation'].update(draws=budget['validation_draws'],save_fits=budget['save_fits'])
    if task.kind=='forecast':c['validation'].update(training_ends=[task.origin],horizon=task.horizon)
    elif task.kind=='pre2019':
        c['forecast_horizon']=1
        if tier=='paper':c['forecast_draws']=10000
    else:c['forecast_horizon']=120
    if resource_class(task,tier)=='long':
        final=bx.load_config(CONFIG/'final.json');c['mcmc']=deepcopy(final['mcmc'])
        if task.kind=='posterior':c['forecast_draws']=final['forecast_draws']
    c['credible_interval']=.95
    c['experiment']=dict(task_id=task.id,tier=tier,batch_kind=task.kind,
                         design=task.design if task.kind=='forecast' else task.kind,
                         reference='1.9.1 identity; three shared innovation scales; separate initial rates')
    if c['analysis'] != 'joint' or c['priors']['shared_shrinkage'].get('pool_initial_slope',True):
        raise ValueError('1.9.1 tasks require R=I and separate initial rates.')
    c['output']=str(ROOT/tier/task.id/'report')
    return c


def fingerprint(config):
    """Do not pool changed source, settings or daily observations under one ID."""
    digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    source=hashlib.sha256()
    for folder in ('bucex','research'):
        for path in sorted((PROJECT/folder).rglob('*.py')):
            source.update(str(path.relative_to(PROJECT)).encode());source.update(path.read_bytes())
    data=Path(config['data']['daily_source'])
    if not data.is_absolute():data=PROJECT/data
    return dict(config_sha256=digest,source_sha256=source.hexdigest(),
                daily_sha256=hashlib.sha256(data.read_bytes()).hexdigest())


def execute(task,*,tier='paper',root=ROOT):
    directory=Path(root)/tier/task.id;directory.mkdir(parents=True,exist_ok=True)
    c=task_config(task,tier);identity=dict(bucex_version=bx.__version__,**fingerprint(c))
    manifest=directory/'task.json'
    if manifest.exists():
        previous=bx.load_config(manifest)
        if any(previous.get(k)!=v for k,v in identity.items()):
            raise ValueError(f'{directory}: configuration, source or data changed; use a new results root')
        if previous.get('status')=='completed':
            print(f'{task.id}: completed ({previous.get("numerical_status")}); {previous["result"]}',flush=True)
            return Path(previous['result'])
        raise RuntimeError(f'{directory}: incomplete prior attempt; inspect the scheduler and use a fresh results root to retry')
    meta=dict(task=asdict(task),tier=tier,**identity,status='running',started=datetime.now(timezone.utc).isoformat())
    with manifest.open('x') as handle:json.dump(meta,handle,indent=2)
    bx.save_config(c,directory/'resolved_config.json')
    try:
        if task.kind=='forecast':
            result=validate(c,directory=directory/'report')
            import pandas as pd
            statuses=pd.read_csv(Path(result)/'joint/folds.csv').numerical_status
            numerical=('passed_numerical_checks' if len(statuses) and statuses.eq('passed_numerical_checks').all() else 'needs_review')
        else:
            result=fit_full(c,directory=directory/'report')
            numerical=bx.load_config(Path(result)/'convergence.json')['status']
        meta.update(status='completed',result=str(Path(result).resolve()),numerical_status=numerical)
        if tier=='paper' and task.id=='posterior_reference':
            from research.seasonal.check_final import assess
            check=assess(Path(result),directory/'resolved_config.json')
            bx.save_config(check,Path(result)/'final_check.json');meta['final_check']=check['status']
    except BaseException as error:
        meta.update(status='failed',error=str(error),traceback=traceback.format_exc());raise
    finally:
        meta['finished']=datetime.now(timezone.utc).isoformat();bx.save_config(meta,manifest)
    print(json.dumps(meta,indent=2),flush=True)
    return Path(result)


def verify(tier='paper',*,output=None):
    """Compile every declared model and check prior units and fold geometry; no fit."""
    import pandas as pd
    from research.monthly.models import joint_model,fit_options
    from research.monthly.validate import validation_splits
    cache,rows,posterior={},{},{}
    for t in tasks('all',tier=tier):
        c=task_config(t,tier);key=json.dumps(c['data'],sort_keys=True)
        if key not in cache:cache[key]=bx.load_uccle_multiseries(**c['data'])
        data=cache[key];model,prior=joint_model(data,c);bx.compile_model(model,data);fit_options(c,family=model.family)
        if c['credible_interval']!=.95 or not c['save_fits']:raise ValueError('Expected 95% intervals and saved fits.')
        if t.kind=='posterior':
            if len(data)!=538 or c['forecast_horizon']!=120:raise ValueError('Full fits require 538 blocks and 30-year forecasts.')
            posterior[t.variant]=c
        elif t.kind=='pre2019':
            if c['data']['end']!='2019-05' or c['forecast_horizon']!=1:raise ValueError('Record-event fit must be prospective.')
        else:
            splits=validation_splits(data,c['validation'])
            if len(splits)!=1 or len(splits[0][1])!=t.horizon:
                raise ValueError('Expected one complete declared validation fold.')
        if model.copula.seasonal or not np.array_equal(model.copula.correlation_matrix({},tuple(data)),np.eye(6)):
            raise ValueError('Every 1.9.1 fit must use fixed identity correlation.')
        if set(prior.shrinkage.anchors) - {'level','slope','seasonal'}:
            raise ValueError('Initial rates must not have a learned shared scale.')
        p=c['priors'];rows[t.id]=dict(task_id=t.id,kind=t.kind,variant=t.variant,origin=t.origin,resource=resource_class(t,tier),
            n_blocks=len(data),chains=c['mcmc']['chains'],warmup=c['mcmc']['warmup'],draws=c['mcmc']['draws'],
            forecast_horizon=c['forecast_horizon'] if t.kind!='forecast' else c['validation']['horizon'],
            log_sd=p['shared_shrinkage']['log_sd'],level_anchor=p['innovation_median']['level'],
            slope_anchor=p['innovation_median']['trend'],seasonal_anchor=p['innovation_median']['season'],
            dependence='identity',pool_initial_slope=False,design=t.design if t.kind=='forecast' else t.kind,
            initial_rate_sd_C_per_decade=40*p['initial_slope_sd'],xi_sd=p['xi_sd'],save_fits=c['save_fits'])
    def moments(c):
        p=c['priors'];tau=p['shared_shrinkage']['log_sd']
        return np.square(list(p['innovation_median'].values()))*np.exp(2*tau*tau)
    for name in ('narrow_log2_matched','wide_log4_matched'):
        if not np.allclose(moments(posterior[name]),moments(posterior['reference']),rtol=1e-12,atol=0):
            raise ValueError('Incorrect second-moment matching: '+name)
    original=bx.load_config(CONFIG/'reference_189.json')['priors']
    for key in ('level','trend','season'):
        if not np.isclose(posterior['original_innovation_anchors']['priors']['innovation_median'][key],original['innovation_median'][key],rtol=1e-12,atol=0):
            raise ValueError('The anchor control must retain the original anchors.')
    result=dict(version=bx.__version__,tier=tier,status='configuration_checks_passed',
        counts={b:len(tasks(b,tier=tier)) for b in BATCHES},
        scope='Model declarations, data windows, units, horizons and matched priors; no MCMC or scheduler execution.')
    if output is not None:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        bx.save_config(result,output/'preflight.json');pd.DataFrame(list(rows.values())).to_csv(output/'resolved_settings.csv',index=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',choices=BATCHES,default='posterior');parser.add_argument('--tier',choices=('screen','paper'),default='screen')
    parser.add_argument('--resource',choices=('all','standard','long'),default='all')
    action=parser.add_mutually_exclusive_group(required=True)
    for flag in ('list','count','verify','resource-info'):action.add_argument('--'+flag,action='store_true')
    action.add_argument('--index',type=int);action.add_argument('--task')
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
            print(f"{i:3d} {t.id:48s} {resource_class(t,args.tier):8s} chains={c['mcmc']['chains']} warmup={c['mcmc']['warmup']} retained={c['mcmc']['draws']}")
        return
    selected=(next((t for t in tasks('all',tier=args.tier) if t.id==args.task),None) if args.task else
              available[args.index-1] if 1<=args.index<=len(available) else None)
    if selected is None:parser.error('Unknown index or task ID')
    execute(selected,tier=args.tier,root=args.root)

if __name__=='__main__':main()
