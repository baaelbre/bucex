"""BUCEX 1.9.6.1 pooled half-normal reference and parallel paper experiments."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
import socket
from pathlib import Path
import shutil
import traceback
import numpy as np
import bucex as bx
from research.monthly.experiment import configured_variant
from research.monthly.run import run as fit_full
from research.monthly.validate import validate
from research.seasonal.job_plan import PROJECT, CONFIG, BATCHES, RESOURCES, BASELINES, plan, variant_entries

ROOT = Path('results/serra_1961')

@dataclass(frozen=True)
class Task:
    id: str
    kind: str
    study: str
    variant: str
    channel: str
    group_id: str
    scope: str
    series: tuple
    resource: str
    origin: str = ''
    horizon: int = 20
    design: str = 'calendar_5y'
    frequency: str = 'seasonal'


def settings():
    return bx.load_config(CONFIG/'experiments.json')


def resource_class(task, tier):
    if task.frequency=='monthly': return 'monthly'
    resource='shared' if task.scope=='shared' else 'separate'
    return resource+('_long' if tier=='paper' and task.kind in ('posterior','pre2019') and task.variant in BASELINES else '')


@lru_cache(maxsize=1)
def _fold_dates():
    return bx.load_uccle_multiseries(**bx.load_config(CONFIG/'main.json')['data']).index


def tasks(batch='posterior', *, tier='paper', resource='all'):
    import pandas as pd
    result=[]
    for g in plan(tier,batch,resource):
        origin=g['origin']
        if g['fraction'] is not None:
            dates=_fold_dates();n=int(np.floor(len(dates)*g['fraction']))
            origin=(dates[n-1]+pd.DateOffset(months=3)-pd.Timedelta(days=1)).strftime('%Y-%m')
        channels=['joint'] if g['scope']=='shared' else g['series']
        for id,ch in zip(g['task_ids'],channels):
            result.append(Task(id,g['kind'],g['family'],g['variant'],ch,g['id'],g['scope'],
                tuple(g['series'] if ch=='joint' else [ch]),g['resource'],origin,g['horizon'],g['design'],g['frequency']))
    return result


def result_directory(directory,task):
    report=Path(directory)/'report'
    return report if task.channel=='joint' and task.kind!='forecast' else report/task.channel


def task_config(task,tier):
    spec=settings();entry=next(v for v in variant_entries() if v['name']==task.variant)
    c=configured_variant(bx.load_config(CONFIG/('monthly_reference.json' if task.frequency=='monthly' else 'main.json')),entry)
    budget=spec['tiers'][tier]
    c['mcmc'].update(budget['mcmc']);c['diagnostic_thresholds']=deepcopy(budget['diagnostic_thresholds'])
    if task.frequency=='monthly':c['mcmc'].update(budget['block_mcmc'])
    for field in ('forecast_draws','predictive_check_draws','save_fits','figures','trace_exports'):
        c[field]=budget[field]
    c['validation'].update(draws=budget['validation_draws'],save_fits=budget['save_fits'])
    if task.kind=='forecast':
        c['validation'].update(training_ends=[task.origin],horizon=task.horizon)
    c['forecast_horizon']=360 if task.frequency=='monthly' else 120
    if task.frequency=='monthly': c['forecast_draws']=10000 if tier=='screen' else 20000
    if resource_class(task,tier).endswith('_long'):
        final=bx.load_config(CONFIG/'final.json')
        c['mcmc'].update(final['mcmc']);c['forecast_draws']=final['forecast_draws']
    if task.kind=='pre2019':
        c['data']['end']='2019-05';c['forecast_horizon']=1
        c['forecast_draws']=50000
        c['risk_periods']=None
    if task.kind=='block':
        c['mcmc'].update(budget['block_mcmc'])
        c['validation'].update(training_ends=[task.origin],horizon=task.horizon*(3 if task.frequency=='monthly' else 1))
    c['data']['series']=list(task.series)
    seed_key=f'{task.channel}/{task.kind}/{task.origin}/{task.frequency}/{tier}'.encode()
    seed=int.from_bytes(hashlib.sha256(seed_key).digest()[:4],'little')
    c['mcmc']['seed']=seed;c['seed']=(seed+1)%(2**32)
    c['credible_interval']=.95;c['variant']=deepcopy(entry)
    from research.seasonal.sweetspot_plan import cells, specification
    cell=next((v for v in cells() if v['name']==task.variant),None)
    if cell is not None and task.frequency=='seasonal':
        c['sweetspot_diagnostics']=True
        c['sweetspot_cell']={key:cell[key] for key in ('A_level','A_slope','A_season')}
        c['sweetspot_specification']=specification()
    c['annual_risk_draws']=5000 if tier=='screen' else 20000
    c['experiment']=dict(task_id=task.id,group_id=task.group_id,tier=tier,scope=task.scope,
        batch_kind=task.kind,channel=task.channel,series=list(task.series),
        design=task.design if task.kind=='forecast' else task.kind,
        frequency=task.frequency,reference='1.9.6.1: pooled half-normal innovation scales; separately calibrated initial rates')
    if c.get('copula') is not None or c.get('contrasts') is not None:
        raise ValueError('This comparison has independent residuals and no historical contrasts.')
    hierarchy=c['priors'].get('shared_shrinkage') or c['priors'].get('independent_shrinkage')
    if hierarchy and (hierarchy['scale_parameterization']!='normal_sd' or hierarchy.get('pool_initial_slope',False)):
        raise ValueError('Expected direct coefficient SDs and separate initial rates.')
    c['output']=str(ROOT/tier/task.id/'report')
    return c
def fingerprint(config):
    """Never reuse completed work after changes to scientific settings, source or data."""
    # Output location is operational, so evidence remains portable across hosts.
    scientific = dict(config)
    scientific.pop('output', None)
    digest=hashlib.sha256(json.dumps(scientific,sort_keys=True).encode()).hexdigest()
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
            return result_directory(directory,task)
        if previous.get('status')!='failed' or not retry_failed:
            raise RuntimeError(f'{directory}: prior attempt is {previous.get("status")}; inspect its log. --retry-failed only retries recorded failures; unfinished running tasks need a fresh root.')
        archive=directory/'attempts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        archive.mkdir(parents=True)
        for old in ('task.json','resolved_config.json','report'):
            if (directory/old).exists():shutil.move(str(directory/old),str(archive/old))
    meta=dict(task=asdict(task),tier=tier,**identity,status='running',pid=os.getpid(),hostname=socket.gethostname(),started=datetime.now(timezone.utc).isoformat())
    with manifest.open('x') as handle:json.dump(meta,handle,indent=2)
    bx.save_config(c,directory/'resolved_config.json')
    try:
        result=result_directory(directory,task)
        if task.kind=='forecast':
            validate(c,directory=directory/'report')
            import pandas as pd
            statuses=pd.read_csv(result/'folds.csv').numerical_status
            numerical=('passed_numerical_checks' if len(statuses) and statuses.eq('passed_numerical_checks').all() else 'needs_review')
        elif task.kind=='block':
            from research.seasonal.block_job import run as run_block
            numerical=run_block(c,task,result)
        else:
            fit_full(c,directory=directory/'report')
            
            if result != directory/'report':
                shutil.copy2(directory/'report/data_window.json',result/'data_window.json')
            numerical=bx.load_config(result/'convergence.json')['status']
        meta.update(status='completed',result=str(result),numerical_status=numerical)
        if tier=='paper' and task.kind=='posterior' and task.variant in BASELINES and task.frequency=='seasonal':
            from research.seasonal.check_final import assess
            check=assess(result,directory/'resolved_config.json')
            bx.save_config(check,result/'final_check.json');meta['final_check']=check['status']
    except BaseException as error:
        meta.update(status='failed',error=str(error),traceback=traceback.format_exc());raise
    finally:
        meta['finished']=datetime.now(timezone.utc).isoformat();bx.save_config(meta,manifest)
    print(f'{task.id}: {meta["status"]}; {numerical}; {result}',flush=True)
    return result



def verify(tier='screen',*,output=None):
    """Compile every model, verify folds/resources, and prove prior matching."""
    import pandas as pd
    from research.monthly.models import independent_model,joint_model,fit_options
    from research.monthly.validate import validation_splits
    rows=[];compiled=set();priors={};datasets={}
    for t in all_tasks(tier):
        c=task_config(t,tier)
        data_config=dict(c['data']);data_config.pop('series',None)
        key=json.dumps(data_config,sort_keys=True)
        if key not in datasets:datasets[key]=bx.load_uccle_multiseries(**data_config)
        data=datasets[key].loc[:,list(t.series)]
        model,prior=(joint_model if t.scope=='shared' else independent_model)(data,c)
        signature=json.dumps([list(t.series),c['model'],c['priors']],sort_keys=True)
        if signature not in compiled:
            bx.compile_model(model,data);compiled.add(signature)
        fit_options(c,family=model.family)
        if model.copula is not None or c['credible_interval']!=.95 or not c['save_fits']:
            raise ValueError('Expected no copula, 95% intervals and retained fit archives.')
        if t.kind=='posterior':
            count,horizon=(1614,360) if t.frequency=='monthly' else (538,120)
            if len(data)!=count or c['forecast_horizon']!=horizon:
                raise ValueError('Expected the complete record and 30-year forecasts.')
        elif t.kind in ('forecast','block'):
            splits=validation_splits(data,c['validation'])
            expected=min(t.horizon*(3 if t.frequency=='monthly' else 1), len(data)-splits[0][0].stop) if splits else 0
            if len(splits)!=1 or len(splits[0][1])!=expected:
                raise ValueError('Validation requires a complete declared fold.')
        elif t.kind=='pre2019':
            if str(data.attrs.get('last_included_day'))!='2019-05-31' or c['forecast_horizon']!=1:
                raise ValueError('Pre-2019 must end in May 2019 with a one-season forecast.')
        if (prior.shrinkage is None)!=(t.scope=='fixed'):
            raise ValueError('Wrong shrinkage scope.')
        if t.scope!='shared' and len(prior.channels)!=1:
            raise ValueError('Separate scopes must contain exactly one response.')
        if prior.shrinkage is not None and 'initial_slope' in prior.shrinkage.anchors:
            raise ValueError('Initial rates must remain separate.')
        for ch,p in prior.channels.items():
            if not np.isclose(p.beta0.sd,c['priors']['initial_slope_sd']):
                raise ValueError('Initial-rate SD differs from its declaration.')
        p=c['priors'];h=prior.shrinkage
        if t.kind=='posterior' and t.frequency=='seasonal':priors[(t.scope,c['variant']['setting'],t.channel)]=prior
        rows.append(dict(task_id=t.id,group_id=t.group_id,kind=t.kind,variant=t.variant,
            scope=t.scope,channel=t.channel,series=','.join(t.series),origin=t.origin,resource=t.resource,
            chains=c['mcmc']['chains'],warmup=c['mcmc']['warmup'],draws=c['mcmc']['draws'],
            n_blocks=len(data),forecast_horizon=c['validation']['horizon'] if t.kind in ('forecast','block') else c['forecast_horizon'],
            level_sd_anchor=p['innovation_sd']['level'],slope_sd_anchor=p['innovation_sd']['trend'],
            seasonal_sd_anchor=p['innovation_sd']['season'],log_sd=h.log_sd if h and h.hyperprior=='lognormal' else None,
            hyperprior=None if h is None else h.hyperprior,
            initial_rate_sd_C_per_decade=10*c['model']['steps_per_year']*p['initial_slope_sd'],pool_initial_slope=False,
            innovation_sd_convention='fixed_normal_sd' if h is None else 'conditional_normal_sd_anchor',
            marginal_sd_inflation=1. if h is None else h.rms_multiplier))
    matched=[]
    for (scope,setting,ch),prior in priors.items():
        if scope!='independent':continue
        pooled=priors[('shared',setting,'joint')]
        if (prior.shrinkage.anchors!=pooled.shrinkage.anchors or prior.shrinkage.hyperprior!=pooled.shrinkage.hyperprior
            or prior.shrinkage.df!=pooled.shrinkage.df or prior.shrinkage.log_sd!=pooled.shrinkage.log_sd
            or prior.shrinkage.scale_parameterization!=pooled.shrinkage.scale_parameterization
            or json.dumps(asdict(prior.channels[ch]),sort_keys=True,default=lambda a:a.tolist())
               !=json.dumps(asdict(pooled.channels[ch]),sort_keys=True,default=lambda a:a.tolist())):
            raise ValueError('Independent/shared one-response priors are not matched.')
        matched.append(dict(setting=setting,channel=ch,matched=True))
    groups=plan(tier,'experiments')
    result=dict(version=bx.__version__,tier=tier,status='configuration_checks_passed',
        counts={b:len(tasks(b,tier=tier)) for b in BATCHES},
        experiments=len(groups),experiment_fits=sum(g['parallel_fits'] for g in groups),
        matched_marginal_prior_pairs=len(matched),
        scope='All specifications, folds, CPU plans and exact independent/shared marginal-prior matching; no MCMC.')
    if output is not None:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        bx.save_config(result,output/'preflight.json')
        pd.DataFrame(rows).to_csv(output/'resolved_settings.csv',index=False)
        pd.DataFrame(matched).to_csv(output/'matched_marginal_priors.csv',index=False)
        pd.DataFrame(plan(tier,'all')).to_csv(output/'experiment_plan.csv',index=False)
    return result


def all_tasks(tier):
    return list({t.id:t for batch in ('all','deferred','sweetspot')
                 for t in tasks(batch,tier=tier)}.values())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',choices=BATCHES,default='experiments');p.add_argument('--tier',choices=('screen','paper'),default='screen')
    p.add_argument('--resource',choices=('all',*RESOURCES),default='all')
    action=p.add_mutually_exclusive_group(required=True)
    for flag in ('list','count','verify'):action.add_argument('--'+flag,action='store_true')
    action.add_argument('--index',type=int);action.add_argument('--task')
    p.add_argument('--retry-failed',action='store_true');p.add_argument('--root',type=Path,default=ROOT)
    a=p.parse_args()
    if a.verify:
        print(json.dumps(verify(a.tier,output=a.root/a.tier/'plan'),indent=2));return
    available=tasks(a.batch,tier=a.tier,resource=a.resource)
    if a.count:print(len(available));return
    if a.list:
        for i,t in enumerate(available,1):
            c=task_config(t,a.tier)
            print(f"{i:3d} {t.id:65s} {t.scope:12s} {c['mcmc']['chains']} chains, {c['mcmc']['warmup']} warmup + {c['mcmc']['draws']} retained")
        return
    selected=(next((t for t in all_tasks(a.tier) if t.id==a.task),None) if a.task else
              available[a.index-1] if a.index and 1<=a.index<=len(available) else None)
    if selected is None:p.error('Unknown index or task ID')
    execute(selected,tier=a.tier,root=a.root,retry_failed=a.retry_failed)

if __name__=='__main__':main()
