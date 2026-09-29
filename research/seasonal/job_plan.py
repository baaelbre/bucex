"""Standard-library-only plan for pooled fits and optional later comparisons."""
import json
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
CONFIG=PROJECT/'research/seasonal/config'
BATCHES=('reference','comparison','posterior','experiments','sensitivity','influence','pre2019','validation','validation10','block_validation','monthly','core','all','deferred')
RESOURCES=('shared','separate','shared_long','separate_long','monthly')
BASELINES=('reference',)

def read(name):
    return json.loads((CONFIG/(name+'.json')).read_text())

def plan(tier='screen',batch='experiments',resource='all'):
    if tier not in ('screen','paper') or batch not in BATCHES or resource not in ('all',*RESOURCES):
        raise ValueError('Unknown tier, batch or resource class.')
    spec=read('experiments')
    entries={v['name']:v for v in spec['variants']+spec.get('deferred_variants',[])}
    if len(entries)!=len(spec['variants'])+len(spec.get('deferred_variants',[])):raise ValueError('Duplicate variant names.')
    def group(kind,name,origin='',fraction=None,frequency='seasonal'):
        v=entries[name];series=[ch for ch in spec['series'] if ch not in v.get('exclude_series',[])]
        coupled=v['scope']=='shared'
        long=tier=='paper' and name in BASELINES and kind in ('posterior','pre2019')
        resource=('shared' if coupled else 'separate')+('_long' if long else '')
        if frequency=='monthly':resource='monthly'
        budget=dict(spec['tiers'][tier]['mcmc'])
        if long:budget.update(read('final')['mcmc'])
        if kind=='block' or frequency=='monthly':budget.update(spec['tiers'][tier]['block_mcmc'])
        suffix=('_split'+str(round(100*fraction))+'_h40' if fraction is not None else '_'+origin if origin else '')
        if kind=='block':suffix='_'+frequency+suffix
        elif frequency=='monthly':suffix='_monthly'+suffix
        id=kind+'_'+name+suffix;ids=[id] if coupled else [id+'_'+ch for ch in series]
        res=spec['resources'][tier+'_'+resource]
        return dict(id=id,kind=kind,variant=name,scope=v['scope'],family=v['family'],series=series,origin=origin,fraction=fraction,frequency=frequency,
            design='fraction_10y' if fraction else 'matched_blocks' if kind=='block' else 'calendar_35y',
            horizon=40 if fraction else 1 if kind=='pre2019' else 20 if kind=='block' else spec.get('validation_horizon',20),resource=resource,task_ids=ids,
            chains=budget['chains'],chain_workers=budget['chain_workers'],warmup=budget['warmup'],draws=budget['draws'],
            parallel_fits=len(ids),required_workers=len(ids)*budget['chain_workers'],**res)
    posterior=[group('posterior',v['name']) for v in spec['variants']]
    pre=[group('pre2019',v,'2019-05') for v in spec['pre2019_variants']]
    forecasts=[group('forecast','reference',o) for o in spec['reference_validation_origins']]
    forecasts += [group('forecast',v,o) for v in spec['validation_variants'] for o in (spec['reference_validation_origins'] if v in spec.get('long_validation_variants',[]) else spec['sensitivity_validation_origins'])]
    original=[group('forecast',v,fraction=f) for v in spec['original_validation_variants'] for f in spec['original_validation_fractions']]
    blocks=[group('block','reference',o,frequency=f) for o in spec['block_validation_origins'] for f in ('monthly','seasonal')]
    monthly=[group('posterior','reference',frequency='monthly')]
    deferred=[group('posterior',v['name']) for v in spec.get('deferred_variants',[])]
    refs=[g for g in posterior if g['variant']=='reference']
    selected=dict(reference=refs,comparison=[g for g in posterior if g['variant'] in ('reference','slope_1e3','fixed_location_seasonality')],
        posterior=posterior,experiments=posterior,sensitivity=[g for g in posterior if g['variant']!='reference'],
        influence=[g for g in posterior if g['family']=='influence'],pre2019=pre,validation=forecasts,validation10=original,block_validation=blocks,
        core=refs+[g for g in pre if g['variant']=='reference']+[g for g in forecasts if g['variant']=='reference'],
        monthly=monthly,all=refs+pre+forecasts+[g for g in posterior if g['variant']!='reference']+original+blocks+monthly,deferred=deferred)[batch]
    if len({g['id'] for g in selected})!=len(selected):raise ValueError('Duplicate experiment IDs.')
    if any(g['required_workers']>g['cpus'] for g in selected):raise ValueError('CPU allocation below chain worker count.')
    return [g for g in selected if resource=='all' or g['resource']==resource]
