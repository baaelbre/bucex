"""Standard-library-only plan for matched separate and shared fits."""
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT/'research/seasonal/config'
BATCHES = ('reference','comparison','posterior','experiments','influence','validation','validation10','all')
RESOURCES = ('shared','separate','shared_long','separate_long')
BASELINES = ('reference','independent_reference','fixed_reference')


def read(name):
    return json.loads((CONFIG/(name+'.json')).read_text())


def plan(tier='screen', batch='experiments', resource='all'):
    if tier not in ('screen','paper') or batch not in BATCHES or resource not in ('all',*RESOURCES):
        raise ValueError('Unknown tier, batch or resource class.')
    spec = read('experiments')
    entries = {v['name']:v for v in spec['variants']}
    if len(entries) != len(spec['variants']):
        raise ValueError('Duplicate variant names.')

    def group(kind, name, origin='', fraction=None):
        v = entries[name]
        series = [ch for ch in spec['series'] if ch not in v.get('exclude_series',[])]
        coupled = v['scope']=='shared'
        long = tier=='paper' and name in BASELINES and kind=='posterior'
        resource = ('shared' if coupled else 'separate') + ('_long' if long else '')
        budget = dict(spec['tiers'][tier]['mcmc'])
        if long:
            budget.update(read('final')['mcmc'])
        suffix = ('_split'+str(round(100*fraction))+'_h40' if fraction is not None else
                  '_'+origin if origin else '')
        id = kind+'_'+name+suffix
        task_ids = [id] if coupled else [id+'_'+ch for ch in series]
        resource_spec = spec['resources'][tier+'_'+resource]
        return dict(id=id,kind=kind,variant=name,scope=v['scope'],family=v['family'],
            series=series,origin=origin,fraction=fraction,design='fraction_10y' if fraction else 'calendar_5y',
            horizon=40 if fraction else 20,resource=resource,task_ids=task_ids,
            chains=budget['chains'],chain_workers=budget['chain_workers'],
            warmup=budget['warmup'],draws=budget['draws'],parallel_fits=len(task_ids),
            required_workers=len(task_ids)*budget['chain_workers'],**resource_spec)

    posterior = [group('posterior',v) for v in entries]
    forecasts = [group('forecast','reference',o) for o in spec['reference_validation_origins']]
    for v in spec['validation_variants']:
        if v=='reference' or v not in entries:
            raise ValueError('Invalid validation variant '+v)
        forecasts.extend(group('forecast',v,o) for o in spec['sensitivity_validation_origins'])
    original = [group('forecast',v,fraction=f) for v in BASELINES
                for f in spec['original_validation_fractions']]
    selected = dict(reference=[g for g in posterior if g['variant']=='reference'],
        comparison=[g for g in posterior if g['variant'] in BASELINES],
        posterior=posterior,experiments=posterior,
        influence=[g for g in posterior if g['family']=='influence'],
        validation=forecasts,validation10=original,all=posterior+forecasts+original)[batch]
    if len({g['id'] for g in selected}) != len(selected):
        raise ValueError('Duplicate experiment IDs.')
    if any(g['required_workers']>g['cpus'] for g in selected):
        raise ValueError('CPU requests do not cover the actual fit/chain workers.')
    return [g for g in selected if resource=='all' or g['resource']==resource]
