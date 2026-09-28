"""Standard-library-only HPC plan: safe under the login node's system Python."""
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[2]
CONFIG=PROJECT/'research/seasonal/config'
BATCHES=('posterior','reference','pre2019','experiments','validation','validation10','all')


def read(name):
    return json.loads((CONFIG/(name+'.json')).read_text())


def plan(tier='screen',batch='experiments',resource='all'):
    if tier not in ('screen','paper') or batch not in BATCHES or resource not in ('all','standard','long'):
        raise ValueError('Unknown tier, batch or resource class.')
    spec=read('experiments');variants=[]
    for study in spec['studies']:
        for v in read(study)['variants']:
            name=v['name']
            if name=='reference' and name in variants:continue
            if name in variants:raise ValueError('Duplicate variant '+name)
            variants.append(name)
    def group(id,kind,variant,series,origin=''):
        r='long' if tier=='paper' and variant=='reference' and kind in ('posterior','pre2019') else 'standard'
        budget=spec['tiers'][tier]['mcmc'];res=spec['resources']['paper_long' if r=='long' else tier]
        return dict(id=id,kind=kind,variant=variant,series=series,origin=origin,resource=r,
            chains=budget['chains'],chain_workers=budget['chain_workers'],
            parallel_responses=len(series),required_workers=len(series)*budget['chain_workers'],**res)
    series=spec['series']
    posterior=[group('posterior_'+v,'posterior',v,series) for v in variants]
    record=[group('pre2019_'+v['name'],'pre2019',v['name'],['TXx']) for v in read('pre2019_sensitivity')['variants']]
    forecasts=[group('forecast_reference_'+o,'forecast','reference',series,o) for o in spec['reference_validation_origins']]
    for v in spec['validation_variants']:
        if v=='reference' or v not in variants:raise ValueError('Unknown validation variant '+v)
        forecasts.extend(group('forecast_'+v+'_'+o,'forecast',v,series,o) for o in spec['sensitivity_validation_origins'])
    original=[group('forecast_reference_split'+str(round(100*f))+'_h'+str(spec['original_validation_horizon']),
                    'forecast','reference',series) for f in spec['original_validation_fractions']]
    selected={'posterior':posterior,'reference':[g for g in posterior if g['variant']=='reference'],
        'pre2019':record,'experiments':posterior+record,'validation':forecasts,'validation10':original,
        'all':posterior+record+forecasts+original}[batch]
    if any(g['required_workers']>g['cpus'] for g in selected):raise ValueError('Requested CPUs do not cover response x chain workers.')
    if len({g['id'] for g in selected})!=len(selected):raise ValueError('Duplicate experiment IDs.')
    return [g for g in selected if resource=='all' or g['resource']==resource]
