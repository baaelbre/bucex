"""Standard-library-only plan for pooled fits and optional later comparisons."""
import json
from pathlib import Path
from research.seasonal.sweetspot_plan import BATCHES as SWEETSPOT_BATCHES, study_variants, validation_cells, specification
from research.monthly.study_plan import BATCHES as MONTHLY_BATCHES, variants as monthly_variants, plan as monthly_plan
from research.seasonal.final_plan import BATCHES as FINAL_BATCHES, variants as final_variants, plan as final_plan
PROJECT=Path(__file__).resolve().parents[2]
CONFIG=PROJECT/'research/seasonal/config'
BATCHES=('reference','comparison','posterior','experiments','sensitivity','influence','pre2019','validation','validation10','block_validation','monthly','core','all','deferred') + SWEETSPOT_BATCHES + MONTHLY_BATCHES + FINAL_BATCHES
RESOURCES=('shared','separate','shared_long','separate_long','monthly')
BASELINES=('reference',)

def read(name):
    return json.loads((CONFIG/(name+'.json')).read_text())

def variant_entries():
    spec = read('experiments')
    legacy=spec['variants'] + spec.get('deferred_variants', [])
    if len({v['name'] for v in legacy}) != len(legacy): raise ValueError('Duplicate legacy variants.')
    entries={v['name']:v for v in legacy}
    entries.update({v['name']:v for v in study_variants()})
    entries.update({v['name']:v for v in monthly_variants()})
    entries.update({v['name']:v for v in final_variants()})
    return list(entries.values())

def plan(tier='screen',batch='experiments',resource='all'):
    if tier not in ('screen','paper') or batch not in BATCHES or resource not in ('all',*RESOURCES):
        raise ValueError('Unknown tier, batch or resource class.')
    if batch in MONTHLY_BATCHES:return monthly_plan(tier,batch,resource)
    spec=read('experiments')
    variants=variant_entries()
    entries={v['name']:v for v in variants}
    if batch in FINAL_BATCHES:return final_plan(tier,batch,resource,entries)
    if len(entries)!=len(variants):raise ValueError('Duplicate variant names.')
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
    calibration=specification()
    sweet_posterior=[group('posterior',c['name']) for c in study_variants()]
    def sweet_forecasts(origins):
        result=[]
        selected_cells=validation_cells()
        for cell in selected_cells:
            for origin in origins:
                g=group('forecast',cell['name'],origin)
                # A distinct ID prevents collision with the legacy 35-year folds.
                g['id']='sweetspot_'+g['id'];g['task_ids']=['sweetspot_'+id for id in g['task_ids']]
                g['design']='sweetspot_30y';g['horizon']=calibration['validation_horizon']
                result.append(g)
        return result
    sweet_recent=sweet_forecasts(calibration['recent_origins'])
    sweet_long=sweet_forecasts(calibration['long_origins'])
    selected=dict(reference=refs,comparison=[g for g in posterior if g['variant'] in ('reference','slope_1e3','fixed_location_seasonality')],
        posterior=posterior,experiments=posterior,sensitivity=[g for g in posterior if g['variant']!='reference'],
        influence=[g for g in posterior if g['family']=='influence'],pre2019=pre,validation=forecasts,validation10=original,block_validation=blocks,
        core=refs+[g for g in pre if g['variant']=='reference']+[g for g in forecasts if g['variant']=='reference'],
        monthly=monthly,all=refs+pre+forecasts+[g for g in posterior if g['variant']!='reference']+original+blocks+monthly,deferred=deferred,
        sweetspot=sweet_posterior+sweet_recent+sweet_long,
        sweetspot_hpc=sweet_posterior+sweet_long,sweetspot_posterior=sweet_posterior,
        sweetspot_validation=sweet_recent,sweetspot_long=sweet_long,
        sweetspot_fixed=[g for g in sweet_posterior+sweet_recent+sweet_long if g['scope']=='fixed'])[batch]
    if len({g['id'] for g in selected})!=len(selected):raise ValueError('Duplicate experiment IDs.')
    if any(g['required_workers']>g['cpus'] for g in selected):raise ValueError('CPU allocation below chain worker count.')
    return [g for g in selected if resource=='all' or g['resource']==resource]
