"""Monthly fixed-Normal study. Planning needs only the Python standard library."""
from copy import deepcopy
import json
import math
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT / 'research/monthly/config'
BATCHES = ('monthly_reference', 'monthly_sensitivity', 'monthly_structural',
           'monthly_observation', 'monthly_all')


def specification():
    return json.loads((CONFIG/'screen_study.json').read_text())


def convert(innovation_sd, initial_slope_sd, *, years=30):
    """Match four endpoint contribution SDs, not an exact time-aggregation law."""
    seasonal, monthly = 4*years, 12*years
    slope_gain2 = lambda h: h*(h-1)*(2*h-1)/6
    return dict(level=innovation_sd['level']*math.sqrt(seasonal/monthly),
                trend=innovation_sd['trend']*math.sqrt(slope_gain2(seasonal)/slope_gain2(monthly)),
                season=innovation_sd['season']), initial_slope_sd/3


def reference_config():
    # Resolve through the public config loader only on compute hosts.
    import bucex as bx
    c = bx.load_config(CONFIG/'fixed_reference.json')
    spec = specification()
    s = spec['seasonal_calibration']
    innovation, slope = convert(s['innovation_sd'], s['initial_slope_sd'], years=spec['matching_years'])
    c['priors']['innovation_sd'] = innovation
    c['priors']['initial_slope_sd'] = slope
    c['priors'].pop('shared_shrinkage', None)
    c['priors'].pop('independent_shrinkage', None)
    c['analysis'] = 'independent'
    c['shrinkage_scope'] = 'fixed'
    c['calibration'] = dict(convention='fixed_normal_sd', reference_sd=innovation,
        initial_slope_sd=slope, seasonal_equivalent=s, matching_years=spec['matching_years'],
        note='Same 30-year innovation contribution SDs and initial rate as the seasonal fixed-Normal reference; this does not equate the complete monthly and seasonal processes.')
    return c


def variants():
    result=[]
    for row in specification()['variants']:
        v=deepcopy(row)
        v.update(name='monthly_fixed_'+row['setting'], scope='fixed',
                 family='monthly_'+row['family'])
        result.append(v)
    return result


def plan(tier, batch, resource='all'):
    spec=specification()
    if tier not in spec['tiers'] or batch not in BATCHES:
        raise ValueError('Unknown monthly tier or batch.')
    groups=[]
    for v in variants():
        reference=v['setting']=='reference'
        if batch=='monthly_reference' and not reference:continue
        if batch=='monthly_sensitivity' and reference:continue
        if batch=='monthly_structural' and v['family']!='monthly_structural' and not reference:continue
        if batch=='monthly_observation' and v['family']!='monthly_observation':continue
        for ch in spec['series']:
            if ch not in v.get('series',spec['series']):continue
            budget=spec['tiers'][tier]['mcmc'];res=spec['resources'][tier]
            task_id='posterior_'+v['name']+'_'+ch
            groups.append(dict(id=task_id,kind='posterior',variant=v['name'],scope='fixed',
                family=v['family'],series=[ch],origin='',fraction=None,frequency='monthly',
                design='monthly_fixed_screen',horizon=360,resource='monthly',task_ids=[task_id],
                parallel_fits=1,required_workers=budget['chain_workers'],**budget,**res))
    if len({g['id'] for g in groups})!=len(groups):raise ValueError('Duplicate monthly tasks.')
    if any(g['required_workers']>g['cpus'] for g in groups):raise ValueError('Insufficient CPUs.')
    return [g for g in groups if resource in ('all','monthly')]


def task_config(task,tier):
    from research.monthly.experiment import configured_variant
    import hashlib
    spec=specification();entry=next(v for v in variants() if v['name']==task.variant)
    c=configured_variant(reference_config(),entry)
    budget=spec['tiers'][tier]
    c['mcmc'].update(budget['mcmc'])
    for key in ('forecast_draws','predictive_check_draws','prior_draws','annual_risk_draws','diagnostic_thresholds'):
        c[key]=deepcopy(budget[key])
    c['data']['series']=[task.channel]
    c['variant']=entry
    seed=int.from_bytes(hashlib.sha256(f'1982/{task.id}/{tier}'.encode()).digest()[:4],'little')
    c['mcmc']['seed']=seed;c['seed']=(seed+1)%(2**32)
    c['experiment']=dict(task_id=task.id,group_id=task.group_id,tier=tier,scope='fixed',
        batch_kind='posterior',channel=task.channel,series=[task.channel],frequency='monthly',
        design='monthly_fixed_screen',reference='Seasonal 1.9.8.1 fixed-Normal central calibration, matched over 30 years')
    c['output']=str(PROJECT/'results/serra_1983_monthly'/tier/task.id/'report')
    return c


def calibration_rows():
    """Explicit conversion audit, including the slightly different 10-year slope gain."""
    from research.monthly.experiment import configured_variant
    ref=reference_config();spec=specification();rows=[]
    for v in variants():
        c=configured_variant(ref,v)
        seasonal=deepcopy(spec['seasonal_calibration'])
        for key,mult in v.get('sd_multipliers',{}).items():
            if key=='initial_slope':seasonal['initial_slope_sd']*=mult
            else:seasonal['innovation_sd'][key]*=mult
        for years in (10,30):
            for frequency,p in (('seasonal',4),('monthly',12)):
                sd=seasonal['innovation_sd'] if p==4 else c['priors']['innovation_sd']
                initial=seasonal['initial_slope_sd'] if p==4 else c['priors']['initial_slope_sd']
                h=p*years
                for component,gain,key in [('level',math.sqrt(h),'level'),
                    ('slope',math.sqrt(h*(h-1)*(2*h-1)/6),'trend'),
                    ('seasonal',math.sqrt(2*h/p),'season'),('initial_slope',h,None)]:
                    active=component!='seasonal' or v.get('seasonal','dynamic')=='dynamic'
                    coefficient=initial if key is None else sd[key]
                    rows.append(dict(setting=v['setting'],frequency=frequency,years=years,
                        component=component,coefficient_sd=coefficient,displacement_sd_C=coefficient*gain if active else 0.,
                        initial_rate_sd_C_per_decade=10*p*initial if key is None else None))
    return rows
