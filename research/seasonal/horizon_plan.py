"""Focused validation plan; standard library only so it runs on login nodes."""
import json
from pathlib import Path

BATCHES = ('horizon_reference', 'horizon_fixed', 'horizon_all')


def specification():
    return json.loads((Path(__file__).parent / 'config/horizon_validation.json').read_text())


def plan(tier, batch, resource, entries):
    spec = specification()
    variants = {'horizon_reference': ['reference'], 'horizon_fixed': ['fixed_reference'],
                'horizon_all': ['reference', 'fixed_reference']}[batch]
    origins = spec['priority_origins'] + [o for o in spec['origins'] if o not in spec['priority_origins']]
    budget = spec['budgets'][tier]
    groups = []
    for name in variants:
        shared = entries[name]['scope'] == 'shared'
        for origin in origins:
            ident = f'validation1984_{name}_{origin}'
            # Independent responses are separate scheduler jobs, not a 12-CPU bundle.
            for channel in (['joint'] if shared else spec['series']):
                key = ident if shared else ident + '_' + channel
                series = spec['series'] if shared else [channel]
                groups.append(dict(id=key, kind='forecast', variant=name,
                    scope=entries[name]['scope'], family='horizon_validation', series=series,
                    origin=origin, fraction=None, frequency='seasonal', design='validation_30y',
                    horizon=spec['horizon'], resource='shared' if shared else 'separate',
                    task_ids=[key], parallel_fits=1, required_workers=budget['chain_workers'],
                    cpus=budget['chain_workers'], memory_gb=(24 if tier == 'paper' else 12) if shared else 8,
                    hours=6, sampling_role='validation_' + tier, **budget))
    return [g for g in groups if resource == 'all' or g['resource'] == resource]


def configure(config, task, tier):
    spec = specification()
    config['mcmc'].update(spec['budgets'][tier])
    config.update(horizon_validation=True, save_fits=True, figures=False, trace_exports=False,
        sweetspot_diagnostics=False, point_summary='mean', credible_interval=.95,
        forecast_horizon=spec['horizon'], forecast_draws=spec['predictive_draws'][tier],
        report_joint_risks=False, cross_summary_contrasts=False,
        diagnostic_thresholds=dict(min_chains=4 if tier == 'paper' else 2,
            max_rhat=1.01 if tier == 'paper' else 1.05, min_ess=400 if tier == 'paper' else 200))
    config['validation'].update(training_ends=[task.origin], horizon=spec['horizon'],
        allow_partial_horizon=True, draws=spec['predictive_draws'][tier], save_fits=True)
    config['horizon_specification'] = spec
    config['experiment'].update(reference='1.9.8.4 validation: pooled HN (0.1, 0.002, 0.1)',
        sampling_role='validation_' + tier,
        note='Expanding training record; frozen calibration. Later-data-informed design is retrospective validation.')
    return config
