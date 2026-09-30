"""The finite 1.9.8.3 manuscript suite; planning uses the standard library only.

One publication-budget reference, screening-budget robustness/validation fits.
The wider discovery grids remain available under their original batch names.
"""
from copy import deepcopy
import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parent / 'config'
BATCHES = ('final_paper', 'final_reference', 'final_checks', 'final_posterior',
           'final_validation', 'final_pre2019', 'final_blocks')
REFERENCE_ID = 'final_posterior_reference'


def specification():
    return json.loads((CONFIG / 'final_suite.json').read_text())


def variants():
    return [dict(name='double_seasonal', scope='shared', family='structural',
                 sd_multipliers={'season': 2}),
            dict(name='initial_state_narrow', scope='shared', family='adequacy',
                 baseline_sd=5., seasonal_initial_sd=5.),
            dict(name='initial_state_wide', scope='shared', family='adequacy',
                 baseline_sd=20., seasonal_initial_sd=20.)]


def plan(tier, batch, resource, entries):
    spec = specification()
    roles = spec['budgets']
    series = spec['series']

    def group(kind, name, origin='', frequency='seasonal', one_channel=None):
        entry = entries[name]
        shared = entry['scope'] == 'shared'
        channels = [one_channel] if one_channel else series
        role = ('reference' if tier == 'paper' and kind == 'posterior'
                and name == 'reference' and frequency == 'seasonal' else 'screen')
        budget = deepcopy(roles[role])
        if frequency == 'monthly':
            budget.update(spec['monthly_mcmc'])
        res = 'shared_long' if role == 'reference' else ('shared' if shared else 'separate')
        if frequency == 'monthly':
            res = 'monthly'
        suffix = ('_' + frequency if kind == 'block' else '') + ('_' + origin if origin else '')
        ident = 'final_' + kind + '_' + name + suffix
        if one_channel:
            ident += '_' + one_channel
        ids = [ident] if shared or one_channel else [ident + '_' + ch for ch in channels]
        workers = len(ids) * budget['chain_workers']
        return dict(id=ident, kind=kind, variant=name, scope=entry['scope'],
            family=entry['family'], series=channels, origin=origin, fraction=None,
            frequency=frequency, design='matched_blocks' if kind == 'block' else 'calendar_35y',
            horizon=20 if kind == 'block' else 1 if kind == 'pre2019' else 140,
            resource=res, task_ids=ids, parallel_fits=len(ids), required_workers=workers,
            cpus=workers, memory_gb=(24 if role == 'reference' else 16 if frequency == 'monthly'
                                     else 12 if shared else 8*len(ids)),
            hours=6, sampling_role=role, **budget)

    posterior = [group('posterior', name) for name in spec['posterior_variants']]
    forecasts = [group('forecast', 'reference', origin) for origin in spec['reference_origins']]
    forecasts += [group('forecast', name, origin) for name in spec['validation_variants']
                  for origin in spec['sensitivity_origins']]
    pre = [group('pre2019', name, '2019-05') for name in spec['pre2019_variants']]
    blocks = [group('block', 'reference', origin, frequency=freq)
              for origin in spec['block_origins'] for freq in ('seasonal', 'monthly')]
    monthly = [group('posterior', 'monthly_fixed_reference', frequency='monthly', one_channel=ch)
               for ch in series]
    # The reference is deliberately the first submission/queue entry.
    all_groups = posterior[:1] + pre + posterior[1:] + forecasts + blocks + monthly
    selected = {
        'final_paper': all_groups,
        'final_reference': posterior[:1],
        'final_checks': all_groups[1:],
        'final_posterior': posterior,
        'final_validation': forecasts,
        'final_pre2019': pre,
        'final_blocks': blocks + monthly,
    }[batch]
    if len({g['id'] for g in selected}) != len(selected):
        raise ValueError('Duplicate final experiment IDs.')
    if any(g['chains'] > 4 or g['chain_workers'] > g['cpus'] for g in selected):
        raise ValueError('Invalid final chain/CPU budget.')
    return [g for g in selected if resource == 'all' or g['resource'] == resource]
