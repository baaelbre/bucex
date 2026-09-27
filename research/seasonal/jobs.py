"""One independent seasonal manuscript fit per PBS array element.

Use --list to inspect the stable, one-based index before submitting. A screen
is for triage only; paper jobs retain the declared four-chain configurations.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

import bucex as bx
from research.monthly.experiment import configured_variant
from research.monthly.run import run as fit_full
from research.monthly.sensitivity import run as fit_sensitivity
from research.monthly.validate import validate
from research.seasonal.compare import run as fit_comparison
from research.seasonal.grid import settings as grid_settings


STUDIES = ('manuscript_sensitivity', 'physical_sensitivity', 'adequacy',
           'dependence_sensitivity', 'pre2019_sensitivity')
CONFIG = Path('research/seasonal/config')
ROOT = Path('results/serra_189_parallel')
FINAL = {
    'reference': 'final', 'pre2019': 'pre2019',
    'constant_copula': 'constant_copula_full',
    'independence': 'independence_full',
    'monthly_supplement': 'research/monthly/config/final.json',
}


@dataclass(frozen=True)
class Task:
    id: str
    kind: str
    study: str = ''
    variant: str = ''
    origin: str = ''
    model: str = ''


def tasks(group='array'):
    """Every array task is one posterior fit or one forecast-origin refit."""
    items = [Task('posterior_reference', 'posterior', 'manuscript_sensitivity', 'reference')]
    for study in STUDIES:
        config = bx.load_config(CONFIG/f'{study}.json')
        for entry in config['variants']:
            name = entry['name']
            if name == 'reference':
                if study == 'pre2019_sensitivity':
                    items.append(Task('posterior_pre2019_reference', 'posterior', study, name))
                continue
            items.append(Task(f'posterior_{study}_{name}', 'posterior', study, name))
    for study in STUDIES[:-1]:
        config = bx.load_config(CONFIG/f'{study}.json')
        for entry in config['variants']:
            if entry['name'] == 'reference' and study != STUDIES[0]:
                continue
            for origin in config['validation']['training_ends']:
                items.append(Task(f'forecast_{study}_{entry["name"]}_{origin}',
                                  'forecast', study, entry['name'], origin))
    for origin in bx.load_config(CONFIG/'comment5.json')['validation']['training_ends']:
        items.append(Task(f'comment5_{origin}', 'comment5', origin=origin))
    grid = bx.load_config(CONFIG/'grid.json')
    for name, _ in grid_settings(grid):
        if name == 'reference':
            continue  # Exactly the same reference folds already run above.
        for origin in ('2015-11', '2020-11'):
            items.append(Task(f'grid_{name}_{origin}', 'grid', variant=name, origin=origin))
    for origin in bx.load_config(CONFIG/'compare_full.json')['comparison']['training_ends']:
        for model in ('monthly', 'seasonal'):
            items.append(Task(f'blocks_{model}_{origin}', 'blocks', origin=origin, model=model))
    if group == 'array':
        return items
    if group == 'long':
        return [Task(f'final_{name}', 'final', study=path)
                for name, path in FINAL.items()]
    raise ValueError('group must be array or long')


def task_config(task, tier):
    if task.kind == 'final':
        path = Path(task.study) if task.study.endswith('.json') else CONFIG/f'{task.study}.json'
        return bx.load_config(path)
    if task.kind in ('posterior', 'forecast'):
        config = bx.load_config(CONFIG/f'{task.study}.json')
        if task.kind == 'forecast':
            entry = next(v for v in config['variants'] if v['name'] == task.variant)
            config = configured_variant(config, entry)
            config['validation']['training_ends'] = [task.origin]
            config['validation']['save_fits'] = False
        else:
            config['variants'] = [next(v for v in config['variants']
                                       if v['name'] == task.variant)]
            # The historical posterior paths and targets matter here; save a
            # short forecast in the report to avoid 30-year replication in
            # every prior sensitivity fit.
            config['forecast_horizon'] = 4
            config['forecast_draws'] = 1000
    elif task.kind == 'comment5':
        config = bx.load_config(CONFIG/'comment5.json')
        config['validation']['training_ends'] = [task.origin]
    elif task.kind == 'grid':
        source = bx.load_config(CONFIG/'grid.json')
        config = next(local for name, local in grid_settings(source) if name == task.variant)
        config['validation'].update(training_ends=[task.origin], horizon=20, draws=4000,
                                    save_fits=False)
        config['mcmc'].update(chains=4, chain_workers=4, warmup=2000, draws=4000)
        config['diagnostic_thresholds'] = bx.load_config(CONFIG/'main.json')['diagnostic_thresholds']
        config.pop('mixing_gate', None)
    elif task.kind == 'blocks':
        config = bx.load_config(CONFIG/'compare_full.json')
        config['comparison']['training_ends'] = [task.origin]
    else:
        raise ValueError(task.kind)
    config['figures'] = False
    config['save_fits'] = False
    config['trace_exports'] = False
    if tier == 'screen':
        config['mcmc'].update(chains=2, chain_workers=2, warmup=700, draws=1300)
        config['diagnostic_thresholds'].update(min_chains=2, max_rhat=1.05, min_ess=100)
        if task.kind == 'blocks':
            config['comparison']['draws'] = 1500
        elif task.kind in ('forecast', 'comment5', 'grid'):
            config['validation']['draws'] = 1500
        config['predictive_check_draws'] = 100
    return config


def execute(task, *, tier='paper', root=ROOT):
    if tier not in ('paper', 'screen'):
        raise ValueError('Unknown tier.')
    directory = Path(root)/tier/task.id
    directory.mkdir(parents=True, exist_ok=True)
    config = task_config(task, tier)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
    manifest = directory/'task.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous.get('config_sha256') != digest or previous.get('bucex_version') != bx.__version__:
            raise ValueError(f'{directory}: saved configuration or package version changed')
        if previous.get('status') == 'completed':
            print(f'{task.id}: already completed at {previous["result"]}', flush=True)
            return Path(previous['result'])
        raise RuntimeError(f'{directory}: incomplete prior attempt; inspect it, then remove task.json to retry')
    meta = dict(task=task.__dict__, tier=tier, bucex_version=bx.__version__,
                config_sha256=digest, status='running', started=datetime.now(timezone.utc).isoformat())
    manifest.write_text(json.dumps(meta, indent=2)+'\n')
    bx.save_config(config, directory/'resolved_config.json')
    try:
        if task.kind == 'posterior':
            result = fit_sensitivity(config, variants=[task.variant], directory=directory/'sensitivity')
        elif task.kind in ('forecast', 'comment5', 'grid'):
            result = validate(config, directory=directory/'validation')
        elif task.kind == 'blocks':
            result = fit_comparison(config, directory=directory/'comparison_fit',
                                    models=(task.model,), aggregate=False)
        else:
            result = fit_full(config)
        meta.update(status='completed', result=str(Path(result).resolve()),
                    finished=datetime.now(timezone.utc).isoformat())
        if task.kind == 'posterior':
            check = bx.load_config(Path(result)/task.variant/'joint'/'convergence.json')
            meta['numerical_status'] = check['status']
        elif task.kind in ('forecast', 'comment5', 'grid'):
            import pandas as pd
            meta['numerical_status'] = pd.read_csv(Path(result)/'joint'/'folds.csv').numerical_status.iloc[0]
        elif task.kind == 'blocks':
            check = bx.load_config(Path(result)/task.model/task.origin/'convergence.json')
            meta['numerical_status'] = check['status']
        elif task.kind == 'final':
            check = bx.load_config(Path(result)/'convergence.json')
            meta['numerical_status'] = check['status']
    except BaseException as error:
        meta.update(status='failed', error=str(error), traceback=traceback.format_exc(),
                    finished=datetime.now(timezone.utc).isoformat())
        manifest.write_text(json.dumps(meta, indent=2)+'\n')
        raise
    manifest.write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta, indent=2), flush=True)
    return Path(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=('array', 'long'), default='array')
    parser.add_argument('--tier', choices=('paper', 'screen'), default='paper')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--index', type=int, help='One-based PBS array index')
    parser.add_argument('--task', help='Explicit task ID')
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    available = tasks(args.group)
    if args.count:
        print(len(available)); return
    if args.list:
        for i, item in enumerate(available, 1):
            print(f'{i:3d} {item.id}')
        return
    if (args.index is None) == (args.task is None):
        parser.error('Supply exactly one of --index or --task')
    selected = (available[args.index-1] if args.index is not None and
                1 <= args.index <= len(available) else next((x for x in available if x.id == args.task), None))
    if selected is None:
        parser.error('Unknown index or task ID')
    execute(selected, tier=args.tier, root=args.root)


if __name__ == '__main__':
    main()
