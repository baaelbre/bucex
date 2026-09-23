"""Screen ten seasonal shrinkage settings with the same held-out forecast cases."""
import argparse
from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import sleep

import numpy as np
import pandas as pd

import bucex as bx
from research.monthly.validate import validate


def settings(config):
    """Nine level/slope combinations and one initial-rate sensitivity control."""
    assert config['data']['frequency'] == 'seasonal' and config['model']['period'] == 4
    g = config['grid']
    levels, slopes = g['level_factors'], g['slope_factors']
    if levels != [0.5, 1.0, 2.0] or slopes != [0.5, 1.0, 2.0]:
        raise ValueError('This predeclared screen uses factors 0.5, 1 and 2 for both components.')
    base = config['priors']
    cases = []
    for level in levels:
        for slope in slopes:
            label = 'reference' if (level, slope) == (1.0, 1.0) else f'level_{level:g}_slope_{slope:g}'
            local = deepcopy(config)
            local['priors']['innovation_median']['level'] = base['innovation_median']['level'] * level
            local['priors']['innovation_median']['trend'] = base['innovation_median']['trend'] * slope
            cases.append((label, local))
    control = deepcopy(config)
    control['priors']['initial_slope_sd'] *= g['extra_initial_slope_factor']
    cases.append(('initial_slope_1.5', control))
    if len(cases) != 10 or len({name for name, _ in cases}) != 10:
        raise ValueError('The reference grid must contain ten unique named settings.')
    return cases


def score_case(directory, name, local):
    """Summarize each completed setting without changing its saved forecast cases."""
    target = directory / name / 'joint'
    scores = pd.read_csv(target / 'scores.csv')
    scores = scores.loc[scores.score.eq('crps'), ['origin', 'time', 'channel', 'horizon', 'value']]
    keys = ['origin', 'time', 'channel', 'horizon']
    if scores.duplicated(keys).any() or not np.isfinite(scores.value).all():
        raise ValueError(f'{name}: repeated or nonfinite CRPS cases.')
    expected = len(local['validation']['training_ends']) * local['validation']['horizon'] * len(local['data']['series'])
    if len(scores) != expected:
        raise ValueError(f'{name}: expected {expected} held-out CRPS cases; found {len(scores)}.')
    pit = pd.read_csv(target / 'held_out_pit.csv')
    if len(pit) != expected or not np.isfinite(pit.pit).all():
        raise ValueError(f'{name}: PIT cases do not match CRPS cases.')
    joint = pd.read_csv(target / 'joint_log_scores.csv')
    joint_cases = len(local['validation']['training_ends']) * local['validation']['horizon']
    if len(joint) != joint_cases or not np.isfinite(joint.score).all():
        raise ValueError(f'{name}: joint log scores do not match held-out seasons.')
    checks = [bx.load_config(path) for path in sorted(target.glob('convergence_*.json'))]
    if len(checks) != len(local['validation']['training_ends']):
        raise ValueError(f'{name}: expected one numerical check per held-out origin.')
    row = dict(setting=name,
        level_median=local['priors']['innovation_median']['level'],
        slope_median=local['priors']['innovation_median']['trend'],
        seasonal_median=local['priors']['innovation_median']['season'],
        initial_slope_sd=local['priors']['initial_slope_sd'],
        mean_crps=float(scores.value.mean()),
        mean_joint_negative_log_score=float(joint.score.mean()),
        mean_pit=float(pit.pit.mean()),
        fraction_pit_above_0_9=float((pit.pit > .9).mean()),
        forecast_cases=len(scores),
        flagged_quantities=sum(len(check['issues']) for check in checks),
        numerical_checks_passed=all(check['status'] == 'passed_numerical_checks' for check in checks))
    return row, scores.set_index(keys).sort_index()['value']


def fit_case(root, name, local):
    """Write only this setting's files; the driver owns all shared summaries."""
    target = root / name
    marker = target / 'complete.json'
    if marker.exists():
        if bx.load_config(target / 'config.json') != local:
            raise ValueError(f'{name}: saved run does not match the requested configuration.')
        score_case(root, name, local)
        print(f'{name}: already completed.', flush=True)
    else:
        print(f'{name}: fitting two chains at each training origin.', flush=True)
        validate(local, directory=target)
        score_case(root, name, local)  # Never mark an incomplete run as complete.
        bx.save_config({'setting': name, 'origins': local['validation']['training_ends']}, marker)


def summarize(root, cases):
    """Only the driver writes shared tables, after a setting has completed."""
    completed = [(n, c) for n, c in cases if (root / n / 'complete.json').exists()]
    if not completed:
        return
    rows, sequences = [], {}
    for n, c in completed:
        row, sequences[n] = score_case(root, n, c)
        rows.append(row)
    reference = sequences.get('reference')
    if reference is not None:
        for row in rows:
            value = sequences[row['setting']]
            if not value.index.equals(reference.index):
                raise ValueError(f'{row["setting"]}: held-out cases differ from reference.')
            row['crps_difference_from_reference'] = float((value - reference).mean())
    ranking = pd.DataFrame(rows).sort_values(['mean_crps', 'setting'])
    ranking.to_csv(root / 'grid_summary.csv', index=False)
    detail = pd.concat([value.rename('crps').reset_index().assign(setting=name)
                        for name, value in sequences.items()], ignore_index=True)
    detail.groupby(['setting', 'channel'])['crps'].agg(mean_crps='mean', n='size').to_csv(
        root / 'grid_by_response.csv')
    best = ranking.iloc[0]
    eligible = ranking.loc[ranking.numerical_checks_passed]
    bx.save_config({'best_pilot_setting': best['setting'],
        'best_mean_crps': float(best['mean_crps']),
        'best_passing_pilot_setting': None if eligible.empty else eligible.iloc[0]['setting'],
        'finished_settings': len(completed), 'planned_settings': len(cases),
        'numerical_checks_passed': bool(best['numerical_checks_passed']),
        'selection_basis': 'Mean marginal CRPS across six responses and identical held-out seasonal cases.',
        'interpretation': 'Scores are provisional when numerical checks fail. Rerun promising settings with four long chains and assess reserved 2024--2026 seasons separately.'},
        root / 'selection.json')
    print(f'Completed {len(completed)}/10; lowest held-out CRPS: {best["setting"]} '
          f'({best["mean_crps"]:.4f} degC).', flush=True)
    if eligible.empty:
        print('No setting passes numerical checks yet; do not select a final prior.', flush=True)


def fit_parallel(root, selected, all_cases, jobs):
    """Run isolated setting processes; each process spawns its two chain workers."""
    pending = [(name, local) for name, local in selected
               if not (root / name / 'complete.json').exists()]
    if not pending:
        return
    config_path = root / 'grid_config.json'
    logs = root / 'logs'
    logs.mkdir(exist_ok=True)
    active, failed = {}, []
    environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                   'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
    try:
        while pending or active:
            while pending and len(active) < jobs:
                name, _ = pending.pop(0)
                log_path = logs / f'{name}.log'
                command = [sys.executable, '-m', 'research.seasonal.grid',
                           '--config', str(config_path.resolve()), '--_worker', name]
                with log_path.open('w') as output:
                    process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                               env=environment, start_new_session=(os.name == 'posix'))
                active[name] = process
                print(f'{name}: started PID {process.pid}; log: {log_path}', flush=True)
            finished = [(name, process) for name, process in active.items()
                        if process.poll() is not None]
            for name, process in finished:
                del active[name]
                if process.returncode != 0 or not (root / name / 'complete.json').exists():
                    failed.append(name)
                    print(f'{name}: failed; inspect {logs / (name + ".log")}', flush=True)
                else:
                    print(f'{name}: finished.', flush=True)
                    summarize(root, all_cases)
            if active and not finished:
                sleep(1)
    except BaseException:
        for process in active.values():
            if process.poll() is None:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
        for process in active.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
        raise
    if failed:
        raise RuntimeError(f'Failed settings: {", ".join(failed)}. Inspect logs and rerun; completed settings will be reused.')


def run(config, *, only=None, dry_run=False, jobs=1):
    cases = settings(config)
    root = Path(config['output'])
    names = {name for name, _ in cases}
    if not isinstance(jobs, int) or isinstance(jobs, bool) or not 1 <= jobs <= len(cases):
        raise ValueError('jobs must be an integer from 1 through 10.')
    if only is not None and only not in names:
        raise ValueError(f'Unknown setting {only!r}. Choose from: {", ".join(sorted(names))}.')
    for name, local in cases:
        print(f'{name:24s} level={local["priors"]["innovation_median"]["level"]:.4g} '
              f'slope={local["priors"]["innovation_median"]["trend"]:.4g} '
              f'initial={local["priors"]["initial_slope_sd"]:.4g}', flush=True)
    if dry_run:
        return root
    root.mkdir(parents=True, exist_ok=True)
    source = Path(config['data']['daily_source'])
    plan = {'base_config': config, 'settings': [name for name, _ in cases],
            'daily_source_sha256': sha256(source.read_bytes()).hexdigest()}
    plan_file = root / 'grid_plan.json'
    if plan_file.exists() and bx.load_config(plan_file) != plan:
        raise ValueError('Existing grid has a different plan; choose a fresh output directory.')
    bx.save_config(plan, plan_file)
    bx.save_config(config, root / 'grid_config.json')
    selected = [(name, local) for name, local in cases if only is None or only == name]
    if jobs > 1:
        fit_parallel(root, selected, cases, jobs)
        summarize(root, cases)
    else:
        for name, local in selected:
            fit_case(root, name, local)
            summarize(root, cases)
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='research/seasonal/config/grid.json')
    parser.add_argument('--dry-run', action='store_true', help='Print settings without fitting.')
    parser.add_argument('--only', help='Run or resume one setting first, for example reference.')
    parser.add_argument('--jobs', type=int, default=1, help='Concurrent settings (1-10); each runs two parallel chains.')
    parser.add_argument('--_worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = bx.load_config(args.config)
    if args._worker:
        name, local = next(((n, c) for n, c in settings(config) if n == args._worker), (None, None))
        if local is None:
            parser.error('Unknown internal setting.')
        fit_case(Path(config['output']), name, local)
    else:
        print(run(config, only=args.only, dry_run=args.dry_run, jobs=args.jobs))
