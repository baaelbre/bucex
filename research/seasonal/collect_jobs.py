"""Audit independent jobs and combine only completed, matching forecast cases."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import shutil

import pandas as pd
import bucex as bx
from research.seasonal.jobs import ROOT, STUDIES, tasks
from research.seasonal.grid import settings as grid_settings
from research.seasonal.check_final import assess as assess_reference


def manifests(root, tier):
    result = {}
    for task in tasks('array') + tasks('long'):
        file = Path(root)/tier/task.id/'task.json'
        result[task.id] = json.loads(file.read_text()) if file.exists() else dict(status='missing')
    return result


def merge_folds(paths, destination):
    """Assemble disjoint single-origin validation runs without discarding cases."""
    destination.mkdir(parents=True, exist_ok=True)
    names = ('scores', 'held_out_pit', 'coverage_by_case', 'predictions',
             'event_counts', 'threshold_cases', 'joint_log_scores',
             'compound_heat_scores')
    for name in names:
        inputs = [p/'joint'/f'{name}.csv' for p in paths]
        if all(p.exists() for p in inputs):
            pd.concat([pd.read_csv(p) for p in inputs], ignore_index=True).to_csv(
                destination/f'{name}.csv', index=False)
    fold_tables = [pd.read_csv(p/'joint'/'folds.csv') for p in paths]
    folds = pd.concat(fold_tables, ignore_index=True)
    if folds.origin.duplicated().any():
        raise ValueError(f'{destination}: duplicate validation origin')
    folds.to_csv(destination/'folds.csv', index=False)
    for path in paths:
        for pattern in ('convergence_*.json', 'shared_shrinkage_*.csv'):
            for file in (path/'joint').glob(pattern):
                shutil.copy2(file, destination/file.name)


def collect(root=ROOT, tier='paper', *, figures=False, require_complete=False):
    root = Path(root)
    records = manifests(root, tier)
    expected = tasks('array') + tasks('long')
    incomplete = [t.id for t in expected if records[t.id]['status'] != 'completed']
    flagged = [t.id for t in expected if records[t.id].get('numerical_status') != 'passed_numerical_checks'
               and records[t.id]['status'] == 'completed']
    reference_check = (assess_reference(Path(records['final_reference']['result']),
                                        Path('research/seasonal/config/final.json'))
                       if tier == 'paper' and records['final_reference']['status'] == 'completed'
                       else None)
    if reference_check is not None and reference_check['status'] != 'passed':
        flagged.append('final_reference_strict_gate')
    out = root/tier/'collected'
    out.mkdir(parents=True, exist_ok=True)
    all_budgets = {t.id: records[t.id].get('config_sha256') for t in expected}
    report = dict(version=bx.__version__, tier=tier, expected=len(expected),
                  completed=len(expected)-len(incomplete), missing_or_failed=incomplete,
                  numerical_flags=flagged,
                  final_reference_check=reference_check,
                  status='passed' if not incomplete and not flagged and tier == 'paper' else 'incomplete_or_screening',
                  config_sha256=all_budgets)
    (out/'status.json').write_text(json.dumps(report, indent=2)+'\n')
    if require_complete and (incomplete or flagged or tier != 'paper'):
        raise RuntimeError(f'{len(incomplete)} incomplete jobs, {len(flagged)} numerical warnings; see {out}/status.json')
    for study in STUDIES:
        variants = [v['name'] for v in bx.load_config(Path('research/seasonal/config')/f'{study}.json')['variants']]
        posterior, predictive = {}, {}
        for variant in variants:
            key = ('posterior_pre2019_reference' if study == 'pre2019_sensitivity' and variant == 'reference'
                   else 'posterior_reference' if variant == 'reference'
                   else f'posterior_{study}_{variant}')
            if records[key]['status'] == 'completed':
                posterior[variant] = {name: Path(records[key]['result'])/variant/'joint'
                                      for name in bx.UCCLE_SERIES}
            if study == 'pre2019_sensitivity':
                continue
            source = bx.load_config(Path('research/seasonal/config')/f'{study}.json')
            origins = source['validation']['training_ends']
            task_study = 'manuscript_sensitivity' if variant == 'reference' else study
            keys = [f'forecast_{task_study}_{variant}_{origin}' for origin in origins]
            if all(records[k]['status'] == 'completed' for k in keys):
                fold_dir = out/study/'validation'/variant
                merge_folds([Path(records[k]['result']) for k in keys], fold_dir)
                predictive[variant] = {name: fold_dir for name in bx.UCCLE_SERIES}
        if posterior or predictive:
            # Partial outputs remain visibly flagged in status.json; matched
            # predictive scoring requires the reference and all same cases.
            if predictive and 'reference' not in predictive:
                predictive = {}
            bx.SensitivityReport(posterior_runs=posterior, predictive_runs=predictive,
                baseline='reference', block_frequency='seasonal').save(
                    out/study/'report', figures=figures and not incomplete and not flagged)
    grid_config = bx.load_config('research/seasonal/config/grid.json')
    grid_predictive = {}
    for variant, _ in grid_settings(grid_config):
        keys = ([f'forecast_manuscript_sensitivity_reference_{origin}' for origin in ('2015-11','2020-11')]
                if variant == 'reference' else
                [f'grid_{variant}_{origin}' for origin in ('2015-11','2020-11')])
        if all(records[k]['status'] == 'completed' for k in keys):
            fold_dir = out/'grid'/'validation'/variant
            merge_folds([Path(records[k]['result']) for k in keys], fold_dir)
            grid_predictive[variant] = {name: fold_dir for name in bx.UCCLE_SERIES}
    if 'reference' in grid_predictive:
        bx.SensitivityReport(predictive_runs=grid_predictive, baseline='reference',
            block_frequency='seasonal').save(out/'grid'/'report',
                figures=figures and not incomplete and not flagged)
    block_paths = [t for t in expected if t.kind == 'blocks']
    for task in block_paths:
        if records[task.id]['status'] != 'completed':
            continue
        source = Path(records[task.id]['result'])/task.model/task.origin
        target = out/'block_comparison'/task.model/task.origin
        target.mkdir(parents=True, exist_ok=True)
        for name in ('complete.json', 'scores.csv', 'calibration.csv', 'convergence.json'):
            shutil.copy2(source/name, target/name)
    if any(records[t.id]['status'] == 'completed' for t in block_paths):
        bx.save_block_comparison(out/'block_comparison', figures=figures and not incomplete and not flagged)
    comment_paths = [Path(records[t.id]['result']) for t in expected
                     if t.kind == 'comment5' and records[t.id]['status'] == 'completed']
    if comment_paths:
        merge_folds(comment_paths, out/'comment5'/'joint')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--tier', choices=('paper','screen'), default='paper')
    parser.add_argument('--figures', action='store_true')
    parser.add_argument('--require-complete', action='store_true')
    a = parser.parse_args()
    print(collect(a.root, a.tier, figures=a.figures, require_complete=a.require_complete))
