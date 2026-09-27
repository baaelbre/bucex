"""Independent fit jobs must stay disjoint and preserve the paper calibration."""
import json

import pytest
import pandas as pd

from research.seasonal import jobs
from research.seasonal.collect_jobs import collect, merge_folds


def test_array_is_one_fit_per_id_and_preserves_reference():
    queue = jobs.tasks('array')
    assert len(queue) == 106
    assert len({task.id for task in queue}) == len(queue)
    reference = jobs.task_config(queue[0], 'paper')
    assert reference['priors']['seasonal_initial_sd'] == 20
    assert reference['priors']['initial_slope_median'] == .01
    assert reference['credible_interval'] == .95
    assert reference['mcmc']['chains'] == 4
    assert jobs.task_config(queue[0], 'screen')['mcmc']['chains'] == 2
    for task in queue:
        config = jobs.task_config(task, 'paper')
        assert config['credible_interval'] == .95
        if task.kind in ('forecast', 'grid', 'comment5', 'blocks'):
            field = 'comparison' if task.kind == 'blocks' else 'validation'
            assert len(config[field]['training_ends']) == 1
    full = jobs.task_config(jobs.tasks('long')[0], 'paper')
    assert full['forecast_horizon'] == 120
    assert full['forecast_draws'] == 12000


def test_manifest_reuses_only_identical_completed_task(tmp_path, monkeypatch):
    task = jobs.tasks('array')[0]
    invoked = []

    def fake_fit(config, *, variants, directory):
        invoked.append(variants)
        target = directory/'reference'/'joint'
        target.mkdir(parents=True)
        (target/'convergence.json').write_text(json.dumps({'status': 'passed_numerical_checks'}))
        return directory

    monkeypatch.setattr(jobs, 'fit_sensitivity', fake_fit)
    result = jobs.execute(task, tier='screen', root=tmp_path)
    assert result.exists()
    assert jobs.execute(task, tier='screen', root=tmp_path) == result
    assert invoked == [['reference']]
    manifest = json.loads((tmp_path/'screen'/task.id/'task.json').read_text())
    assert manifest['numerical_status'] == 'passed_numerical_checks'
    with pytest.raises(RuntimeError, match='incomplete'):
        collect(tmp_path, 'screen', require_complete=True)
    status = json.loads((tmp_path/'screen'/'collected'/'status.json').read_text())
    assert status['completed'] == 1
    assert len(status['missing_or_failed']) == 110


def test_forecast_origin_assembly_preserves_cases(tmp_path):
    roots = []
    for index, origin in enumerate((100, 120)):
        root = tmp_path/f'origin_{origin}'
        joint = root/'joint'
        joint.mkdir(parents=True)
        pd.DataFrame({'origin':[origin], 'numerical_status':['passed_numerical_checks']}).to_csv(
            joint/'folds.csv', index=False)
        pd.DataFrame({'origin':[origin], 'channel':['TXx'], 'value':[index+.1]}).to_csv(
            joint/'scores.csv', index=False)
        (joint/f'convergence_{origin}.json').write_text('{"status":"passed_numerical_checks"}')
        roots.append(root)
    target = tmp_path/'merged'
    merge_folds(roots, target)
    assert pd.read_csv(target/'folds.csv').origin.tolist() == [100, 120]
    assert pd.read_csv(target/'scores.csv').value.tolist() == [.1, 1.1]
    assert (target/'convergence_120.json').exists()
