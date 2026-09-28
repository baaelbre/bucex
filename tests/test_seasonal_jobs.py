"""Scientific configuration, task isolation and resource-routing regressions."""
from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest
import bucex as bx
from research.seasonal import jobs
from research.seasonal.collect_jobs import collect,merge_folds


def test_batches_are_disjoint_and_paper_has_one_reference_per_response():
    groups=[jobs.tasks(b) for b in ('posterior','pre2019','validation','validation10')]
    assert [len(g) for g in groups]==[138,4,348,18]
    assert len({t.id for g in groups for t in g})==508
    assert all(t.kind=='posterior' for t in jobs.tasks())
    for tier,nchains in [('screen',2),('paper',2)]:
        for task in jobs.tasks('all',tier=tier):
            c=jobs.task_config(task,tier)
            assert c['mcmc']['chains']==c['mcmc']['chain_workers']==nchains
            assert c['credible_interval']==.95 and c['save_fits'] and c['validation']['save_fits']
            assert 'variants' not in c  # no second application of an anchor multiplier
            if task.kind=='posterior':assert c['forecast_horizon']==120
            elif task.kind=='pre2019':assert c['forecast_horizon']==1 and c['data']['end']=='2019-05'
            else:assert c['validation']['training_ends']==[task.origin] and c['validation']['horizon']==task.horizon
    assert [t.id for t in jobs.tasks('posterior',tier='paper',resource='long')]==['posterior_reference_'+ch for ch in jobs.settings()['series']]
    assert not jobs.tasks('posterior',tier='screen',resource='long')
    full=jobs.task_config(jobs.tasks('reference')[0],'paper')
    assert (full['mcmc']['warmup'],full['mcmc']['draws'],full['forecast_draws'])==(3000,8000,12000)


def test_fixed_calibration_and_global_sd_controls():
    configs={t.variant:jobs.task_config(t,'screen') for t in jobs.tasks()}
    z=.6744897501960817
    ref=configs['reference']['priors']
    assert ref['innovation_sd']==pytest.approx(dict(level=.01/z,trend=.0001/z,season=.01/z))
    assert ref['initial_slope_sd']==pytest.approx(.01/z)
    assert ref.get('shared_shrinkage') is None and ref.get('independent_shrinkage') is None
    for name,factor in [('half_all_shrinkage',.5),('double_all_shrinkage',2.)]:
        p=configs[name]['priors']
        assert p['innovation_sd']==pytest.approx({k:v*factor for k,v in ref['innovation_sd'].items()})
        assert p['initial_slope_sd']==pytest.approx(ref['initial_slope_sd']*factor)
    assert not any('log2' in n or 'log4' in n or 'matched' in n for n in configs)


def test_each_physical_sensitivity_changes_only_its_declared_prior_anchor():
    cs={t.variant:jobs.task_config(t,'paper') for t in jobs.tasks()}
    for name,key in [('level','level'),('slope','trend'),('seasonal','season'),('initial_slope',None)]:
        for prefix,factor in [('half',.5),('double',2.)]:
            expected=deepcopy(cs['reference']['priors'])
            if key:expected['innovation_sd'][key]*=factor
            else:
                expected['initial_slope_sd']*=factor
            assert cs[prefix+'_'+name]['priors']==expected


def test_manifest_reuses_only_identical_completed_task(tmp_path,monkeypatch):
    task=jobs.tasks()[0];invoked=[]
    def fake_fit(config,*,directory):
        invoked.append(config['priors']['innovation_sd'].copy());directory.mkdir(parents=True)
        (directory/config['data']['series'][0]).mkdir()
        (directory/'data_window.json').write_text('{}')
        (directory/config['data']['series'][0]/'convergence.json').write_text('{"status":"passed_numerical_checks"}')
        return directory
    monkeypatch.setattr(jobs,'fit_full',fake_fit)
    result=jobs.execute(task,tier='screen',root=tmp_path)
    assert jobs.execute(task,tier='screen',root=tmp_path)==result and len(invoked)==1
    with pytest.raises(RuntimeError,match='incomplete'):collect(tmp_path,'screen',batch='posterior',require_complete=True)
    status=bx.load_config(tmp_path/'screen/collected/posterior/status.json')
    assert status['completed']==1 and len(status['missing_or_failed'])==137
    manifest=tmp_path/'screen'/task.id/'task.json'
    m=bx.load_config(manifest);m['source_sha256']='changed';bx.save_config(m,manifest)
    with pytest.raises(ValueError,match='changed'):jobs.execute(task,tier='screen',root=tmp_path)


def test_forecast_origin_assembly_preserves_cases_and_rejects_duplicates(tmp_path):
    roots=[]
    for index,origin in enumerate((100,120)):
        root=tmp_path/f'origin_{origin}';joint=root; joint.mkdir(parents=True)
        pd.DataFrame({'origin':[origin],'numerical_status':['passed_numerical_checks']}).to_csv(joint/'folds.csv',index=False)
        pd.DataFrame({'origin':[origin],'channel':['TXx'],'value':[index+.1]}).to_csv(joint/'scores.csv',index=False)
        (joint/f'convergence_{origin}.json').write_text('{"status":"passed_numerical_checks"}');roots.append(root)
    target=tmp_path/'merged';merge_folds(roots,target)
    assert pd.read_csv(target/'folds.csv').origin.tolist()==[100,120]
    assert pd.read_csv(target/'scores.csv').value.tolist()==[.1,1.1]
    assert (target/'convergence_120.json').exists()
    with pytest.raises(ValueError,match='duplicate'):merge_folds([roots[0],roots[0]],tmp_path/'duplicate')
