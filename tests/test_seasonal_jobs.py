"""Scientific configuration, task isolation and resource-routing regressions."""
from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest
import bucex as bx
from research.seasonal import jobs
from research.seasonal.collect_jobs import collect,merge_folds


def test_batches_are_disjoint_and_paper_has_only_one_main_reference():
    groups=[jobs.tasks(b) for b in ('posterior','pre2019','validation')]
    assert [len(g) for g in groups]==[25,4,37]
    assert len({t.id for g in groups for t in g})==66
    assert all(t.kind=='posterior' for t in jobs.tasks())
    for tier,nchains in [('screen',2),('paper',4)]:
        for task in jobs.tasks('all',tier=tier):
            c=jobs.task_config(task,tier)
            assert c['mcmc']['chains']==c['mcmc']['chain_workers']==nchains
            assert c['credible_interval']==.95 and c['save_fits'] and c['validation']['save_fits']
            assert 'variants' not in c  # no second application of an anchor multiplier
            if task.kind=='posterior':assert c['forecast_horizon']==120
            elif task.kind=='pre2019':assert c['forecast_horizon']==1 and c['data']['end']=='2019-05'
            else:assert c['validation']['training_ends']==[task.origin] and c['validation']['horizon']==20
    assert [t.id for t in jobs.tasks('posterior',tier='paper',resource='long')]==['posterior_reference']
    assert not jobs.tasks('posterior',tier='screen',resource='long')
    full=jobs.task_config(jobs.tasks('reference')[0],'paper')
    assert (full['mcmc']['warmup'],full['mcmc']['draws'],full['forecast_draws'])==(3000,8000,12000)


def test_matched_widths_preserve_physical_prior_rms_and_original_anchor_control():
    configs={t.variant:jobs.task_config(t,'screen') for t in jobs.tasks()}
    z=.6744897501960817
    def rms(c):
        p=c['priors'];anchors=[p['innovation_median'][x] for x in ('level','trend','season')]+[p['initial_slope_median']]
        return np.array(anchors)*np.exp(p['shared_shrinkage']['log_sd']**2)/z
    np.testing.assert_allclose(rms(configs['reference']),rms(configs['old_reference']),rtol=1e-13)
    np.testing.assert_allclose(rms(configs['reference']),rms(configs['wider_log4']),rtol=1e-13)
    assert configs['wide_original_anchors']['priors']['shared_shrinkage']['log_sd']==np.log(3)
    old=bx.load_config(jobs.CONFIG/'reference_189.json')['priors']
    for name in ('old_reference','wide_original_anchors'):
        assert configs[name]['priors']['innovation_median']==pytest.approx(old['innovation_median'])
        assert configs[name]['priors']['initial_slope_median']==pytest.approx(old['initial_slope_median'])
    # A hierarchy must not be silently tightened by the static-seasonal variant.
    assert configs['fixed_location_seasonality']['priors']['shared_shrinkage']['log_sd']==np.log(3)
    assert configs['fixed_location_seasonality']['priors']['shared_shrinkage']['components']==['level','slope']


def test_each_physical_sensitivity_changes_only_its_declared_prior_anchor():
    cs={t.variant:jobs.task_config(t,'paper') for t in jobs.tasks()}
    for name,key in [('level','level'),('slope','trend'),('seasonal','season'),('initial_slope',None)]:
        for prefix,factor in [('half',.5),('double',2.)]:
            expected=deepcopy(cs['reference']['priors'])
            if key:expected['innovation_median'][key]*=factor
            else:
                expected['initial_slope_median']*=factor;expected['initial_slope_sd']*=factor
            assert cs[prefix+'_'+name]['priors']==expected


def test_manifest_reuses_only_identical_completed_task(tmp_path,monkeypatch):
    task=jobs.tasks()[0];invoked=[]
    def fake_fit(config,*,directory):
        invoked.append(config['priors']['innovation_median'].copy());directory.mkdir(parents=True)
        (directory/'convergence.json').write_text('{"status":"passed_numerical_checks"}')
        return directory
    monkeypatch.setattr(jobs,'fit_full',fake_fit)
    result=jobs.execute(task,tier='screen',root=tmp_path)
    assert jobs.execute(task,tier='screen',root=tmp_path)==result and len(invoked)==1
    with pytest.raises(RuntimeError,match='incomplete'):collect(tmp_path,'screen',require_complete=True)
    status=bx.load_config(tmp_path/'screen/collected/posterior/status.json')
    assert status['completed']==1 and len(status['missing_or_failed'])==24
    manifest=tmp_path/'screen'/task.id/'task.json'
    m=bx.load_config(manifest);m['source_sha256']='changed';bx.save_config(m,manifest)
    with pytest.raises(ValueError,match='changed'):jobs.execute(task,tier='screen',root=tmp_path)


def test_forecast_origin_assembly_preserves_cases_and_rejects_duplicates(tmp_path):
    roots=[]
    for index,origin in enumerate((100,120)):
        root=tmp_path/f'origin_{origin}';joint=root/'joint';joint.mkdir(parents=True)
        pd.DataFrame({'origin':[origin],'numerical_status':['passed_numerical_checks']}).to_csv(joint/'folds.csv',index=False)
        pd.DataFrame({'origin':[origin],'channel':['TXx'],'value':[index+.1]}).to_csv(joint/'scores.csv',index=False)
        (joint/f'convergence_{origin}.json').write_text('{"status":"passed_numerical_checks"}');roots.append(root)
    target=tmp_path/'merged';merge_folds(roots,target)
    assert pd.read_csv(target/'folds.csv').origin.tolist()==[100,120]
    assert pd.read_csv(target/'scores.csv').value.tolist()==[.1,1.1]
    assert (target/'convergence_120.json').exists()
    with pytest.raises(ValueError,match='duplicate'):merge_folds([roots[0],roots[0]],tmp_path/'duplicate')
