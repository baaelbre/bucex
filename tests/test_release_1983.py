"""Final-suite budgets, dispatch, calendar matching and reporting semantics."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from research.seasonal.jobs import tasks,task_config,PROJECT
from research.seasonal.job_plan import plan
from research.seasonal.bundles import members
from research.seasonal.overnight import make_queue


def test_final_inventory_has_one_long_reference_and_no_duplicate_settings():
    selected=tasks('final_paper',tier='paper')
    assert len(selected)==123
    assert len({t.id for t in selected})==123
    q=make_queue(('paper',),'final_paper')
    assert q[0].task.id=='final_posterior_reference'
    assert sum(t.sampling_role=='reference' for t in selected)==1
    posterior_settings=set()
    for t in selected:
        c=task_config(t,'paper');m=c['mcmc']
        assert c['credible_interval']==.95 and c['point_summary']=='mean'
        assert c['forecast_uncertainty_levels']==[.95,.99]
        if t.sampling_role=='reference':
            assert (m['chains'],m['warmup'],m['draws'])==(4,2000,4000)
            assert c['diagnostic_thresholds']==dict(min_chains=4,max_rhat=1.01,min_ess=400)
        else:
            assert (m['chains'],m['warmup'])==(2,500 if t.frequency=='monthly' else 1000)
            assert m['draws']==(500 if t.frequency=='monthly' else 1000)
        assert m['chain_workers']==m['chains']<=4
        if t.kind=='posterior':
            assert c['forecast_horizon']==(360 if t.frequency=='monthly' else 120)
            assert c['forecast_draws']==50000
        if t.kind=='pre2019':
            assert c['data']['end']=='2019-05' and c['forecast_horizon']==1
        if t.kind=='posterior' and t.frequency=='seasonal':
            signature=json.dumps([c['priors'],c['model'],c['data']['series']],sort_keys=True)
            assert signature not in posterior_settings
            posterior_settings.add(signature)
    for g in plan('paper','final_paper'):
        ts=members(g,'paper')
        assert sum(task_config(t,'paper')['mcmc']['chain_workers'] for t in ts)==g['cpus']
    assert {t.id for t in tasks('final_reference',tier='paper')} | {t.id for t in tasks('final_checks',tier='paper')}=={t.id for t in selected}


def test_reference_calibration_and_monthly_conversion():
    ts=tasks('final_paper',tier='paper')
    ref=task_config(ts[0],'paper')
    assert ref['priors']['innovation_sd']==dict(level=.1,trend=.002,season=.1)
    assert ref['priors']['shared_shrinkage']['hyperprior']=='half_normal'
    assert ref['priors']['initial_slope_sd']==.01
    assert ref['priors']['baseline_sd']==ref['priors']['seasonal_initial_sd']==10
    gain=lambda h:h*(h-1)*(2*h-1)/6
    for t in ts:
        if t.frequency!='monthly':continue
        c=task_config(t,'paper');p=c['priors']
        assert p['innovation_sd']['level']*np.sqrt(360)==pytest.approx(.1*np.sqrt(120))
        assert p['innovation_sd']['trend']*np.sqrt(gain(360))==pytest.approx(.002*np.sqrt(gain(120)))
        assert p['initial_slope_sd']*120==pytest.approx(.4)


def test_hpc_submission_starts_reference_before_check_arrays(tmp_path,monkeypatch):
    import job_scripts.submit as submit
    commands=[]
    def fake_run(args,**kwargs):
        commands.append(args)
        return subprocess.CompletedProcess(args,0,str(2000+len(commands))+'\n','')
    monkeypatch.setattr(submit.subprocess,'run',fake_run)
    monkeypatch.setenv('BUCEX_PYTHON',sys.executable)
    monkeypatch.setenv('BUCEX_RESULTS_ROOT',str(tmp_path))
    monkeypatch.delenv('BUCEX_ENV_SETUP',raising=False)
    assert submit.main(['paper','final_paper','--scheduler','slurm'])==0
    assert '--array=1-1%48' in commands[1]
    assert '--cpus-per-task=4' in commands[1]
    for cmd in commands[2:-1]:
        assert '--dependency=afterok:2001,after:2002' in cmd
    assert '--dependency=afterany:2002:2003:2004:2005' in commands[-1]


def test_smoke_fit_and_mean_figures(tmp_path):
    """Real four-process inference/report path with intentionally tiny chains."""
    import bucex as bx
    import pandas as pd
    from research.monthly.run import run
    from research.seasonal.final_report import reference_figures
    t=tasks('final_reference',tier='paper')[0]
    c=task_config(t,'paper')
    c['data']['start']='2020-03'
    c['risk_periods']=dict(reference=['2020-03','2023-02'],comparison=['2023-03','2026-02'])
    c['mcmc'].update(chains=4,chain_workers=4,warmup=4,draws=10,progress=False)
    c.update(figures=False,forecast_draws=80,predictive_check_draws=24,prior_draws=80,
             annual_risk_draws=24,trace_exports=True,sweetspot_diagnostics=False)
    directory=tmp_path/'paper'/t.id
    run(c,directory=directory/'report')
    report=directory/'report'
    fit=bx.load_fit(report/'fit.bucex')
    assert fit.metadata['shrinkage_scale_update']=='gig'
    mean=fit.component_draws('level',channel='TXx').mean(axis=0)
    np.testing.assert_allclose(pd.read_csv(report/'TXx_level.csv')['mean'],mean)
    f=pd.read_csv(report/'TXx_forecast_uncertainty.csv')
    assert len(f)==720 and set(f.nominal)=={.95,.99} and 'mean' in f
    bx.save_config(dict(status='completed',final_check='failed'),directory/'task.json')
    output=reference_figures(tmp_path)
    assert output.name=='diagnostic_figures_NOT_FINAL'
    assert (output/'seasonal_forecast_uncertainty_30y.png').exists()
    assert (output/'seasonal_levels.png').exists()
    assert 'Posterior means' in bx.load_config(output/'figure_manifest.json')['point_summary']
    bx.SensitivityReport(posterior_runs={'reference':{'TXx':report}},baseline='reference',
                         block_frequency='seasonal').save(tmp_path/'sensitivity',figures=True)
    assert (tmp_path/'sensitivity'/'TXx_level.png').exists()
