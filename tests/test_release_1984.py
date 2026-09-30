"""Validation dates, paired targets, resource isolation and actual forecast outputs."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import pytest
import bucex as bx
from research.seasonal.jobs import tasks, task_config, PROJECT
from research.seasonal.job_plan import plan
from research.seasonal.horizon_plan import specification
from research.seasonal.horizon_diagnostics import annotate
from research.seasonal.horizon_report import summarize, paired_origins, paired_models
from research.monthly.validate import validation_splits


def test_inventory_and_tier_resources():
    assert bx.__version__ == '1.9.8.4'
    for tier, nchains in [('screen',2),('paper',4)]:
        ref=tasks('horizon_reference',tier=tier)
        fixed=tasks('horizon_fixed',tier=tier)
        assert len(ref)==10 and len(fixed)==60
        assert len({t.id for t in tasks('horizon_all',tier=tier)})==70
        assert [t.origin for t in ref[:3]]==['1956-11','1976-11','1996-11']
        for task in ref+fixed:
            c=task_config(task,tier)
            assert c['mcmc']['chains']==c['mcmc']['chain_workers']==nchains
            assert c['validation']['horizon']==120 and c['validation']['save_fits']
            assert c['validation']['draws']==(20000 if tier=='screen' else 50000)
            assert c['priors']['innovation_sd']==dict(level=.1,trend=.002,season=.1)
            assert c['priors']['initial_slope_sd']==.01
            assert c['data']['start']=='1892-01' and c['horizon_validation']
            assert c['point_summary']=='mean' and c['credible_interval']==.95
            assert (c['priors'].get('shared_shrinkage') is not None)==(task.scope=='shared')
        for g in plan(tier,'horizon_all'):
            assert len(g['task_ids'])==1 and g['cpus']==g['required_workers']==nchains
        assert not {t.id for t in ref+fixed}&{t.id for t in tasks('final_paper',tier=tier)}


def test_real_calendar_cutoffs_and_partial_windows():
    expected={'1956-11':120,'1966-11':120,'1976-11':120,'1986-11':120,
              '1996-11':119,'2000-11':103,'2006-11':79,'2010-11':63,'2015-11':43,'2020-11':23}
    for task in tasks('horizon_reference',tier='screen'):
        c=task_config(task,'screen');data=bx.load_uccle_multiseries(**c['data'])
        ((train,test),)=validation_splits(data,c['validation'])
        origin=int(task.origin[:4])
        assert data.index[train.stop-1]==pd.Timestamp(f'{origin}-09-01')
        assert data.index[test.start]==pd.Timestamp(f'{origin}-12-01')
        assert len(test)==expected[task.origin]
        assert train.stop==test.start and train.start==0
        assert data.index[test.stop-1]<=pd.Timestamp('2026-06-01')


def evidence(origin,n,bias=-.3,crps=.8,variant='reference'):
    time=pd.date_range(f'{origin}-12-01',periods=n,freq='QS-DEC')
    observed=np.asarray(time.year,dtype=float)/100
    frame=pd.DataFrame(dict(time=time,horizon=np.arange(1,n+1),channel='TXx',
        origin_date=f'{origin}-11',variant=variant,observed=observed,
        predictive_mean=observed+bias,crps=crps,pit=.5,numerical_status='needs_review'))
    for low,high in [('q050','q950'),('q025','q975'),('q005','q995')]:
        frame[low]=observed-1;frame[high]=observed+1
    return annotate(frame)


def test_partial_is_not_called_a_complete_thirty_year_window():
    out=summarize(evidence(1996,119),specification())
    all30=out.loc[out.window.eq('first 30 years')&out.season.eq('ALL')].iloc[0]
    assert all30.n_cases==119 and all30.n_requested==120 and not all30.complete
    last=out.loc[out.window.eq('21-30 years')&out.season.eq('ALL')].iloc[0]
    assert last.n_cases==39 and last.n_requested==40 and not last.complete
    summer=out.loc[out.window.eq('first 30 years')&out.season.eq('JJA')].iloc[0]
    autumn=out.loc[out.window.eq('first 30 years')&out.season.eq('SON')].iloc[0]
    assert summer.n_cases==30 and summer.complete
    assert autumn.n_cases==29 and not autumn.complete
    assert last.bias==pytest.approx(-.3) and last.coverage95==1
    assert last.width95==2 and last.upper_misses95==0


def test_common_calendar_pairing_is_identical_observations_different_leads():
    cases=pd.concat([evidence(1976,120,-.6,.8),evidence(1996,119,-.2,.4)],ignore_index=True)
    out=paired_origins(cases,specification())
    row=out.loc[out.calendar_window.eq('1997-2006')&out.season.eq('ALL')].iloc[0]
    assert row.n_cases==40 and row.complete
    assert (row.earlier_first_lead,row.earlier_last_lead)==(81,120)
    assert (row.later_first_lead,row.later_last_lead)==(1,40)
    assert row.crps_later_minus_earlier==pytest.approx(-.4)
    fixed=evidence(1996,119,-.1,.3,variant='fixed_reference')
    comp=paired_models(pd.concat([cases,fixed]),specification())
    row=comp.loc[comp.window.eq('available')&comp.season.eq('ALL')].iloc[0]
    assert row.n_cases==119 and row.crps_fixed_minus_pooled==pytest.approx(-.1)


def test_launcher_resolves_environment_instead_of_inheriting_empty_venv(tmp_path):
    env=os.environ.copy();env.pop('BUCEX_VENV',None)
    env.update(BUCEX_ENV_SETUP='/environment.sh',BUCEX_PYTHON='/bin/python',
        BUCEX_RESULTS_ROOT=str(tmp_path),VSC_ARRAY_LIMIT='10')
    result=subprocess.run(['bash','RUN_VALIDATION_HPC.sh','screen','reference','--dry-run'],
        cwd=PROJECT,env=env,capture_output=True,text=True,check=True)
    assert '--array=1-10%10' in result.stdout
    assert '--cpus-per-task=2' in result.stdout
    assert '--dependency=afterok:PROBE_JOB_ID' in result.stdout
    assert '10 HPC experiments; 10 fits' in result.stdout
    assert 'bx1984_' in result.stdout and 'bx1983_' not in result.stdout


def test_real_parallel_validation_and_report(tmp_path,monkeypatch):
    """A real short two-chain fit, 119 forecast seasons and honest needs-review status."""
    import research.seasonal.jobs as jobs
    from research.monthly.validate import validate
    from research.seasonal.horizon_report import finish
    task=next(t for t in tasks('horizon_reference',tier='screen') if t.origin=='1996-11')
    c=task_config(task,'screen')
    c['data']['start']='1990-03'
    c['mcmc'].update(chains=2,chain_workers=2,warmup=4,draws=12,progress=False)
    c['validation'].update(draws=96)
    c['prior_draws']=40
    directory=tmp_path/'screen'/task.id
    validate(c,directory=directory/'report')
    report=directory/'report/joint'
    paths=pd.read_csv(report/'forecast_paths.csv')
    assert len(paths)==119*6 and np.isfinite(paths.predictive_mean).all()
    assert (paths.q005<=paths.q025).all() and (paths.q975<=paths.q995).all()
    fold=pd.read_csv(report/'folds.csv').iloc[0]
    assert fold.horizon==119 and fold.partial_horizon
    assert fold.numerical_status=='needs_review'
    counts=pd.read_csv(report/'risk_count_windows.csv')
    row=counts.loc[counts.channel.eq('TXx')&counts.threshold.eq(35)&counts.season.eq('JJA')&counts.window_kind.eq('available')].iloc[0]
    assert row.n_cases==30 and row.count_lower95<=row.count_upper95
    pmf=pd.read_csv(report/'risk_count_pmf.csv')
    np.testing.assert_allclose(pmf.groupby(['channel','threshold','window_kind','window','season']).predictive_probability.sum(),1)
    fit=bx.load_fit(report/f'fit_{int(fold.origin)}.bucex')
    assert fit.n_chains==2 and fit.metadata['shrinkage_scale_update']=='gig'
    # QA manifest/config describe this deliberately tiny fit, not a production run.
    monkeypatch.setattr(jobs,'tasks',lambda *args,**kwargs:[task])
    monkeypatch.setattr(jobs,'task_config',lambda *args,**kwargs:c)
    bx.save_config(dict(status='completed',numerical_status='needs_review',
        bucex_version=bx.__version__,**jobs.fingerprint(c)),directory/'task.json')
    out=finish(tmp_path,'screen','horizon_reference')
    html=(out/'report.html').read_text()
    assert 'Completed 1/1 fits; 0 pass' in html
    assert 'data:image/png;base64,' in html and '119 of 120' in html
    assert (out/'origin_horizon_metrics.csv').exists()
    assert (out/'figures/reference_TXx_forecasts.pdf').exists()
    assert list((tmp_path/'exports').glob('*.zip'))
