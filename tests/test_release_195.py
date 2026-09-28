"""Pooled half-family targets, calibration, reporting and queue contracts."""
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm,t
import bucex as bx
from bucex.inference.fit.shrinkage import shared_scale_log_target,SharedShrinkageState
from research.seasonal import jobs,overnight
from research.monthly.models import joint_model,fit_options
from research.monthly.report import write_report


def config(name='reference'):
    task=next(t for t in jobs.tasks('posterior') if t.variant==name)
    return jobs.task_config(task,'screen')


@pytest.mark.parametrize('family,nu',[('half_normal',4),('half_t',4),('half_cauchy',1)])
def test_log_scale_target_matches_density_and_jacobian(family,nu):
    a=.01;coefficients=np.array([.003,-.008,.014,-.001,.018,.006])
    def exact(u):
        tau=a*np.exp(u)
        hp=norm.logpdf(tau,scale=a) if family=='half_normal' else t.logpdf(tau/a,df=nu)-np.log(a)
        return norm.logpdf(coefficients,scale=tau).sum()+np.log(2)+hp+np.log(tau)
    def target(u):return shared_scale_log_target(u,coefficients,anchor=a,scale_conversion=1.,hyperprior=family,df=nu)
    for u in (-5.,-2.,-.25,0.,1.,3.):
        assert target(u)-target(0)==pytest.approx(exact(u)-exact(0),rel=1e-11,abs=1e-9)


def test_half_normal_slice_samples_the_conditional_distribution():
    a=.01;values=[.003,-.008,.014,-.001,.018,.006]
    spec=bx.SharedShrinkage.half_normal({'level':a})
    states=[SimpleNamespace(params_state={'s_level':x}) for x in values]
    sampler=SharedShrinkageState(spec,{'level':0.},{'level':states})
    target=lambda u:shared_scale_log_target(u,values,anchor=a,scale_conversion=1,hyperprior='half_normal')
    normalizer=quad(lambda u:np.exp(target(u)),-8,5)[0]
    expected=quad(lambda u:a*np.exp(u+target(u)),-8,5)[0]/normalizer
    rng=np.random.default_rng(195);draws=[]
    for i in range(6500):
        sampler.update(rng)
        if i>=500:draws.append(sampler.medians['level'])
    assert np.mean(draws)==pytest.approx(expected,rel=.025)


def test_hyperprior_calibration_and_pooling():
    data=bx.load_uccle_multiseries(**config()['data'])
    _,reference=joint_model(data,config())
    _,ht=joint_model(data,config('half_t4'))
    _,hc=joint_model(data,config('half_cauchy'))
    for component in reference.shrinkage.anchors:
        a=reference.shrinkage.anchors[component]
        assert ht.shrinkage.anchors[component]*ht.shrinkage.rms_multiplier==pytest.approx(a)
        assert hc.shrinkage.scale_quantile(component,.95)==pytest.approx(reference.shrinkage.scale_quantile(component,.95))
    assert np.isinf(hc.shrinkage.rms_multiplier)
    prior=bx.draw_marginal_prior(reference,80000,seed=195)
    for name in ('TXm','TNm'):
        values=prior['channels'][name]['sd.level']
        assert np.mean(values**2)==pytest.approx(.01**2,rel=.035)
        assert np.std(prior['channels'][name]['initial.slope'])==pytest.approx(.01,rel=.015)
    assert np.corrcoef(prior['channels']['TXm']['sd.level'],prior['channels']['TNm']['sd.level'])[0,1]>.35


def test_complete_queue_excludes_deferred_and_matches_monthly_effects():
    for tier in ('screen','paper'):
        ts=jobs.tasks('all',tier=tier)
        assert len(ts)==106 and len({t.id for t in ts})==106
        assert all(t.scope=='shared' for t in ts)
        assert len(jobs.tasks('deferred',tier=tier))==12
        assert not set(t.id for t in ts).intersection(t.id for t in jobs.tasks('deferred'))
    c=config();monthly=jobs.task_config(next(t for t in jobs.tasks('all') if t.frequency=='monthly'),'paper')
    for z in (c,monthly):
        d=bx.load_uccle_multiseries(**z['data']);_,p=joint_model(d,z)
        table=pd.DataFrame(p.shrinkage.calibration(period=z['model']['period'],**z['prior_calibration'])).set_index('component')
        if z is c:expected=table.displacement_sd_marginal
        else:np.testing.assert_allclose(table.displacement_sd_marginal,expected)
    assert monthly['priors']['initial_slope_sd']*120==pytest.approx(.4)
    queue=overnight.make_queue(('screen','paper'),'all')
    assert len(queue)==212 and queue[0].task.id=='posterior_reference' and queue[0].tier=='paper'


@pytest.mark.parametrize('tail',['upper','lower'])
def test_annual_return_level_solves_product_cdf(tail):
    from tests.test_calendar_forecasts import forecast
    # This helper uses monthly dates; a full year gives an independent check.
    f=forecast(family='gev',tail=tail)
    annual=f.aggregate(frequency='year')
    for r in (10,100):
        q=annual.return_level(r)
        np.testing.assert_allclose(annual.conditional_cdf(q),1/r if tail=='lower' else 1-1/r,atol=1e-10)


@pytest.mark.parametrize('variant',['reference','half_t4','half_cauchy'])
def test_half_family_fit_archive_risk_report(tmp_path,variant):
    c=config(variant);c['data'].update(start='2021-03')
    c['mcmc'].update(warmup=2,draws=8,chain_workers=1,progress=False)
    c.update(forecast_horizon=8,forecast_draws=24,predictive_check_draws=12,prior_draws=80,figures=False)
    data=bx.load_uccle_multiseries(**c['data']);model,prior=joint_model(data,c)
    fit=bx.fit(data,model,priors=prior,**fit_options(c,family=model.family))
    write_report(fit,tmp_path,config=c,risks=c['risks'],horizon=8)
    restored=bx.load_fit(tmp_path/'fit.bucex')
    assert restored.priors.shrinkage.hyperprior==prior.shrinkage.hyperprior
    frame=pd.read_csv(tmp_path/'shared_shrinkage.csv')
    assert frame.credible_interval.eq(.95).all()
    if variant=='reference':
        row=frame.query("component=='level' and distribution=='prior'").iloc[0]
        assert row['median']==pytest.approx(.01*norm.ppf(.75))
    annual=pd.read_csv(tmp_path/'TXx_forecast_annual_return_levels.csv')
    assert annual.credible_interval.eq(.95).all() and annual.complete.all()
    assert set(annual.return_period)=={10,20,50,100}
    assert (tmp_path/'TXn_historical_annual_risk.csv').exists()
    assert 'initial_slope' not in restored.priors.shrinkage.anchors


def test_overnight_runner_has_one_budget_and_records_failures(tmp_path,monkeypatch):
    original=overnight.make_queue(('screen','paper'),'comparison')
    chosen=original[:4]
    for w in chosen:w.memory_gb=1.
    monkeypatch.setattr(overnight,'make_queue',lambda *a:chosen)
    monkeypatch.setattr(overnight,'verify',lambda *a,**k:None)
    real_popen=subprocess.Popen
    script='''import json,os,sys,time
from pathlib import Path
start=time.time();time.sleep(.3)
Path(sys.argv[1]).write_text(json.dumps({'start':start,'end':time.time(),'threads':os.environ['OPENBLAS_NUM_THREADS']}))
sys.exit(int(sys.argv[2]))
'''
    def launch(args,**kwargs):
        tier=args[args.index('--tier')+1];task=args[args.index('--task')+1]
        return real_popen([sys.executable,'-c',script,str(tmp_path/(tier+'_'+task+'.json')),
            '1' if task==chosen[1].task.id and tier==chosen[1].tier else '0'],**kwargs)
    monkeypatch.setattr(overnight.subprocess,'Popen',launch)
    assert overnight.run(root=tmp_path/'results',cpus=4,memory_gb=10,max_jobs=8,collect=False)==1
    intervals=[json.loads((tmp_path/(w.tier+'_'+w.task.id+'.json')).read_text()) for w in chosen]
    points=sorted([(r['start'],1) for r in intervals]+[(r['end'],-1) for r in intervals])
    assert np.cumsum([p[1] for p in points]).max()==2
    assert all(r['threads']=='1' for r in intervals)


def test_pre2019_uses_only_training_then_checks_all_six_observations(tmp_path):
    from research.monthly.run import run
    task=jobs.tasks('pre2019')[0];c=jobs.task_config(task,'screen')
    c['data']['start']='2016-03'
    c['mcmc'].update(warmup=2,draws=8,chain_workers=1,progress=False)
    c.update(forecast_draws=32,predictive_check_draws=12,prior_draws=80,figures=False)
    run(c,directory=tmp_path)
    window=bx.load_config(tmp_path/'data_window.json')
    assert window['last_included_day']=='2019-05-31'
    predictions=pd.read_csv(tmp_path/'pre2019_observed_predictions.csv')
    assert set(predictions.channel)==set(bx.UCCLE_SERIES)
    assert predictions.set_index('channel').loc['TXx','observed']==pytest.approx(39.7)
    curve=pd.read_csv(tmp_path/'pre2019_TXx_risk_curve.csv')
    assert {35.,36.6,39.7}<=set(curve.threshold) and curve.credible_interval.eq(.95).all()


def test_validation_and_parallel_block_jobs_score_same_cases(tmp_path):
    from dataclasses import replace
    from research.monthly.validate import validate
    from research.seasonal.block_job import run
    task=next(t for t in jobs.tasks('validation') if t.origin=='2015-11')
    c=jobs.task_config(task,'screen');c['data']['start']='2010-03'
    c['mcmc'].update(warmup=2,draws=8,chain_workers=1,progress=False)
    c['validation'].update(horizon=4,draws=24)
    c.update(figures=False,forecast_draws=24)
    validate(c,directory=tmp_path/'validation')
    coverage=pd.read_csv(tmp_path/'validation/joint/coverage_by_case.csv')
    assert set(coverage[coverage.kind=='central_interval'].nominal)=={.9,.95,.99}
    tables=[]
    for task in [t for t in jobs.tasks('block_validation') if t.origin=='2015-11']:
        task=replace(task,horizon=4);c=jobs.task_config(task,'screen');c['data']['start']='2010-03'
        c['mcmc'].update(warmup=2,draws=8,chain_workers=1,progress=False)
        c['validation']['draws']=24;c['figures']=False
        directory=tmp_path/task.frequency
        run(c,task,directory)
        table=pd.read_csv(directory/'scores.csv')
        tables.append(table[table.score=='crps'].sort_values(['channel','time']).reset_index(drop=True))
    pd.testing.assert_series_equal(tables[0].time,tables[1].time)
    np.testing.assert_allclose(tables[0].observed,tables[1].observed,atol=1e-10)
