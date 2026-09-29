"""Scientific and operational regression checks for the 1.9.6 study."""
from dataclasses import replace
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, genextreme
import bucex as bx
from research.seasonal.jobs import tasks,task_config,verify
from research.monthly.models import joint_model
from research.monthly.validate import validation_splits
from bucex.inference.fit.fs_utils import static_seasonal_design
from bucex.inference.fit.continuous import coefficient_prior


def reference():
    return task_config(tasks('reference',tier='screen')[0],'screen')


def test_reference_and_complete_hn_only_grid():
    c=reference();p=c['priors']
    assert p['innovation_sd']==dict(level=.01,trend=.0002,season=.01)
    assert p['initial_slope_sd']==.01 and p['baseline_sd']==10
    assert p['seasonal_initial_sd']==10 and p['seasonal_initial_basis']=='orthonormal'
    assert c['mcmc']['warmup']==c['mcmc']['draws']==1000 and c['mcmc']['chains']==2
    grid=set()
    for t in tasks('posterior'):
        cfg=task_config(t,'screen');prior=cfg['priors']
        h=prior.get('shared_shrinkage') or prior.get('independent_shrinkage')
        assert h['hyperprior']=='half_normal' and not h['pool_initial_slope']
        if t.study=='structural':grid.add((prior['innovation_sd']['level'],prior['innovation_sd']['trend']))
    assert {(a,b) for a in (.005,.01,.02) for b in (.0001,.0002,.0004,.001)}<=grid
    assert len(tasks('all'))==100


@pytest.mark.parametrize('period',[4,12])
def test_initial_cycle_exchangeable_and_sampler_uses_covariance(period):
    prior=bx.fs_priors('gaussian',period=period,seasonal_initial_sd=10,seasonal_initial_basis='orthonormal')
    cov=prior.gamma0_season.covariance_array()
    design=static_seasonal_design(period,period-1)
    np.testing.assert_allclose(design@cov@design.T,100*(np.eye(period)-np.ones((period,period))/period),atol=1e-10)
    names=['alpha_c','beta0']+[f'gamma0_season_{i+1}' for i in range(period-1)]
    mean,root=coefficient_prior(prior,names,3.,None,1.)
    np.testing.assert_allclose((root@root.T)[2:,2:],cov,atol=1e-12)
    sample=bx.draw_structural_prior(prior,40000,seed=196)['initial.seasonal']@design.T
    np.testing.assert_allclose(sample.std(axis=0),10*np.sqrt((period-1)/period),rtol=.025)
    assert np.max(abs(sample.sum(axis=1)))<1e-10


def test_long_validation_exact_windows_and_no_reused_training():
    c=reference();data=bx.load_uccle_multiseries(**c['data'])
    expected=dict(zip(range(1970,2021,5),[140]*5+[123,103,83,63,43,23]))
    windows={}
    for task in tasks('validation',tier='screen'):
        config=task_config(task,'screen');train,test=validation_splits(data,config['validation'])[0]
        year=int(task.origin[:4]);assert len(test)==expected[year]
        assert test.start==train.stop
        assert data.index[test.start]==pd.Timestamp(f'{year}-12-01')
        assert data.index[train.stop-1]==pd.Timestamp(f'{year}-09-01')
        case=(train.stop,test.start,test.stop)
        assert windows.setdefault(task.origin,case)==case
    assert len(windows)==11


@pytest.mark.parametrize('family,sign',[('gaussian',1),('gev',1),('gev',-1)])
def test_cdf_quantiles_agree_with_known_distribution(family,sign):
    n,h=20,3;loc=np.array([1.,3.,-2.]);scale=np.array([.5,1.,2.]);xi=-.2
    f=bx.Forecast(observations=np.tile(loc,(n,1)),eta=np.tile(loc,(n,1)),states=np.zeros((n,h,1)),
        parameters={'sigma':np.ones(n),'sigma_path':np.tile(scale,(n,1)),**({'xi':np.full(n,xi)} if family=='gev' else {})},
        dates=np.arange(h),family=family,tail='lower' if sign<0 else 'upper',
        observation_model=bx.Gaussian() if family=='gaussian' else bx.GEV(),transform_sign=sign)
    probs=np.array([.005,.025,.5,.975,.995])
    expected=(norm.ppf(probs[:,None],loc=loc,scale=scale) if family=='gaussian' else
              sign*genextreme.ppf((probs if sign>0 else 1-probs)[:,None],-xi,loc=sign*loc,scale=scale))
    np.testing.assert_allclose(f.predictive_quantiles(probs),expected,atol=1e-6)
    f.quantile_method='cdf'
    np.testing.assert_allclose(f.summary(.95)['lower'],expected[1],atol=1e-6)


def test_vectorized_forecast_matches_analytic_state_variance():
    from types import SimpleNamespace
    from bucex.api.predict import posterior_predict
    channels=(bx.Channel('a',bx.Gaussian(),[bx.LocalLinearTrend()]),)
    model=bx.MultiSeriesModel(channels,copula=None)
    compiled=bx.compile_model(model,pd.DataFrame({'a':np.zeros(8)}))
    params={'sigma.a':np.array([.7])}
    for name in compiled.noise_names:params['sd.'+name]=np.array([.1 if 'level' in name else .02])
    states=np.zeros((1,1,9,compiled.state_dim));states[0,0,-1]=[2.,.03]
    fit=SimpleNamespace(model=model,compiled=compiled,n_draws=1,n_time=8,is_multiseries_model=True,
        state_draws=states,parameter_draws=params,parameter=lambda n:params[n],dates=None,
        channel_names=('a',),state_names=compiled.state_names,transform_sign=np.ones(1),family='gaussian')
    f=posterior_predict(fit,40,draws=20000,seed=196)
    h=np.arange(1,41);v=.1**2*h+.02**2*h*(h-1)*(2*h-1)/6
    np.testing.assert_allclose(f.eta[:,:,0].mean(axis=0),2+.03*h,atol=.06)
    np.testing.assert_allclose(f.eta[:,:,0].var(axis=0),v,rtol=.04)
    np.testing.assert_allclose(f.observations[:,:,0].var(axis=0),v+.7**2,rtol=.04)


def test_submission_uses_all_arrays_then_collection(tmp_path,monkeypatch):
    monkeypatch.setenv('BUCEX_RESULTS_ROOT',str(tmp_path))
    monkeypatch.setenv('BUCEX_PYTHON',sys.executable)
    r=subprocess.run([sys.executable,'job_scripts/submit.py','screen','all','--dry-run'],text=True,capture_output=True)
    assert r.returncode==0,r.stderr
    assert '--array=1-90%48' in r.stdout and '--array=1-4%48' in r.stdout
    assert '--dependency=afterany:ARRAY_JOB_IDS' in r.stdout
    assert 'half_cauchy' not in r.stdout
