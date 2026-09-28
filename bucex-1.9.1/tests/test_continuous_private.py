"""Independent numerical targets and contracts for the 1.6.4 private kernel."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm
import bucex as bx
from bucex.inference.fit.continuous import gaussian_reference, ShrinkageState, coefficient_step, interweave_scales
from bucex.inference.fit.conditional_margin import ConditionalMargin
from bucex.inference.fit.private_channel import _initial_channel_state
from bucex.inference.fit import fs_utils as fs


def test_gaussian_reference_matches_dense_conditioning_without_floor():
    rng=np.random.default_rng(3)
    X=rng.normal(size=(20,4)); X[:,3]*=1e10
    L=np.diag([3.,.2,.01,1e-12]); L[0,1]=.5
    pm=np.array([1.,.02,0.,0.]); y=rng.normal(size=20); R=np.linspace(.3,2.,20)
    m,root,P=gaussian_reference(X,y,R,pm,L)
    V=L@L.T; S=np.diag(R)+X@V@X.T
    expected=pm+V@X.T@np.linalg.solve(S,y-X@pm)
    covariance=V-V@X.T@np.linalg.solve(S,X@V)
    np.testing.assert_allclose(pm+L@m,expected,rtol=1e-9,atol=1e-13)
    np.testing.assert_allclose(L@root@root.T@L.T,covariance,rtol=1e-8,atol=1e-24)


def test_exact_gev_coefficient_slice_matches_quadrature():
    model=bx.Model(bx.GEV(),[bx.LocalLinearTrend(level_mode='static',trend_mode='off')])
    y=np.array([-.4,.1,.7,1.2,.3]); compiled=bx.compile_model(model,y)
    prior=bx.fs_priors('gev',period=None,innovation='normal',initial_level=bx.NormalPrior(0,1))
    rng=np.random.default_rng(871)
    state=_initial_channel_state('a','gev',y,compiled,prior,rng,{})
    state.params_state.update(s_level=0.,s_trend=0.,s_season=0.)
    state.params_obs.update(sigma=1.,sigma2=1.,xi=-.15)
    margin=ConditionalMargin(model.observation,0.,1.)
    mix=ShrinkageState.initialize(prior,state.layout)
    def density(alpha):
        return np.exp(np.sum(model.observation.logpdf(y,alpha,sigma=1.,xi=-.15)))*norm.pdf(alpha)
    normalizer=quad(density,-7.,7.,epsabs=1e-11)[0]
    truth=quad(lambda a:a*density(a),-7.,7.,epsabs=1e-11)[0]/normalizer
    draws=[]
    for iteration in range(3500):
        coefficient_step(state,margin,state.params_obs,prior,mix,bx.Laplace(),rng)
        if iteration>=500: draws.append(state.params_state['alpha0'])
    assert np.mean(draws)==pytest.approx(truth,abs=.035)


@pytest.mark.parametrize('family',['normal','lasso','triple_gamma'])
def test_prior_calibration_and_no_spike(family):
    prior=bx.fs_priors('gaussian',innovation=family)
    draws=bx.draw_structural_prior(prior,60000,seed=55)
    for key,median in [('level',.01),('slope',.00005),('seasonal',.02)]:
        assert np.median(draws['sd.'+key])==pytest.approx(median,rel=.035)
        assert np.all(draws['sd.'+key]>0)


def test_lasso_zero_coefficient_has_non_degenerate_gamma_conditional():
    model=bx.Model(bx.Gaussian(),[bx.LocalLinearTrend(trend_mode='off')])
    layout=fs.infer_ncp_layout(model); prior=bx.fs_priors('gaussian',period=None,innovation='lasso')
    rng=np.random.default_rng(42); values=[]
    for _ in range(6000):
        tau,lam=fs.update_lasso_scales({'s_level':0.},{'level':1.},1.,prior,layout,variance_scale=1.,rng=rng)
        values.append(tau['level']); assert lam==1.
    assert np.mean(values)==pytest.approx(1.,abs=.055)
    assert np.var(values)==pytest.approx(2.,rel=.12)


def test_asis_preserves_tiny_centered_paths_without_scale_floor():
    model=bx.Model(bx.Gaussian(),[bx.LocalLinearTrend(),bx.DummySeasonal(4)])
    rng=np.random.default_rng(33); prior=bx.fs_priors('gaussian',period=4)
    state=_initial_channel_state('a','gaussian',np.arange(12.),bx.compile_model(model,np.arange(12.)),prior,rng,{})
    G,Q=fs.build_ncp_system(state.layout)
    for t in range(1,len(state.z_path)):
        state.z_path[t]=G@state.z_path[t-1]+np.sqrt(np.diag(Q))*rng.normal(size=len(Q))
    state.params_state.update(s_level=1e-17,s_trend=1e-18,s_season=1e-16)
    before=fs.map_ncp_to_centered(state.z_path,state.params_state,state.layout)
    mix=ShrinkageState.initialize(prior,state.layout)
    for _ in range(10): interweave_scales(state,prior,mix,rng)
    after=fs.map_ncp_to_centered(state.z_path,state.params_state,state.layout)
    np.testing.assert_allclose(after,before,atol=1e-14,rtol=1e-14)
    assert abs(state.params_state['s_level'])<1e-15


@pytest.mark.parametrize('family',['normal','lasso','triple_gamma'])
def test_independence_copula_preserves_every_continuous_kernel_draw(family):
    model=bx.MultiSeriesModel([bx.Channel('a',bx.Gaussian(),[bx.LocalLinearTrend()]),
                               bx.Channel('b',bx.GEV(),[bx.LocalLinearTrend()])])
    y=np.random.default_rng(2).normal(size=(12,2))
    priors=bx.MarginalPriors({c.name:bx.fs_priors(c.family,period=None,innovation=family) for c in model.channels})
    options=dict(priors=priors,parameterization='fs',asis=True,mcmc=bx.MCMC(draws=2,warmup=2,chains=1,seed=87))
    a=bx.fit(y,model,**options)
    b=bx.fit(y,replace(model,copula=bx.GaussianCopula(correlation=np.eye(2))),**options)
    np.testing.assert_array_equal(a.state_draws,b.state_draws)
    for key in a.parameter_draws:
        np.testing.assert_array_equal(a.parameter_draws[key],b.parameter_draws[key])
    assert not a.metadata['structural_ssvs']
    assert not any(k.startswith('state.') for k in a.parameter_draws)


def test_scale_and_seasonal_copula_calendar_forecast_restart(tmp_path):
    scale=bx.LogScale('constant',bx.SeasonalScale(12))
    channels=[bx.Channel('a',bx.Gaussian(scale=scale),[bx.LocalLinearTrend(trend_mode='static')]),
              bx.Channel('b',bx.GEV(scale=scale),[bx.LocalLinearTrend()])]
    model=bx.MultiSeriesModel(channels,copula=bx.SeasonalGaussianCopula())
    assert bx.MultiSeriesModel.from_dict(model.to_dict())==model
    dates=pd.date_range('2000-03-01',periods=12,freq='MS')
    data=pd.DataFrame(np.random.default_rng(2).normal(size=(12,2)),index=dates,columns=['a','b'])
    prior=bx.MarginalPriors({c.name:bx.fs_priors(c.family,period=None,innovation='normal') for c in channels})
    options=dict(priors=prior,parameterization='fs',asis=True,mcmc=bx.MCMC(draws=2,warmup=1,chains=1,seed=18))
    fit=bx.fit(data,model,**options)
    fit.save(tmp_path/'fit.bucex'); loaded=bx.load_fit(tmp_path/'fit.bucex')
    future=loaded.forecast(2,seed=23)
    np.testing.assert_array_equal(future.parameters['copula_phase'],[[3,4],[3,4]])
    assert fit.copula_correlation_draws().shape==(2,12,2,2)
    assert len(fit.copula_summary())==12
    assert np.all(np.isfinite(future.joint_log_score(np.ones((2,2)))))
    assert len(bx.residual_dependence_check(fit,draws=2))==1
    assert future.compound_probability_draws({'a':('>',3.),'b':('>',3.)}).shape==(2,2)
    restart=bx.fit(data,model,init=loaded,**options)
    assert restart.sigma_draws(channel='b').shape==(2,12)
    assert 'sd.channel.a.slope' not in fit.parameter_draws


def test_seasonal_prior_baseline_and_zero_effects():
    cp=bx.SeasonalGaussianCopula(structure='seasons')
    params=cp.initial_parameters(['a','b','c'])
    np.testing.assert_array_equal(cp.correlation_matrix(params,['a','b','c']),np.repeat(np.eye(3)[None],12,axis=0))
    params['copula.effect.1.0.0']=.4
    R=cp.correlation_matrix(params,['a','b','c'])
    np.testing.assert_array_equal(R[0],R[1]); np.testing.assert_array_equal(R[0],R[11])
    assert np.all(np.linalg.eigvalsh(R)>0)
    assert not np.allclose(R[0],R[6])


def test_bivariate_quadrature_rare_events_and_known_identity():
    from bucex.dependence.probability import bivariate_normal_cdf
    assert bivariate_normal_cdf(0,0,.7)==pytest.approx(.25+np.arcsin(.7)/(2*np.pi),rel=1e-9)
    assert bivariate_normal_cdf(-7,-7,0)==pytest.approx(norm.cdf(-7)**2,rel=1e-12,abs=0.)
    p=bivariate_normal_cdf(-7,-7,.7)
    assert norm.cdf(-7)**2<p<norm.cdf(-7)


def test_config_inheritance_cycles_and_replacements(tmp_path):
    bx.save_config({'model':{'scale':'constant','season':True},'list':[1,2]},tmp_path/'base.json')
    bx.save_config({'extends':'base.json','model':{'scale':'rw'},'list':[3]},tmp_path/'child.json')
    assert bx.load_config(tmp_path/'child.json')=={'model':{'scale':'rw','season':True},'list':[3]}
    bx.save_config({'extends':'child.json'},tmp_path/'base.json')
    with pytest.raises(ValueError,match='cycle'): bx.load_config(tmp_path/'child.json')


def test_static_auto_route_preserves_generic_api_and_accepts_explicit_fs_priors():
    model=bx.Model(bx.Gaussian(),[bx.LocalLinearTrend(trend_mode='static')])
    options=dict(mcmc=bx.MCMC(draws=2,warmup=1,chains=1,seed=91))
    generic=bx.fit(np.arange(6.),model,**options)
    continuous=bx.fit(np.arange(6.),model,priors=bx.fs_priors('gaussian',period=None),**options)
    assert generic.plan.parameterization=='disturbance'
    assert continuous.plan.parameterization=='fruehwirth_schnatter'
    assert 'sd.slope' not in continuous.parameter_draws


def test_shape_prior_predictions_respect_asymmetric_model_support():
    prior=bx.fs_priors('gev',xi_max_abs=.75)
    draws=bx.draw_structural_prior(prior,10000,seed=71,xi_bounds=(-.75,.2))
    assert np.all((draws['xi']>-.75)&(draws['xi']<.2))
