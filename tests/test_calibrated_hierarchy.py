"""Scientific-unit calibration and exact initial-slope hierarchy references."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import bucex as bx
from bucex.inference.fit.shrinkage import shared_scale_log_target, SharedShrinkageState
from bucex.io.posterior import _decode


def test_compstat_coefficients_and_log_normal_second_moment():
    # Independent integrated random-walk formula and seasonal impulse cycle.
    gains = bx.innovation_response_gains(360, period=12)
    assert gains['level'] == pytest.approx(np.sqrt(360))
    assert gains['slope'] == pytest.approx(np.sqrt(sum(k*k for k in range(360))))
    assert gains['seasonal'] == pytest.approx(np.sqrt(2*30))
    spec = bx.SharedShrinkage.from_effects(horizon=360,
        level_displacement_sd=.02*np.sqrt(360),
        slope_displacement_sd=.00005*np.sqrt(sum(k*k for k in range(360))),
        seasonal_displacement_sd=.02*np.sqrt(60), initial_slope_sd=.30)
    prior = spec.conditional_prior(bx.fs_priors('gaussian'), spec.anchors)
    assert prior.s_level.sd == pytest.approx(.02)
    assert prior.s_trend.sd == pytest.approx(.00005)
    assert prior.s_season.sd == pytest.approx(.02)
    assert prior.beta0.sd == pytest.approx(.30/120)
    table = pd.DataFrame(spec.calibration()).set_index('component')
    np.testing.assert_allclose(table.displacement_sd_marginal/table.displacement_sd_at_anchor,
                               np.exp(np.log(2.)**2))
    marginal = bx.SharedShrinkage.from_effects(horizon=360, level_displacement_sd=.1,
        slope_displacement_sd=.2, seasonal_displacement_sd=.3, initial_slope_sd=.3,
        calibration='marginal')
    t = pd.DataFrame(marginal.calibration()).set_index('component')
    np.testing.assert_allclose(t.loc[['level','slope','seasonal'],'displacement_sd_marginal'],[.1,.2,.3])
    assert t.loc['initial_slope','initial_rate_sd_marginal'] == pytest.approx(.3)


def test_initial_scale_conditional_normalization_and_prior_integration():
    coefficients = np.array([-.004,.001,.002,-.003,.005,.0002])
    anchor, width = .0025, np.log(2.)
    def reference(u):
        return norm.logpdf(coefficients,0,anchor*np.exp(u)).sum()+norm.logpdf(u,0,width)
    for u in [-3.,-.4,0.,1.,3.]:
        got = shared_scale_log_target(u,coefficients,anchor=anchor,log_sd=width,scale_conversion=1.)
        zero = shared_scale_log_target(0.,coefficients,anchor=anchor,log_sd=width,scale_conversion=1.)
        assert got-zero == pytest.approx(reference(u)-reference(0.),abs=1e-9)
    prior = bx.fs_priors('gaussian')
    spec = bx.SharedShrinkage({'level':.0025})
    priors = bx.MarginalPriors({'a':prior,'b':prior},shrinkage=spec)
    samples = bx.draw_marginal_prior(priors,80000,seed=182)
    scale = samples['shared']['initial_slope']
    a,b = [samples['channels'][n]['initial.slope'] for n in ['a','b']]
    assert np.mean((a/scale)**2) == pytest.approx(1.,abs=.02)
    assert abs(np.corrcoef(a,b)[0,1]) < .04  # no shared mean or signed trajectory
    assert np.corrcoef(abs(a),abs(b))[0,1] > .25
    states=[SimpleNamespace(name=n,layout=SimpleNamespace(has_beta=active),
                            params_state={'beta0':v})
            for n,active,v in [('a',True,.001),('b',True,-.002),('off',False,100.)]]
    initial_only=bx.SharedShrinkage({},initial_slope_sd=anchor)
    state=SharedShrinkageState.initialize(initial_only,states,np.random.default_rng(1))
    assert [s.name for s in state.members['initial_slope']] == ['a','b']
    state.update(np.random.default_rng(2))
    assert np.isfinite(state.parameter_values()['shrinkage.shared.initial_slope'])


def test_legacy_hierarchy_decode_does_not_add_a_new_prior():
    restored=_decode({'__dataclass__':'bucex.priors.shrinkage:SharedShrinkage',
        'fields':{'medians':{'level':.0025},'log_sd':np.log(2.)}}, {})
    assert restored.initial_slope_sd is None
    assert set(restored.anchors) == {'level'}


@pytest.mark.parametrize('xi',[-.75,.65])
def test_unrestricted_shape_support_and_bounded_opt_in(xi):
    obs=bx.GEV()
    assert obs.xi_bounds == (-np.inf,np.inf)
    assert obs.to_dict()['xi_bounds'] == [None,None]
    json.dumps(obs.to_dict(),allow_nan=False)
    p=np.array([.1,.4,.8])
    y=obs.ppf(p,0.,sigma=1.,xi=xi)
    np.testing.assert_allclose(obs.cdf(y,0.,sigma=1.,xi=xi),p)
    assert obs.support_ok(y,0.,sigma=1.,xi=xi)
    assert not obs.support_ok(-2/xi,0.,sigma=1.,xi=xi)
    prior=bx.fs_priors('gev',xi_prior=bx.NormalPrior(xi,.0001))
    draws=bx.draw_structural_prior(prior,300,seed=18)['xi']
    assert np.all(draws < -.5) if xi<0 else np.all(draws > .5)
    assert bx.GEV(xi_bounds=(None,.5)).xi_bounds == (-np.inf,.5)
    with pytest.raises(ValueError,match='finite'):
        bx.UniformPrior(-np.inf,np.inf)
    # Fit outside the old artificial bounds, retaining observation support.
    model=bx.Model(bx.GEV(scale=bx.SeasonalScale(4)),[bx.LocalLinearTrend()])
    data=pd.Series(np.tile(y,4),index=pd.date_range('2000-01-01',periods=12,freq='MS'))
    fit=bx.fit(data,model,priors=prior,init={'xi':xi},parameterization='fs',
               mcmc=bx.MCMC(chains=1,warmup=2,draws=3,seed=182))
    values=fit.parameter('xi')
    assert np.all(values < -.5) if xi<0 else np.all(values > .5)
    assert np.all(np.isfinite(fit.forecast(4,seed=4).observations))


def test_draft_configuration_matches_the_scientific_contract():
    from research.serra.preflight import inspect
    from research.serra.models import joint_model
    path=Path(__file__).resolve().parents[1]/'research/serra/config/draft'
    config=bx.load_config(path/'main.json')
    summary=inspect(config)
    assert summary['start']=='1892-03-01' and summary['end']=='2026-08-01' and summary['n_months']==1614
    assert len(summary['prior_calibration'])==4
    assert summary['chain_execution']['workers']==4
    assert config['save_fits'] and config['copula']['structure']=='seasons'
    assert config['priors']['xi_bounds']==[None,None]
    assert config['priors']['shared_shrinkage']['pool_initial_slope']
    data=bx.load_uccle_multiseries(**config['data'])
    model,priors=joint_model(data,config)
    assert priors.shrinkage.initial_slope_sd == pytest.approx(.0025*np.exp(-np.log(2.)**2))
    assert model.channel('TXx').observation.xi_bounds==(-np.inf,np.inf)
    assert np.isinf(priors.channels['TXx'].xi_max_abs)
    assert bx.load_config(path/'independent.json')['priors']['shared_shrinkage'] is None


@pytest.mark.parametrize('kwargs',[{'horizon':1,'slope_displacement_sd':.1},
    {'horizon':360,'period':None,'seasonal_displacement_sd':.1},
    {'horizon':360,'level_displacement_sd':-.1},
    {'horizon':360,'calibration':'interval'}])
def test_invalid_physical_calibration_fails(kwargs):
    with pytest.raises(ValueError):
        bx.SharedShrinkage.from_effects(**kwargs)


@pytest.mark.parametrize('bounds',[(-np.inf,np.inf),(-np.inf,.5),(-.5,np.inf),(-.5,.5)])
def test_shape_transform_inverse_and_absolute_jacobian(bounds):
    from bucex.core.numerics import bounded_to_real, real_to_bounded, bounded_log_jacobian
    for u in [-2.,-.1,.4,2.]:
        theta=real_to_bounded(u,*bounds)
        assert bounded_to_real(theta,*bounds)==pytest.approx(u)
        derivative=(real_to_bounded(u+1e-5,*bounds)-real_to_bounded(u-1e-5,*bounds))/2e-5
        assert np.log(abs(derivative))==pytest.approx(bounded_log_jacobian(u,*bounds),abs=1e-8)


@pytest.mark.parametrize('bounds',[(None,None),(None,.5),(-.5,None)])
def test_general_disturbance_shape_updates_are_not_frozen(bounds):
    model=bx.Model(bx.GEV(xi_bounds=bounds),[bx.LocalLevel(initial_mean=0.,initial_sd=1.)])
    y=bx.GEV().ppf(np.linspace(.05,.95,30),0.,sigma=1.,xi=0.)
    fit=bx.fit(y,model,priors='normal',parameterization='disturbance',engine='laplace_mh',
        mcmc=bx.MCMC(chains=1,warmup=3,draws=20,seed=182))
    assert np.all(np.isfinite(fit.parameter('xi')))
    assert len(np.unique(fit.parameter('xi'))) > 2


def test_identity_conditional_skips_score_derivatives(monkeypatch):
    from bucex.inference.fit.conditional_margin import ConditionalMargin
    observation=bx.GEV()
    conditional=ConditionalMargin(observation,0.,1.)
    def fail(*args):
        raise AssertionError('An exactly zero copula correction needs no score derivatives.')
    monkeypatch.setattr(conditional,'scores',fail)
    y=np.array([-.1,.2,.5]);eta=np.zeros(3);params={'sigma':1.,'xi':-.2}
    np.testing.assert_array_equal(conditional.logpdf(y,eta,params),observation.logpdf(y,eta,params))
    gradient,hessian=conditional.derivatives(y,eta,params)
    np.testing.assert_array_equal(gradient,observation.grad_eta(y,eta,params))
    np.testing.assert_array_equal(hessian,observation.hess_eta(y,eta,params))


def test_empty_forecast_panel_is_explicit_placeholder(tmp_path):
    from tests.test_publication_reporting import report
    directory=report(tmp_path/'source')
    pd.DataFrame(columns=['year','lower','median','upper']).to_csv(directory/'A_forecast_year.csv',index=False)
    reports=bx.ReportCollection.from_directories(directory)
    recipes=[dict(name='annual',kind='bands',table='forecast_year',x='year')]
    manifest=bx.save_publication_figures(reports,tmp_path/'figures',recipes=recipes)
    assert manifest['figures'][0]['status']=='placeholder'
    with pytest.raises(ValueError,match='no observations'):
        bx.save_publication_figures(reports,tmp_path/'strict',recipes=recipes,strict=True)
