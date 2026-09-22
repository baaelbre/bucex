"""Independent density references and API contracts for shared FS shrinkage."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm

import bucex as bx
from bucex.inference.fit.shrinkage import shared_scale_log_target
from bucex.inference.fit.fs_utils import _slice_sample_real
from bucex.priors.shrinkage import NORMAL_ABSOLUTE_MEDIAN


def mixed_problem(copula=None):
    components = [bx.LocalLinearTrend(), bx.DummySeasonal(4)]
    model = bx.MultiSeriesModel([
        bx.Channel('mean', bx.Gaussian(scale=bx.SeasonalScale(4)), components),
        bx.Channel('minimum', bx.GEV(scale=bx.SeasonalScale(4)), components, tail='lower')], copula=copula)
    data = pd.DataFrame(np.random.default_rng(9).normal(size=(20,2)), columns=model.channel_names,
                        index=pd.date_range('2000-01-01', periods=20, freq='MS'))
    priors = bx.MarginalPriors({name:bx.fs_priors(family,period=4) for name,family in
                               [('mean','gaussian'),('minimum','gev')]},
        shrinkage=bx.SharedShrinkage(medians={'level':.02,'slope':.0001,'seasonal':.02}))
    return data,model,priors


def test_hyperconditional_matches_normal_density_and_quadrature():
    coefficients = np.array([.01,-.002,.006,.015,-.004,.008])
    anchor, log_sd = .0025, np.log(2.)
    def reference(u):
        sigma = anchor*np.exp(u)/NORMAL_ABSOLUTE_MEDIAN
        return norm.logpdf(coefficients,0,sigma).sum()+norm.logpdf(u,0,log_sd)
    def target(u):
        return shared_scale_log_target(u,coefficients,anchor=anchor,log_sd=log_sd)
    for u in [-3.,-.5,0.,1.,3.]:
        assert target(u)-target(0.) == pytest.approx(reference(u)-reference(0.),abs=1e-9)
    # Integrate an independently expressed density, then check the slice kernel.
    constant=reference(1.)
    normalizer=quad(lambda u:np.exp(reference(u)-constant),-8,8)[0]
    expected=quad(lambda u:u*np.exp(reference(u)-constant),-8,8)[0]/normalizer
    rng=np.random.default_rng(171)
    u=0.; draws=[]
    for i in range(4200):
        u,_=_slice_sample_real(u,target,rng,width=.5)
        if i>=200: draws.append(u)
    assert np.mean(draws) == pytest.approx(expected,abs=.03)
    assert np.isfinite(shared_scale_log_target(-100.,np.zeros(6),anchor=anchor,log_sd=log_sd))


def test_shared_prior_integrates_hyperparameters_and_keeps_conditional_normal():
    prior=bx.fs_priors('gaussian',period=4)
    spec=bx.SharedShrinkage({'level':.01,'trend':.0001})
    conditional=spec.conditional_prior(prior,{'level':.005,'slope':.00002})
    assert conditional.s_level.sd == pytest.approx(.005/NORMAL_ABSOLUTE_MEDIAN)
    assert conditional.s_trend.sd == pytest.approx(.00002/NORMAL_ABSOLUTE_MEDIAN)
    assert conditional.s_season == prior.s_season
    priors=bx.MarginalPriors({'a':prior,'b':prior},shrinkage=spec)
    sampled=bx.draw_marginal_prior(priors,50000,seed=124)
    m=sampled['shared']['level']
    a=sampled['channels']['a']['sd.level']
    b=sampled['channels']['b']['sd.level']
    # Given each sampled shared scale, signed coefficient squares have mean one.
    assert np.mean((a*NORMAL_ABSOLUTE_MEDIAN/m)**2) == pytest.approx(1.,abs=.025)
    assert np.corrcoef(a,b)[0,1] > .25
    assert np.median(m) == pytest.approx(.01,rel=.02)
    assert np.quantile(a,.975) > 1.7*np.quantile(bx.draw_structural_prior(prior,50000,seed=1)['sd.level'],.975)


@pytest.mark.parametrize('anchors,width', [({},.5),({'foo':1.},.5),({'level':0.},.5),({'slope':1.},0.),
                                          ({'trend':1.,'slope':2.},.5),({'level':np.inf},.5)])
def test_invalid_shared_specification_is_rejected(anchors,width):
    with pytest.raises(ValueError):
        bx.SharedShrinkage(anchors,log_sd=width)


def test_conflicting_shrinkage_and_noncentered_mean_are_rejected():
    spec=bx.SharedShrinkage({'level':.01})
    with pytest.raises(ValueError,match='continuous normal'):
        bx.MarginalPriors({'a':bx.ssvs_gaussian_priors(4)},shrinkage=spec)
    with pytest.raises(ValueError,match='zero-mean'):
        bx.MarginalPriors({'a':replace(bx.fs_priors('gaussian',period=4),s_level=bx.NormalPrior(.1,.2))},shrinkage=spec)


def test_identity_copula_same_hierarchy_kernel_and_unconditional_prior():
    data,model,prior=mixed_problem()
    options=dict(priors=prior,mcmc=bx.MCMC(chains=1,warmup=2,draws=4,seed=104))
    left=bx.fit(data,model,**options)
    right=bx.fit(data,replace(model,copula=bx.GaussianCopula(correlation=np.eye(2))),**options)
    np.testing.assert_allclose(left.state_draws,right.state_draws,atol=1e-12)
    for key in left.parameter_draws:
        np.testing.assert_allclose(left.parameter_draws[key],right.parameter_draws[key],atol=1e-12)
    comparison=bx.compare_innovation_priors(left,channel='mean',size=1000,seed=7)
    direct=bx.draw_marginal_prior(prior,1000,seed=7)['channels']['mean']['sd.level']
    row=comparison.query("component == 'level' and distribution == 'prior' and scale == 'SD'").iloc[0]
    assert row['median'] == np.median(direct)
    assert left.metadata['hierarchical_innovations'] and not left.metadata['shared_temporal_state']


def test_parallel_mixed_copula_restart_and_chain_diagnostics(tmp_path):
    data,model,priors=mixed_problem(bx.GaussianCopula())
    options=dict(priors=priors,mcmc=bx.MCMC(chains=2,chain_workers=2,warmup=2,draws=4,seed=60))
    parallel=bx.fit(data,model,**options)
    options['mcmc']=replace(options['mcmc'],chain_workers=1)
    serial=bx.fit(data,model,**options)
    np.testing.assert_array_equal(parallel.state_draws,serial.state_draws)
    for component in ['level','slope','seasonal']:
        key='shrinkage.shared.'+component
        np.testing.assert_array_equal(parallel.parameter_draws[key],serial.parameter_draws[key])
        assert np.all(parallel.parameter_draws[key]>0)
    assert not np.array_equal(parallel.state_draws[0],parallel.state_draws[1])
    diagnostics=parallel.diagnostics()['parameters']
    assert 'scale.seasonal.mean[01]' in diagnostics.index
    assert 'shrinkage.shared.level' in diagnostics.index
    assert 'shrinkage.shared.seasonal' in diagnostics.index
    traces=bx.parameter_trace_draws(parallel,channel='mean')
    assert any(k.startswith('scale.seasonal.mean[') for k in traces)
    path=tmp_path/'shared.bucex'; parallel.save(path)
    restored=bx.load_fit(path)
    assert restored.priors.shrinkage==priors.shrinkage
    np.testing.assert_array_equal(restored.parameter_draws['shrinkage.shared.level'],parallel.parameter_draws['shrinkage.shared.level'])
    np.testing.assert_array_equal(restored.parameter_draws['shrinkage.shared.seasonal'],parallel.parameter_draws['shrinkage.shared.seasonal'])
    restarted=bx.fit(data,model,priors=restored.priors,init=restored,mcmc=bx.MCMC(chains=1,warmup=1,draws=2,seed=81))
    future=restarted.forecast(4,seed=23)
    assert np.all(np.isfinite(future.joint_log_score(data.to_numpy()[-4:])))
    directory=bx.save_shared_shrinkage_report(parallel,tmp_path/'report',figures=False)
    assert (directory/'shared_shrinkage_traces.csv.gz').exists()
    assert set(pd.read_csv(directory/'shared_shrinkage.csv').component)=={'level','slope','seasonal'}
    for name in parallel.channel_names:
        comparison=bx.compare_innovation_priors(parallel,channel=name,size=500,seed=114)
        prior_season=bx.draw_marginal_prior(priors,500,seed=114)['channels'][name]['sd.seasonal']
        row=comparison.query("component == 'seasonal' and distribution == 'prior' and scale == 'SD'").iloc[0]
        assert row['median']==np.median(prior_season)
    assert not np.array_equal(parallel.parameter_draws['sd.channel.mean.seasonal'],
                              parallel.parameter_draws['sd.channel.minimum.seasonal'])


def test_only_active_channels_in_hyperconditional():
    dynamic=bx.LocalLinearTrend()
    static=bx.LocalLinearTrend(level_mode='static')
    model=bx.MultiSeriesModel([bx.Channel(name,bx.Gaussian(),[component]) for name,component in
                               [('a',dynamic),('b',dynamic),('fixed',static)]])
    y=pd.DataFrame(np.random.default_rng(72).normal(size=(10,3)),columns=model.channel_names)
    priors=bx.MarginalPriors({name:bx.fs_priors('gaussian') for name in model.channel_names},
                            shrinkage=bx.SharedShrinkage({'level':.01}))
    fit=bx.fit(y,model,priors=priors,mcmc=bx.MCMC(chains=1,warmup=1,draws=2,seed=12))
    assert fit.metadata['shared_shrinkage_members']['level']==['a','b']
    # A static component has no sampled innovation coefficient to report.
    assert 'sd.channel.fixed.level' not in fit.parameter_draws
    with pytest.raises(ValueError,match='at least two'):
        bx.fit(y[['a','fixed']],bx.MultiSeriesModel([model.channels[0],model.channels[2]]),
               priors=bx.MarginalPriors({k:priors.channels[k] for k in ['a','fixed']},shrinkage=priors.shrinkage),
               mcmc=bx.MCMC(chains=1,warmup=1,draws=1))


def test_declared_calendar_origins_include_2019_and_no_silent_drops():
    from research.serra.prior_assessment import study_plan
    directory=Path(__file__).resolve().parents[1]/'research/serra/config/hierarchy'
    config=bx.load_config(directory/'pilot.json'); plan=study_plan(config)
    assert plan['n_months']==1616 and plan['fitted_end']=='2026-08-01'
    assert plan['posterior_fits']==4 and plan['predictive_fits']==16
    assert plan['effective_chain_workers']==4
    for candidate in plan['candidates']:
        if candidate['name'].startswith('pooled'):
            assert candidate['shared_shrinkage']['components']==['level','slope','seasonal']
            assert candidate['innovation_median']['season']==.02
    assert plan['folds'][2]['forecast_start']=='2016-01-01'
    assert plan['folds'][2]['forecast_end']=='2020-12-01'
    dates=pd.date_range('2000-01-01',periods=36,freq='MS')
    a,b=next(bx.calendar_origin_splits(dates,['2001-12'],horizon=12))
    assert a.stop==24 and b.start==24 and b.stop==36
    with pytest.raises(ValueError,match='complete forecast'):
        tuple(bx.calendar_origin_splits(dates,['2002-12'],horizon=12))
    with pytest.raises(ValueError,match='consecutive'):
        tuple(bx.calendar_origin_splits(dates.delete(5),['2001-12'],horizon=12))


def test_joint_compact_reports_select_responses_without_repeating_global_tables(tmp_path):
    import json
    posterior={}; predictive={}
    for variant in ['reference','candidate']:
        directory=tmp_path/variant;directory.mkdir()
        posterior[variant]={c:directory for c in ['a','b']}
        predictive[variant]={c:directory for c in ['a','b']}
        (directory/'config.json').write_text(json.dumps(dict(analysis='copula',credible_interval=.95,risks={'a':1,'b':2})))
        status={'status':'needs_review','issues':[{'reason':'short chain'}]}
        (directory/'convergence.json').write_text(json.dumps(status))
        (directory/'convergence_12.json').write_text(json.dumps(status))
        prior=pd.DataFrame([dict(component='level',scale='SD',distribution=d,lower=.001,median=.01,
                                 upper=.03,credible_interval=.95) for d in ['prior','posterior']])
        for c in ['a','b']:
            prior.to_csv(directory/f'{c}_prior_posterior.csv',index=False)
            for quantity in ['level','slope_C_per_decade','risk']:
                pd.DataFrame(dict(time=['2000-01-01','2000-02-01'],lower=[0,0],median=[1,1],upper=[2,2])).to_csv(
                    directory/f'{c}_{quantity}.csv',index=False)
        prior.assign(scale='population_median').to_csv(directory/'shared_shrinkage.csv',index=False)
        pd.DataFrame(dict(quantity=['a.level.change','b.level.change','a_minus_b.level.change'],
                          lower=[0,0,0],median=[1,2,3],upper=[2,3,4])).to_csv(directory/'period_and_endpoint_targets.csv',index=False)
        pd.DataFrame(dict(parameter=['sd.channel.a.level','sd.channel.b.level','shrinkage.shared.level'],
                          rhat=[1.01,1.02,1.03])).to_csv(directory/'mcmc.csv',index=False)
        cases=pd.DataFrame(dict(channel=['a','b'],origin=[12,12],time=['2001-01-01']*2,horizon=[1,1]))
        cases.assign(score='crps',setting=np.nan,value=[1.,2.]).to_csv(directory/'scores.csv',index=False)
        cases.assign(kind='central_interval',nominal=.95,covered=True).to_csv(directory/'coverage_by_case.csv',index=False)
        cases.assign(pit=[.3,.7]).to_csv(directory/'held_out_pit.csv',index=False)
        cases.assign(observed=[1.,2.],lower=0.,median=1.,upper=3.).to_csv(directory/'predictions.csv',index=False)
        pd.DataFrame(dict(origin=[12])).to_csv(directory/'folds.csv',index=False)
        pd.DataFrame(dict(origin=[12],time=['2001-01-01'],score=[3.])).to_csv(directory/'joint_log_scores.csv',index=False)
    report=bx.SensitivityReport(posterior_runs=posterior,predictive_runs=predictive,baseline='reference')
    tables=report.tables()
    assert len(tables['shared_shrinkage'])==4
    assert len(tables['joint_scientific_targets'])==6
    assert len(tables['scientific_targets'])==4
    assert len(tables['scores'])==4
    assert len(tables['joint_log_scores'])==2
    assert set(tables['mcmc'].quantity)=={'sd.channel.a.level','sd.channel.b.level'}
    assert len(tables['joint_log_scores_comparison'])==2  # all + horizon 1..12
    assert tables['convergence'].numerical_status.eq('needs_review').all()
    assert not tables['scientific_targets'].quantity.str.contains('minus').any()
