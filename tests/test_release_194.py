"""Distributional checks and end-to-end contracts for the matched hierarchy."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm
import bucex as bx
from research.seasonal import jobs
from research.seasonal.job_plan import plan
from research.monthly.models import joint_model,independent_model,fit_options
from research.monthly.report import write_report,convergence_parameters
from bucex.inference.fit.shrinkage import SharedShrinkageState


def config(variant,channel=None):
    task=next(t for t in jobs.tasks('posterior',tier='screen')+jobs.tasks('deferred',tier='screen') if t.variant==variant and
              (channel is None or channel==t.channel))
    return jobs.task_config(task,'screen')


def test_direct_sd_update_has_the_correct_normalizing_constant(monkeypatch):
    spec=bx.SharedShrinkage.from_sd({'level':.01},log_sd=np.log(3))
    prior=bx.fs_priors('gaussian',period=4,initial_slope=bx.NormalPrior(0,.01))
    assert spec.coefficient_sd('level',.01)==.01
    assert spec.conditional_prior(prior,spec.anchors).s_level.sd==.01
    assert spec.conditional_prior(prior,spec.anchors).beta0.sd==.01
    coefficients=np.array([.003,-.008,.014,-.001,.018,.006])
    states=[SimpleNamespace(params_state={'s_level':v}) for v in coefficients]
    state=SharedShrinkageState(spec,{'level':0.},{'level':states})
    checked=[]
    def check(current,target,rng,width):
        reference=lambda u:norm.logpdf(coefficients,scale=.01*np.exp(u)).sum()+norm.logpdf(u,scale=np.log(3))
        for u in (-2.,-.25,1.,2.5):
            assert target(u)-target(0)==pytest.approx(reference(u)-reference(0),abs=1e-10)
        checked.append(True)
        return .2,4
    monkeypatch.setattr('bucex.inference.fit.shrinkage._slice_sample_real',check)
    state.update(np.random.default_rng(1))
    assert checked and state.medians['level']==pytest.approx(.01*np.exp(.2))


def test_direct_sd_prior_matches_independent_integrated_cdf():
    anchor=.01;width=np.log(3)
    prior=bx.fs_priors('gaussian',period=4,initial_slope=bx.NormalPrior(0,.01))
    shared=bx.SharedShrinkage.from_sd({'level':anchor},log_sd=width)
    private=bx.IndependentShrinkage.from_sd({'level':anchor},log_sd=width)
    a=bx.draw_marginal_prior(bx.MarginalPriors({'a':prior,'b':prior},shrinkage=shared),80000,seed=194)
    b=bx.draw_marginal_prior(bx.MarginalPriors({'a':prior},shrinkage=private),80000,seed=194)
    np.testing.assert_array_equal(a['channels']['a']['sd.level'],b['channels']['a']['sd.level'])
    assert a['independent']=={} and b['shared']=={}
    samples=a['channels']['a']['sd.level']
    for threshold in (.005,.01,.025):
        exact=quad(lambda u:(2*norm.cdf(threshold/(anchor*np.exp(u)))-1)*norm.pdf(u,scale=width),-9*width,9*width)[0]
        assert np.mean(samples<threshold)==pytest.approx(exact,abs=.005)
    assert np.std(a['channels']['a']['initial.slope'])==pytest.approx(.01,rel=.02)
    assert np.mean((samples/a['shared']['level'])**2)==pytest.approx(1,abs=.025)
    assert np.corrcoef(np.log(samples),np.log(a['channels']['b']['sd.level']))[0,1]>.35


def test_scopes_widths_and_omissions_are_explicit():
    assert len(plan('screen','experiments'))==41
    assert len(jobs.tasks('experiments',tier='screen'))==41
    for v in ('reference','independent_reference','fixed_reference'):
        c=config(v)
        assert c['priors']['innovation_sd']==dict(level=.01,trend=.0001,season=.01)
        assert c['priors']['initial_slope_sd']==.01
        assert c['contrasts'] is None and c['copula'] is None
        assert c['mcmc']['draws']==2000 and c['mcmc']['warmup']==1000
    for ch in bx.UCCLE_SERIES:
        c=config('leave_out_'+ch)
        assert len(c['data']['series'])==5 and ch not in c['data']['series']
    shared=config('reference');private=config('independent_reference')
    assert shared['priors']['shared_shrinkage']==private['priors']['independent_shrinkage']
    assert shared['priors']['innovation_sd']==dict(level=.01,trend=.0001,season=.01)
    c=config('fixed_location_seasonality');d=bx.load_uccle_multiseries(**c['data'])
    _,p=joint_model(d,c)
    assert set(p.shrinkage.anchors)=={'level','slope'}


@pytest.fixture(scope='module')
def reports(tmp_path_factory):
    root=tmp_path_factory.mktemp('matched_reports');runs={};fits={}
    for v in ('reference','independent_reference','fixed_reference','constant_dispersion','fixed_location_seasonality'):
        c=config(v,channel=None if v in ('reference','constant_dispersion','fixed_location_seasonality') else 'TXm')
        c['data'].update(start='2021-03')
        c['mcmc'].update(warmup=2,draws=8,chain_workers=1,progress=False)
        c.update(forecast_horizon=8,forecast_draws=16,predictive_check_draws=12,prior_draws=80,figures=False)
        d=bx.load_uccle_multiseries(**c['data'])
        model,priors=(joint_model if c['analysis']=='joint' else independent_model)(d,c)
        fit=bx.fit(d,model,priors=priors,**fit_options(c,family=model.family))
        directory=root/v
        write_report(fit,directory,config=c,risks=c['risks'],horizon=8)
        restored=bx.load_fit(directory/'fit.bucex')
        assert restored.model.copula is None
        if restored.priors.shrinkage is not None:
            assert restored.priors.shrinkage.scale_parameterization=='normal_sd'
            assert 'initial_slope' not in restored.priors.shrinkage.anchors
            stem='shared' if c['analysis']=='joint' else 'independent'
            table=pd.read_csv(directory/(stem+'_shrinkage.csv'))
            assert set(table.scale)=={'normal_SD'}
            assert not (directory/'fixed_prior_settings.csv').exists()
        assert (directory/'scientific_target_traces.csv.gz').exists()
        assert not (directory/'period_contrasts.csv').exists()
        runs[v]={'TXm':directory};fits[v]=fit
    return runs,fits


def test_mixed_scope_reports_collect_with_95_percent_intervals(reports,tmp_path):
    runs,_=reports
    report=bx.SensitivityReport(posterior_runs=runs,baseline='independent_reference',block_frequency='seasonal')
    tables=report.tables()
    assert set(tables['prior_posterior'].variant)==set(runs)
    assert tables['prior_posterior'].credible_interval.eq(.95).all()
    assert {'shared_shrinkage','independent_shrinkage','fixed_prior_settings'}<=set(tables)
    report.save(tmp_path,figures=True)
    assert list(tmp_path.glob('*.png'))


def test_only_declared_constant_scale_coordinates_leave_the_gate(reports):
    _,fits=reports
    for variant in ('reference','constant_dispersion'):
        fit=fits[variant];raw=fit.diagnostics()['parameters']
        checked=convergence_parameters(raw,fit.n_chains,fit)
        keys=[str(k) for k in checked.index if str(k).startswith('scale.seasonal.')]
        assert bool(keys)==(variant=='reference')
        assert any(str(k).startswith('shrinkage.shared.') for k in checked.index)
        assert any(str(k).startswith('initial.channel.') for k in checked.index)
