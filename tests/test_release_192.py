"""Private normal-lognormal priors, real single-response fits and held-out reports."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
import bucex as bx
from bucex.inference.fit.shrinkage import shared_scale_log_target
from bucex.priors.shrinkage import NORMAL_ABSOLUTE_MEDIAN as Z75
from research.monthly.models import independent_model,fit_options,joint_model
from research.monthly.report import write_report
from research.seasonal import jobs


def small_config(name,variant='reference'):
    task=next(t for t in jobs.tasks('posterior') if t.channel==name and t.variant==variant)
    c=jobs.task_config(task,'screen')
    c['data']['start']='2016-03'
    c['mcmc'].update(chains=2,chain_workers=1,warmup=3,draws=10,progress=False)
    c.update(figures=False,forecast_draws=24,predictive_check_draws=12,prior_draws=80,contrasts=None)
    return c


def test_exact_one_coefficient_hyperconditional_and_unchanged_marginal_prior():
    anchor=.004836005867750226;tau=np.log(3);coef=.008
    for u in [-2.,0.,1.5]:
        target=shared_scale_log_target(u,[coef],anchor=anchor,log_sd=tau)
        ref=norm.logpdf(coef,0,anchor*np.exp(u)/Z75)+norm.logpdf(u,0,tau)
        at0=shared_scale_log_target(0.,[coef],anchor=anchor,log_sd=tau)
        ref0=norm.logpdf(coef,0,anchor/Z75)+norm.logpdf(0.,0,tau)
        assert target-at0==pytest.approx(ref-ref0,abs=1e-10)
    base=bx.fs_priors('gaussian',period=4)
    priors=bx.MarginalPriors({'a':base},shrinkage=bx.IndependentShrinkage({'level':anchor},log_sd=tau))
    independent=bx.draw_marginal_prior(priors,50000,seed=144)
    historical=bx.MarginalPriors({'a':base,'b':base},shrinkage=bx.SharedShrinkage({'level':anchor},log_sd=tau,initial_slope_sd=None))
    shared=bx.draw_marginal_prior(historical,50000,seed=144)
    # Identical marginal law, including the mixing distribution, not plug-in anchors.
    np.testing.assert_array_equal(independent['channels']['a']['sd.level'],shared['channels']['a']['sd.level'])
    assert independent['shared']=={} and set(independent['independent'])=={'level'}
    with pytest.raises(ValueError,match='one response'):
        bx.MarginalPriors({'a':base,'b':base},shrinkage=priors.shrinkage)
    with pytest.raises(ValueError,match='initial slope'):
        bx.IndependentShrinkage({'level':anchor},initial_slope_sd=.0125)


def test_every_task_is_one_response_without_copula_or_pooling():
    for task in jobs.tasks('all'):
        c=jobs.task_config(task,'screen')
        data=pd.DataFrame({task.channel:np.zeros(8)})
        model,prior=independent_model(data,c)
        assert len(model.channels)==1 and model.copula is None
        assert list(prior.channels)==[task.channel]
        assert c['analysis']=='independent' and c['data']['series']==[task.channel]
        assert isinstance(prior.shrinkage,bx.IndependentShrinkage)
        assert 'initial_slope' not in prior.shrinkage.anchors
        expected={'half_initial_slope':.25,'double_initial_slope':1.}.get(task.variant,.5)
        assert prior.channels[task.channel].beta0.sd*40==expected
        assert not c['report_joint_risks'] and not c['cross_summary_contrasts']
        with pytest.raises(ValueError,match='private hierarchy'):joint_model(data,c)


@pytest.mark.parametrize('name,variant',[('TXm','reference'),('TXx','reference'),('TNn','reference'),
    ('TNm','fixed_location_seasonality'),('TXn','constant_dispersion')])
def test_single_response_fit_archive_forecast_and_report(tmp_path,name,variant):
    c=small_config(name,variant)
    data=bx.load_uccle_multiseries(**c['data']);model,prior=independent_model(data,c)
    fit=bx.fit(data,model,priors=prior,**fit_options(c,family=model.family))
    assert not fit.metadata['joint_model'] and not fit.metadata['copula_feedback']
    assert fit.metadata['shared_shrinkage'] is None
    assert fit.metadata['independent_shrinkage'] is not None
    assert not any(k.startswith(('copula.','shrinkage.shared.')) for k in fit.parameter_draws)
    write_report(fit,tmp_path,config=c,horizon=120,level=.95,risks=c['risks'])
    restored=bx.load_fit(tmp_path/'fit.bucex')
    assert isinstance(restored.priors.shrinkage,bx.IndependentShrinkage)
    np.testing.assert_array_equal(restored.parameter_draws['shrinkage.independent.slope'],fit.parameter_draws['shrinkage.independent.slope'])
    assert not (tmp_path/'shared_shrinkage.csv').exists()
    assert not (tmp_path/'residual_dependence.csv').exists()
    assert not (tmp_path/'copula_correlations.csv').exists()
    assert len(pd.read_csv(tmp_path/'initial_slope_prior_posterior.csv'))==2
    assert len(pd.read_csv(tmp_path/f'{name}_forecast_uncertainty.csv'))==360
    assert set(pd.read_csv(tmp_path/'independent_shrinkage.csv').component)==set(prior.shrinkage.anchors)
    tables=bx.SensitivityReport(posterior_runs={'reference':{name:tmp_path}},baseline='reference',block_frequency='seasonal').tables()
    assert {'paths','independent_shrinkage','independent_shrinkage_updates','initial_slope_prior_posterior'}<=set(tables)
    assert not any(k.startswith('joint_') for k in tables)
    # Warm starts must recover the private scale keys from a saved archive.
    resumed=bx.fit(data,model,priors=restored.priors,init=restored,
        mcmc=bx.MCMC(chains=1,chain_workers=1,warmup=1,draws=3,seed=42))
    forecast=resumed.forecast(4,draws=3,seed=42)
    assert np.isfinite(forecast.pit(data[name].to_numpy()[-4:],channel=name)).all()


def test_independent_chains_identical_serial_and_parallel():
    c=small_config('TXm');c['data']['start']='2021-03'
    data=bx.load_uccle_multiseries(**c['data']);model,prior=independent_model(data,c)
    options=fit_options(c,family=model.family)
    serial=bx.fit(data,model,priors=prior,**options)
    options['mcmc']=replace(options['mcmc'],chain_workers=2)
    parallel=bx.fit(data,model,priors=prior,**options)
    np.testing.assert_array_equal(serial.state_draws,parallel.state_draws)
    for key in serial.parameter_draws:np.testing.assert_array_equal(serial.parameter_draws[key],parallel.parameter_draws[key])


def test_single_response_validation_uses_original_scale_and_all_coverages(tmp_path):
    from research.monthly.validate import validate
    c=small_config('TNn');c['validation'].update(training_ends=['2020-11'],horizon=20,draws=24)
    validate(c,directory=tmp_path)
    target=tmp_path/'TNn'
    coverage=pd.read_csv(target/'coverage_by_case.csv')
    assert set(coverage[coverage.kind.eq('central_interval')].nominal)=={.9,.95,.99}
    scores=pd.read_csv(target/'scores.csv');cases=pd.read_csv(target/'threshold_cases.csv')
    assert set(scores.channel)=={'TNn'} and cases.direction.eq('<').all()
    merged=scores[scores.score.eq('exceedance_brier')].merge(cases,on=['channel','origin','time','horizon'])
    np.testing.assert_allclose(merged.value,merged.brier)
    assert not (target/'joint_log_scores.csv').exists()
    assert list(target.glob('independent_shrinkage_*.csv'))
    assert pd.read_csv(target/'folds.csv').horizon.tolist()==[20]


def test_task_execution_collection_figures_and_compact_export(tmp_path,monkeypatch):
    from research.seasonal import collect_jobs
    from research.seasonal.export_results import export
    import zipfile
    import matplotlib.pyplot as plt
    real_config=jobs.task_config
    selected=[t for t in jobs.tasks('all') if t.channel=='TXm' and t.variant in ('reference','double_slope')
              and (t.kind=='posterior' or (t.kind=='forecast' and t.origin=='2020-11'))]
    assert len(selected)==4
    def config(task,tier):
        c=real_config(task,tier)
        c['data']['start']='2016-03'
        c['mcmc'].update(chains=2,chain_workers=1,warmup=2,draws=8,progress=False)
        c.update(figures=False,forecast_draws=24,predictive_check_draws=12,prior_draws=40,contrasts=None)
        c['validation']['draws']=24
        return c
    monkeypatch.setattr(jobs,'task_config',config)
    monkeypatch.setattr(collect_jobs,'task_config',config)
    monkeypatch.setattr(collect_jobs,'tasks',lambda *a,**k:selected)
    for task in selected:
        result=jobs.execute(task,tier='screen',root=tmp_path)
        assert result.name=='TXm'
        assert bx.load_config(result.parents[1]/'task.json')['status']=='completed'
        assert jobs.execute(task,tier='screen',root=tmp_path)==result
    out=collect_jobs.collect(tmp_path,'screen',batch='all',figures=True)
    status=bx.load_config(out/'status.json')
    assert status['completed']==4 and not status['missing_or_failed']
    pair=bx.load_config(out/'paired_validation.json')['comparisons']
    assert len(pair)==1 and pair[0]['channel']=='TXm' and pair[0]['origins']==['2020-11']
    images=list(out.rglob('*.png'));assert images
    assert plt.get_fignums()==[]
    path,count=export(tmp_path,'screen',tmp_path/'compact.zip')
    with zipfile.ZipFile(path) as archive:
        assert len(archive.namelist())==count
        assert all(not n.endswith(('.bucex','.png')) for n in archive.namelist())
        assert any(n.endswith('independent_shrinkage.csv') for n in archive.namelist())
    with pytest.raises(FileExistsError):export(tmp_path,'screen',path)
