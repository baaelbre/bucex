"""Monthly calibration, isolated dispatch and calendar aggregation regressions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
import bucex as bx
from research.monthly.study_plan import calibration_rows, specification, variants, convert
from research.seasonal.jobs import tasks, task_config, all_tasks
from research.seasonal.job_plan import plan
from research.seasonal.bundles import members
from research.monthly.models import independent_model, fit_options


def config(channel='TXm',setting='reference',tier='screen'):
    t=next(t for t in tasks('monthly_all',tier=tier) if t.channel==channel and t.variant=='monthly_fixed_'+setting)
    return task_config(t,tier)


def test_monthly_calibration_matches_physical_thirty_year_effects():
    f=pd.DataFrame(calibration_rows());f=f[f.years==30]
    pivot=f.pivot(index=['setting','component'],columns='frequency',values='displacement_sd_C')
    np.testing.assert_allclose(pivot.monthly,pivot.seasonal,rtol=1e-14)
    c=config();sd=c['priors']['innovation_sd']
    assert sd['level']==pytest.approx(.1/np.sqrt(3))
    assert sd['trend']==pytest.approx(.0003832923311016883)
    assert sd['season']==.1
    assert c['priors']['initial_slope_sd']*120==pytest.approx(.4)
    # Matching at 30y does not purport to match integrated slope at every H.
    f=pd.DataFrame(calibration_rows());p=f[(f.component=='slope')&(f.years==10)&(f.setting=='reference')]
    assert p.displacement_sd_C.max()/p.displacement_sd_C.min()<1.015


def test_complete_plan_and_nonoverlapping_host_split():
    expected={'monthly_reference':6,'monthly_structural':90,'monthly_observation':38,'monthly_sensitivity':122,'monthly_all':128}
    assert {b:len(tasks(b)) for b in expected}==expected
    a={t.id for t in tasks('monthly_structural')};b={t.id for t in tasks('monthly_observation')}
    assert not a & b
    assert a|b=={t.id for t in tasks('monthly_all')}
    assert {t.id for t in tasks('monthly_reference')}<=a
    assert len(variants())==22
    for g in plan('screen','monthly_all'):
        assert len(members(g,'screen'))==1
        assert g['cpus']==g['required_workers']==2
    assert {t.id for t in tasks('monthly_all')}<={t.id for t in all_tasks('screen')}


def test_all_monthly_variants_are_separate_fixed_normals():
    for t in tasks('monthly_all'):
        c=task_config(t,'screen')
        assert c['analysis']=='independent' and c['copula'] is None
        assert not c['priors'].get('shared_shrinkage') and not c['priors'].get('independent_shrinkage')
        assert c['credible_interval']==.95 and c['forecast_horizon']==360
        assert c['forecast_draws']==50000 and c['model']['steps_per_year']==12
        assert c['model']['period']==12 and len(c['data']['series'])==1
        model,priors=independent_model(pd.DataFrame(columns=[t.channel]),c)
        assert priors.shrinkage is None
        prior=priors.channels[t.channel]
        assert prior.s_level.sd==pytest.approx(c['priors']['innovation_sd']['level'],rel=1e-14)
        assert prior.s_trend.sd==pytest.approx(c['priors']['innovation_sd']['trend'],rel=1e-14)
        assert prior.s_season.sd==pytest.approx(c['priors']['innovation_sd']['season'],rel=1e-14)
        scale=model.channel(t.channel).observation.scale
        if t.variant.endswith('constant_dispersion'):assert scale is None or scale.period==1
        else:assert scale.period==12
        bx.compile_model(model,pd.DataFrame({t.channel:np.linspace(10.,12.,24)},index=pd.date_range('2024-01',periods=24,freq='MS')))


def test_monthly_screen_and_paper_budgets_are_explicit():
    for tier,chains,warmup,draws in [('screen',2,1000,1000),('paper',4,4000,12000)]:
        c=config(tier=tier)
        assert (c['mcmc']['chains'],c['mcmc']['warmup'],c['mcmc']['draws'])==(chains,warmup,draws)
        assert c['model']['seasonal_scale']
    # Shape sensitivities do not refit unchanged Gaussian responses.
    assert len([t for t in tasks('monthly_all') if 'xi_narrow' in t.variant])==4


def test_annual_aggregation_subsamples_complete_paths(tmp_path):
    from bucex.plotting.report import save_prediction_report
    c=config('TXx');data=bx.load_uccle_multiseries(**dict(c['data'],start='2023-03'))
    model,prior=independent_model(data,c)
    c['mcmc'].update(warmup=2,draws=8,chains=2,chain_workers=1,progress=False)
    fit=bx.fit(data,model,priors=prior,**fit_options(c,family='gev'))
    forecast=fit.forecast(36,draws=40,seed=1982)
    notes=save_prediction_report(fit,tmp_path,channel='TXx',forecast=forecast,
        predictive=fit.posterior_predictive(draws=12,seed=12),threshold=35,draws=12,seed=1982,
        figures=False,trace_exports=False,aggregate_draws=12)
    assert notes['marginal_forecast_draws']==40
    assert notes['aggregate_forecast_draws']==12
    assert len(pd.read_csv(tmp_path/'TXx_forecast_year.csv'))==2
    assert len(pd.read_csv(tmp_path/'TXx_forecast_season.csv'))==12
    # The original forecast is untouched and still has all paths for fan charts.
    assert forecast.n_draws==40


def test_compact_host_exports_merge_without_overwriting_evidence(tmp_path, monkeypatch):
    from research.seasonal.export_results import export
    import research.monthly.combine as merger
    archives=[]
    for host, task in [('gallade','reference'),('biobot','xi_wide')]:
        root=tmp_path/host
        path=root/'screen'/task/'report'
        path.mkdir(parents=True)
        (path/'summary.csv').write_text('median\n1\n')
        (path/'fit.bucex').write_bytes(b'posterior archive stays on host')
        derived=root/'screen'/'collected'/host
        derived.mkdir(parents=True)
        (derived/'index.html').write_text('host report')
        archives.append(export(root,'screen',tmp_path/(host+'.zip'),figures=True)[0])
    calls=[]
    monkeypatch.setattr(merger,'build',lambda root,**kwargs:calls.append((root,kwargs)))
    out=tmp_path/'combined'
    merger.combine(archives,out)
    assert len(list(out.rglob('summary.csv')))==2
    assert not list(out.rglob('fit.bucex')) and not (out/'screen'/'collected').exists()
    assert calls==[(out,{'tier':'screen','batch':'monthly_all'})]
    merger.combine(archives,out)  # Identical reruns are safe.
    (out/'screen'/'reference'/'report'/'summary.csv').write_text('changed\n')
    with pytest.raises(ValueError,match='Conflicting evidence'):
        merger.combine(archives,out)
