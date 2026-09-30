"""Fixed-prior regressions: declared priors, matched forecasts and collection."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import bucex as bx
from research.monthly.models import independent_model
from research.monthly.validate import validate
from research.seasonal.jobs import tasks, task_config
from research.seasonal.sweetspot_plan import cells, fixed_cells, study_variants
from research.seasonal.sweetspot_diagnostics import local_derivative
from research.seasonal.prior_simulations import simulate_paths, variant_config, run_suite
from research.seasonal.sweetspot_report import aggregate_posterior, validation_tables, CHANNELS
from tests.test_sweetspot_1961 import entry


def fixed_config(channel='TXm'):
    task=next(t for t in tasks('sweetspot_posterior',tier='screen')
              if t.variant=='fixed_reference' and t.channel==channel)
    return task_config(task,'screen')


def test_fixed_priors_have_no_hyperparameters_at_every_grid_cell():
    assert len(cells())==len(fixed_cells())==14
    assert len(study_variants())==42
    selected=[t for t in tasks('sweetspot_posterior',tier='screen') if t.scope=='fixed']
    assert len(selected)==84
    for task in selected:
        config=task_config(task,'screen');p=config['priors']
        assert 'shared_shrinkage' not in p and 'independent_shrinkage' not in p
        assert config['analysis']=='independent' and task.series==(task.channel,)
        _,priors=independent_model(pd.DataFrame(columns=task.series),config)
        assert priors.shrinkage is None
        prior=priors.channels[task.channel]
        for field,key in [('s_level','level'),('s_trend','trend'),('s_season','season')]:
            normal=getattr(prior,field)
            assert isinstance(normal,bx.NormalPrior) and normal.mean==0
            assert normal.sd==p['innovation_sd'][key]


def test_all_fixed_validation_cases_match_their_pooled_calibration():
    selected=tasks('sweetspot',tier='screen')
    pooled={(t.variant,t.origin):t for t in selected if t.kind=='forecast' and t.scope=='shared'}
    fixed=[t for t in selected if t.kind=='forecast' and t.scope=='fixed']
    assert len(fixed)==14*5*6
    for task in fixed:
        config=task_config(task,'screen');setting=config['variant']['setting']
        reference=task_config(pooled[setting,task.origin],'screen')
        assert config['validation']==reference['validation']
        assert config['priors']['innovation_sd']==reference['priors']['innovation_sd']
        assert config['mcmc']['draws']==reference['mcmc']['draws']


def test_fixed_prior_simulation_is_independent_and_gaussian():
    config=variant_config('fixed_reference')
    result=simulate_paths(config,draws=20000,seed=1981,
        dates=pd.date_range('2000-03-01',periods=4,freq='3MS'))
    sampled=result['sampled']
    assert not sampled.get('shared')
    a,b=[sampled['channels'][ch]['sd.slope'] for ch in ('TXm','TNm')]
    assert np.mean(a*a)==pytest.approx(.002**2,rel=.03)
    # Squaring an ordinary Gaussian has fourth/second moment ratio three,
    # unlike the HN scale mixture, whose ratio is nine.
    assert np.mean(a**4)/np.mean(a*a)**2==pytest.approx(3.,rel=.06)
    assert abs(np.corrcoef(a,b)[0,1])<.03


def test_fixed_prior_local_derivative_matches_density_reweighting():
    rng=np.random.default_rng(1981);s=rng.normal(0,.3,size=(2,1500))
    values=s*s+.1*rng.normal(size=s.shape);anchor=.4
    def mean(delta):
        weights=np.exp(.5*(s.ravel()/anchor)**2*(1-np.exp(-2*delta)))
        return np.average(values.ravel(),weights=weights)
    eps=1e-5
    derivative=(mean(eps)-mean(-eps))/(2*eps)
    np.testing.assert_allclose(local_derivative(values,abs(s),anchor)*(s.size-1)/s.size,
                               derivative,rtol=1e-7)


def test_fixed_prior_simulation_outputs(tmp_path):
    out=run_suite(tmp_path,suite='fixed_reference',draws=24,figures=False)
    calibration=pd.read_csv(out/'fixed_reference/analytic/analytic_calibration.csv')
    assert not calibration.sampled_hyperparameter.any()
    assert (out/'fixed_reference/target_summary.csv').is_file()


@pytest.mark.parametrize('channel',['TXm','TXx','TXn'])
def test_fixed_forecast_scores_and_diagnostics(tmp_path,channel):
    config=fixed_config(channel)
    config['data']['start']='2021-03'
    config['mcmc'].update(chains=2,chain_workers=1,warmup=2,draws=8,progress=False)
    config['validation'].update(training_ends=['2024-11'],horizon=2,draws=24)
    validate(config,directory=tmp_path)
    report=tmp_path/channel
    scores=pd.read_csv(report/'scores.csv')
    assert np.isfinite(scores.loc[scores.score=='crps','value']).all()
    coverage=pd.read_csv(report/'coverage_by_case.csv')
    assert set(coverage.loc[coverage.kind=='central_interval','nominal'])=={.9,.95,.99}
    draws=pd.read_csv(next((report/'sweetspot').glob('*/sweetspot_draws.csv.gz')))
    assert not any(c.startswith('tau_') for c in draws)
    assert channel+'_q_slope' in draws
    saved=bx.load_fit(next(report.glob('fit_*.bucex')))
    assert saved.priors.shrinkage is None


def test_matched_crps_uses_same_calibration_and_cases(tmp_path):
    rows=[]
    for name,values,dates in [
        ('reference',[20.,20.],['2021-03-01','2021-06-01']),
        ('ss_a5e2_b1e3',[1.,3.],['2021-03-01','2021-06-01']),
        ('fixed_ss_a5e2_b1e3',[2.,9.],['2021-03-01','2021-09-01'])]:
        path=tmp_path/name;path.mkdir()
        pd.DataFrame(dict(channel='TXm',time=dates,score='crps',value=values,
            horizon=[1,3] if name.startswith('fixed') else [1,2])).to_csv(path/'scores.csv',index=False)
        rows.append(dict(task=SimpleNamespace(variant=name,origin='2020-11'),report=path,passed=True))
    result=validation_tables(rows)['matched_prior_crps'].iloc[0]
    assert result.setting=='ss_a5e2_b1e3' and result.matched_cases==1
    assert result.difference==1. and result.pooled_crps==1.


def test_collected_report_retains_fixed_scope_and_partial_status(tmp_path,monkeypatch):
    from research.seasonal import sweetspot_report as report
    parts={}
    for scope,name in [('shared','reference'),('independent','independent_reference'),('fixed','fixed_reference')]:
        owners=['joint'] if scope=='shared' else CHANNELS
        for owner in owners:
            e=entry();e.update(scope=scope,directory=tmp_path/owner,local=pd.DataFrame())
            if owner!='joint':
                for key in ('paths','forecasts','allocation','risk'):
                    e[key]=e[key][e[key].channel==owner].copy()
            parts.setdefault(name,[]).append((owner,e))
    posterior=aggregate_posterior(parts)
    assert posterior['fixed_reference']['complete']
    records=pd.DataFrame([dict(status='completed',numerically_passed=True)]*13)
    monkeypatch.setattr(report,'load_evidence',lambda *a,**kw:(records,posterior,[]))
    monkeypatch.setattr(report,'figures',lambda *a,**kw:[])
    monkeypatch.setattr(report,'comparison_figures',lambda *a,**kw:[])
    out=report.build(tmp_path,output=tmp_path/'review')
    differences=pd.read_csv(out/'matched_fixed_prior_comparisons.csv')
    assert len(differences)==12
    assert (differences.left_scope.eq('fixed')|differences.right_scope.eq('fixed')).all()
    html=(out/'sweetspot_review.html').read_text()
    assert '3/42' in html and 'no hyperparameters to estimate' in html
    regions=pd.read_csv(out/'stability_regions.csv')
    assert set(regions.scope)=={'shared','independent','fixed'}
    assert not regions.history_candidate.any()
