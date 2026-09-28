"""Regression checks for identity observations and separately regularized initial rates."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import bucex as bx
from research.monthly.models import joint_model, fit_options
from research.monthly.report import write_report
from research.seasonal import jobs


def test_frozen_191_has_identity_dependence_and_separate_initial_rates():
    config=bx.load_config(jobs.CONFIG/'reference_191.json')
    data=bx.load_uccle_multiseries(**config['data'])
    model,priors=joint_model(data,config)
    np.testing.assert_array_equal(model.copula.correlation_matrix({},tuple(data)),np.eye(6))
    assert 'initial_slope' not in priors.shrinkage.anchors
    assert all(p.beta0.sd*40==.5 for p in priors.channels.values())


def test_width_controls_do_not_move_fixed_initial_rate_prior():
    configs={t.variant:jobs.task_config(t,'screen') for t in jobs.tasks()}
    for name in ('narrow_log2_matched','wide_log4_matched','narrow_log2_fixed_anchors','wide_log4_fixed_anchors'):
        assert configs[name]['priors']['initial_slope_sd']==.0125
    for name in ('narrow_log2_fixed_anchors','wide_log4_fixed_anchors'):
        assert configs[name]['priors']['innovation_median']==configs['reference']['priors']['innovation_median']


def test_original_fraction_splits_are_complete_ten_year_forecasts():
    from research.monthly.validate import validation_splits
    for fraction,task in zip((.6,.8,.9),[t for t in jobs.tasks('validation10') if t.channel=='TXm']):
        config=jobs.task_config(task,'screen')
        data=bx.load_uccle_multiseries(**config['data'])
        train,test=validation_splits(data,config['validation'])[0]
        assert train.stop==int(np.floor(fraction*len(data)))
        assert len(test)==40 and test.stop<=len(data)
        assert task.design=='fraction_10y'


def test_fixed_initial_rate_prior_effect_is_gaussian(tmp_path):
    from research.seasonal.prior_effects import run
    config=bx.load_config(jobs.CONFIG/'main.json')
    run(config,tmp_path,draws=80000,seed=191)
    table=pd.read_csv(tmp_path/'thirty_year_effects.csv')
    slopes=table[table.component.eq('initial_slope')]
    np.testing.assert_allclose(slopes.sd,1.5,rtol=.02)
    np.testing.assert_allclose(slopes.upper_95,1.96*1.5,rtol=.02)


def test_unpooled_initial_rates_survive_fit_archive_and_reporting(tmp_path):
    config=bx.load_config(Path(__file__).parent/'fixtures/seasonal_smoke_191.json')
    config['data'].update(series=['TXm','TNm'],start='2016-03')
    config['mcmc'].update(chains=2,chain_workers=1,warmup=3,draws=10,progress=False)
    config.update(figures=False,forecast_draws=24,predictive_check_draws=12,prior_draws=40,
                  contrasts=None,save_fits=True)
    data=bx.load_uccle_multiseries(**config['data'])
    model,priors=joint_model(data,config)
    fit=bx.fit(data,model,priors=priors,**fit_options(config,family=model.family))
    write_report(fit,tmp_path,config=config,horizon=120,level=.95,risks=config['risks'])
    restored=bx.load_fit(tmp_path/'fit.bucex')
    assert not any('shrinkage.shared.initial_slope' in k for k in restored.parameter_draws)
    assert len(pd.read_csv(tmp_path/'initial_slope_prior_posterior.csv'))==4
    assert set(pd.read_csv(tmp_path/'shared_shrinkage.csv').component)=={'level','slope','seasonal'}
    assert not (tmp_path/'compound_heat_conditional_risk.csv').exists()
    # Phase 1 is DJF even though the record starts with MAM. Previously identity
    # diagnostics used the record-start phase and mislabeled all four seasons.
    dep=pd.read_csv(tmp_path/'residual_dependence_by_season.csv')
    dates=pd.DatetimeIndex(fit.time)
    for phase,months in enumerate(((12,1,2),(3,4,5),(6,7,8),(9,10,11)),1):
        assert dep[dep.phase.eq(phase)].n_time.eq(dates.month.isin(months).sum()).all()
    from research.seasonal.manuscript_figures import _shared_shrinkage,Inputs
    import matplotlib.pyplot as plt
    figure=_shared_shrinkage(Inputs(),tmp_path)
    assert len(figure.axes)==3
    plt.close(figure)
