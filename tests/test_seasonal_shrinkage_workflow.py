"""Three-component pooling and matched SERRA comparison contracts."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bucex as bx
from bucex.priors.shrinkage import NORMAL_ABSOLUTE_MEDIAN
from research.serra.experiment import configured_variant
from research.serra.models import joint_model, fit_options
from research.serra.prior_assessment import study_plan

CONFIG=Path(__file__).resolve().parents[1]/'research/serra/config/hierarchy'


def test_seasonal_hyperprior_maps_to_season_coefficient_only():
    prior=bx.fs_priors('gaussian',period=12)
    spec=bx.SharedShrinkage({'season':.02}, initial_slope_sd=None)
    conditional=spec.conditional_prior(prior,{'seasonal':.007})
    assert conditional.s_season.sd==pytest.approx(.007/NORMAL_ABSOLUTE_MEDIAN)
    assert conditional.s_level==prior.s_level and conditional.s_trend==prior.s_trend
    assert conditional.gamma0_season==prior.gamma0_season
    marginal=bx.MarginalPriors({'a':prior,'b':prior},shrinkage=spec)
    samples=bx.draw_marginal_prior(marginal,20000,seed=121)
    m=samples['shared']['seasonal']
    z=samples['channels']['a']['sd.seasonal']*NORMAL_ABSOLUTE_MEDIAN/m
    assert np.mean(z*z)==pytest.approx(1.,abs=.04)
    assert np.corrcoef(samples['channels']['a']['sd.seasonal'],samples['channels']['b']['sd.seasonal'])[0,1]>.2


@pytest.mark.parametrize('name,n_fits', [('pilot',4),('seasonality',4),('dependence',2),('adequacy',3),('reviewer_sensitivity',15),('confirm',2)])
def test_documented_experiments_resolve_through_2026(name,n_fits):
    plan=study_plan(bx.load_config(CONFIG/(name+'.json')))
    assert plan['fitted_end']=='2026-08-01' and plan['n_months']==1614
    assert plan['posterior_fits']==n_fits and plan['predictive_fits']==4*n_fits
    assert plan['effective_chain_workers']==4
    assert all('model' in candidate and 'priors' in candidate and 'copula' in candidate for candidate in plan['candidates'])


def test_seasonal_candidates_isolate_pooling_and_anchor_changes():
    config=bx.load_config(CONFIG/'seasonality.json')
    cases={v['name']:configured_variant(config,v) for v in config['variants']}
    base=cases['pooled_quarter']
    for name,case in cases.items():
        assert case['model']==base['model'] and case['copula']==base['copula']
        assert case['priors']['innovation_median']['level']==.0025
        assert case['priors']['innovation_median']['trend']==.0000125
        assert case['priors']['seasonal_initial_sd']==base['priors']['seasonal_initial_sd']
    assert cases['pooled_level_slope']['priors']['shared_shrinkage']['components']==['level','slope']
    assert cases['season_anchor_half']['priors']['innovation_median']['season']==.01
    assert cases['season_anchor_double']['priors']['innovation_median']['season']==.04
    assert base['priors']['innovation_median']['season']==.02  # no in-place multiplier leak
    final=bx.load_config(CONFIG/'final.json')
    half=bx.load_config(CONFIG/'final_half.json')
    assert final['priors']['shared_shrinkage']==half['priors']['shared_shrinkage']
    assert final['priors']['shared_shrinkage']['components']==['level','slope','seasonal']
    assert final['priors']['innovation_median']['season']==half['priors']['innovation_median']['season']==.02
    assert bx.load_config(CONFIG/'independent.json')['priors']['shared_shrinkage'] is None


def test_static_seasonal_check_retains_initial_pattern_without_a_seasonal_hyperparameter():
    config=bx.load_config(CONFIG/'adequacy.json')
    config['data'].update(start='2023-01-01',series=['TXm','TXx'])
    config['mcmc'].update(chains=1,chain_workers=1,warmup=1,draws=2,progress=False)
    data=bx.load_uccle_multiseries(**config['data'])
    variant=next(v for v in config['variants'] if v['name']=='fixed_location_seasonality')
    case=configured_variant(config,variant)
    model,prior=joint_model(data,case)
    fit=bx.fit(data,model,priors=prior,**fit_options(case,family=model.family))
    assert set(fit.metadata['shared_shrinkage_members'])=={'level','slope','initial_slope'}
    assert 'shrinkage.shared.seasonal' not in fit.parameter_draws
    for channel in model.channel_names:
        assert 'sd.channel.'+channel+'.seasonal' not in fit.parameter_draws
        seasonal=fit.component_draws('seasonal',channel=channel)
        np.testing.assert_allclose(seasonal[:,12:],seasonal[:,:-12],atol=1e-12)
        assert np.std(seasonal)>.1
        assert np.all(np.isfinite(fit.forecast(12,seed=183).observations))


def test_shared_comparison_plot_handles_two_and_three_component_hierarchies(tmp_path,monkeypatch):
    import bucex.reporting.sensitivity_plots as plots
    import matplotlib.pyplot as plt
    rows=[]
    for variant,components in [('two',['level','slope']),('three',['level','slope','seasonal'])]:
        for component in components:
            for distribution in ['prior','posterior']:
                rows.append(dict(variant=variant,component=component,distribution=distribution,
                                 lower=.001,median=.01,upper=.02))
    captured=[]
    def capture(fig,path,**kwargs):
        captured.extend(text.get_text() for ax in fig.axes for text in ax.texts)
        fig.savefig(path.with_suffix('.png'))
        plt.close(fig)
        return [path.with_suffix('.png')]
    monkeypatch.setattr(plots,'save_figure',capture)
    plots.save_sensitivity_plots({'shared_shrinkage':pd.DataFrame(rows)},tmp_path)
    assert captured==['not pooled']
    assert (tmp_path/'shared_shrinkage.png').exists()
