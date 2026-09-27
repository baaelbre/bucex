"""Check physical units and saved evidence through the real report writer."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtri
import bucex as bx


@pytest.mark.parametrize('median_absolute', [True, False])
def test_report_physical_effects_intervals_and_saved_fit(tmp_path, median_absolute):
    from research.monthly.models import joint_model, fit_options
    from research.monthly.report import write_report
    source = Path(__file__).resolve().parents[1]/'research/seasonal/config/smoke.json'
    config = bx.load_config(source)
    config['data'].update(series=['TXm', 'TNm'], start='2016-03')
    config['mcmc'].update(chains=2, chain_workers=1, warmup=2, draws=8, progress=False)
    config.update(figures=False, forecast_draws=24, predictive_check_draws=12,
                  prior_draws=40, contrasts=None, save_fits=True)
    if not median_absolute:
        config['priors']['initial_slope_median'] = None
    data = bx.load_uccle_multiseries(**config['data'])
    model, priors = joint_model(data, config)
    fit = bx.fit(data, model, priors=priors, **fit_options(config, family=model.family))
    write_report(fit, tmp_path, config=config, horizon=120, level=.95, risks=config['risks'])
    restored = bx.load_fit(tmp_path/'fit.bucex')
    assert restored.n_chains == 2 and restored.draws_per_chain == 8
    assert len(pd.read_csv(tmp_path/'forecast.csv').time.unique()) == 120
    assert (tmp_path/'period_and_endpoint_targets.csv').read_bytes() == (tmp_path/'scientific_targets.csv').read_bytes()
    for name in ('posterior_predictive_checks', 'residual_serial', 'residual_dependence', 'residual_dependence_by_season'):
        frame = pd.read_csv(tmp_path/f'{name}.csv')
        assert len(frame) and frame.envelope_level.eq(.95).all()

    raw = pd.read_csv(tmp_path/'shared_shrinkage.csv').set_index(['component', 'distribution'])
    effects = pd.read_csv(tmp_path/'shared_shrinkage_effects.csv').set_index(['component', 'distribution'])
    factor = 40/ndtri(.75) if median_absolute else 40
    for distribution in ('prior', 'posterior'):
        key = ('initial_slope', distribution)
        assert effects.loc[key, 'median'] == pytest.approx(raw.loc[key, 'median']*factor)

    horizons = pd.read_csv(tmp_path/'TXm_innovation_effects_by_horizon.csv')
    assert set(zip(horizons.horizon_years, horizons.horizon_updates)) == {(10,40), (30,120)}
    seasonal = horizons[horizons.component.eq('seasonal')].set_index('horizon_years')
    assert seasonal.loc[30, 'median']/seasonal.loc[10, 'median'] == pytest.approx(np.sqrt(3))
    legacy = pd.read_csv(tmp_path/'TXm_innovation_effects.csv')
    assert legacy.horizon_updates.eq(40).all() and legacy.horizon_months.eq(120).all()
    probabilities = pd.read_csv(tmp_path/'TXm_innovation_effect_probabilities.csv')
    assert set(probabilities.threshold) == {.05,.1,.2}
    for row in probabilities.itertuples():
        draws = restored.innovation_effect_draws(row.horizon_updates, channel='TXm', combine_chains=False)[row.component]
        assert row.probability_below == pytest.approx(np.mean(draws < row.threshold))
        assert row.probability_above == pytest.approx(np.mean(draws > row.threshold))
        assert row.n_draws == 16
    comparison = bx.SensitivityReport(posterior_runs={
        'reference': {ch:tmp_path for ch in fit.channel_names}}, baseline='reference', block_frequency='seasonal').tables()
    assert {'innovation_effects', 'innovation_effect_probabilities', 'initial_slope_prior_posterior',
            'shared_shrinkage_effects', 'forecast_uncertainty'} <= set(comparison)
    assert {'seasonal', 'observation_scale'} <= set(comparison['paths'].quantity)
    assert 'channel' in comparison['initial_slope_prior_posterior']
