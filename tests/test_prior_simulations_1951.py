"""Independent scientific checks on complete prior replications."""
import copy
import numpy as np
import pandas as pd
import pytest
from scipy.stats import genextreme
import bucex as bx
from research.seasonal.prior_simulations import calendar, simulate_paths, summarize, run_suite, variant_config


def test_calendar_does_not_load_observations(monkeypatch):
    monkeypatch.setattr(bx, 'load_uccle_multiseries', lambda **kw: pytest.fail('Prior simulation read data'))
    config = variant_config('reference')
    dates, history = calendar(config)
    assert history == 538 and len(dates) == 658
    assert str(dates[0].date()) == '1892-03-01'
    assert str(dates[history-1].date()) == '2026-06-01'
    assert str(dates[-1].date()) == '2056-06-01'
    r = simulate_paths(config, draws=4, seed=12, dates=dates)
    assert r['observations'].shape == (4, 658, 6)


def test_horizon_variances_match_state_space_calculation():
    config = variant_config('reference')
    config['data']['series'] = ['TXm']
    dates = pd.date_range('2000-03-01', periods=120, freq='3MS')
    r = simulate_paths(config, draws=24000, seed=1951, dates=dates)
    gains = bx.innovation_response_gains(120, period=4)
    for component, scale in [('level', .1), ('slope', .002), ('seasonal', .1)]:
        assert np.std(r[component][:, -1, 0]) == pytest.approx(gains[component]*scale, rel=.09)
    assert np.std(r['initial_slope'][:, -1, 0]) == pytest.approx(1.2, rel=.025)
    assert np.std(r['rate'][:, -1, 0]) == pytest.approx(40*np.sqrt(.01**2+120*.002**2), rel=.025)


def test_initial_cycle_matches_fitted_fs_lag_basis(monkeypatch):
    from bucex.inference.fit.fs_utils import static_seasonal_design
    config = variant_config('reference')
    config['data']['series'] = ['TXm']
    dates = pd.date_range('2000-03-01', periods=12, freq='3MS')
    original = bx.draw_marginal_prior
    def fixed(*args, **kwargs):
        sample = original(*args, **kwargs)
        p = sample['channels']['TXm']
        p['initial.level'][:] = 3.
        p['initial.slope'][:] = .02
        p['initial.seasonal'][:] = [1., 2., 4.]
        for c in ('level', 'slope', 'seasonal'):
            p['sd.'+c][:] = 0
        return sample
    monkeypatch.setattr(bx, 'draw_marginal_prior', fixed)
    r = simulate_paths(config, draws=2, seed=2, dates=dates)
    expected = 3.+.02*np.arange(1, 13)+static_seasonal_design(12, 3)@np.array([1., 2., 4.])
    np.testing.assert_allclose(r['location'][0, :, 0], expected)


def test_minimum_reflection_and_gev_sampling_are_consistent():
    config = variant_config('reference')
    dates = pd.date_range('2000-03-01', periods=8, freq='3MS')
    r = simulate_paths(config, draws=3000, seed=8, dates=dates)
    for name, sign in [('TXx', 1), ('TXn', -1)]:
        j = r['names'].index(name)
        xi = r['sampled']['channels'][name]['xi'][:, None]
        pit = genextreme.cdf(sign*r['observations'][:, :, j], -xi,
                loc=sign*r['location'][:, :, j], scale=r['scales'][:, :, j])
        assert np.mean(pit) == pytest.approx(.5, abs=.01)
        assert np.mean(pit < .05) == pytest.approx(.05, abs=.006)
        q = genextreme.cdf(sign*r['tail_quantile'][:, :, j], -xi,
                loc=sign*r['location'][:, :, j], scale=r['scales'][:, :, j])
        np.testing.assert_allclose(q, .99, atol=1e-8)


def test_shared_scale_draws_and_initial_rates_have_distinct_dependence():
    config = variant_config('reference')
    dates = pd.date_range('2000-03-01', periods=4, freq='3MS')
    r = simulate_paths(config, draws=25000, seed=6, dates=dates)
    a, b = [r['sampled']['channels'][ch] for ch in ('TXm', 'TNm')]
    expected = (2/np.pi)*(1-2/np.pi)/(1-4/np.pi**2)
    assert np.corrcoef(a['sd.slope'], b['sd.slope'])[0, 1] == pytest.approx(expected, abs=.025)
    assert abs(np.corrcoef(a['initial.slope'], b['initial.slope'])[0, 1]) < .025
    assert set(r['sampled']['shared']) == {'level', 'slope', 'seasonal'}


def test_tail_summaries_do_not_discard_infinities():
    x = np.r_[np.zeros(94), np.full(6, np.inf)]
    s = summarize(x)
    assert np.isinf(s['q975']) and s['nonfinite_fraction'] == .06
    assert np.isnan(summarize([0, np.nan])['median'])


def test_prior_runner_reproducibility_and_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(bx, 'load_uccle_multiseries', lambda **kw: pytest.fail('Prior suite read data'))
    out = run_suite(tmp_path, suite='reference,ss_gamma_2e1', draws=40, figures=False)
    before = (out/'reference/target_summary.csv').read_bytes()
    run_suite(tmp_path, suite='reference,ss_gamma_2e1', draws=40, figures=False)
    assert (out/'reference/target_summary.csv').read_bytes() == before
    assert not bx.load_config(out/'manifest.json')['observations_read']
    wide = pd.read_csv(out/'ss_gamma_2e1/analytic/thirty_year_effects.csv')
    ref = pd.read_csv(out/'reference/analytic/thirty_year_effects.csv')
    assert wide.loc[wide.component == 'seasonal', 'sd'].iloc[0] == pytest.approx(2*ref.loc[ref.component == 'seasonal', 'sd'].iloc[0])
    with pytest.raises(ValueError, match='differs'):
        run_suite(tmp_path, suite='reference', draws=41, figures=False)
