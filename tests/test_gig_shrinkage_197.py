"""GIG full-conditionals and the public half-normal fit contract."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

import bucex as bx
from bucex.inference.fit import shrinkage as implementation
from bucex.inference.fit.shrinkage import (
    SharedShrinkageState, _half_normal_gig_log_multiplier,
)


@pytest.mark.parametrize("count,anchor,relative_size", [
    (1, .01, .03), (2, .0002, 1.), (2, .0002, .01),
    (6, .01, 1.e-6), (6, .0002, 20.),
])
def test_gig_draws_match_independent_density_quadrature(count, anchor, relative_size):
    values = anchor * relative_size * np.linspace(-.8, 1.2, count)
    r = np.linalg.norm(values / anchor)
    # Mode and CDF of the conditional in log(tau/A), expressed from the
    # original normal coefficient densities and half-normal hyperprior.
    z_mode = 2 * r**2 / (np.hypot(count - 1, 2 * r) + count - 1)
    mode = .5 * np.log(z_mode)

    def log_density(u):
        tau = anchor * np.exp(u)
        return norm.logpdf(values, scale=tau).sum() + norm.logpdf(tau, scale=anchor) + u

    offset = log_density(mode)
    density = lambda u: np.exp(log_density(u) - offset)
    low, high = mode - 25., mode + 25.
    mass = quad(density, low, high, epsabs=1e-10)[0]
    quantiles = [brentq(lambda u: quad(density, low, u)[0] / mass - p, low, high)
                 for p in (.025, .5, .975)]
    rng = np.random.default_rng(197 + count)
    draws = np.array([_half_normal_gig_log_multiplier(values, anchor=anchor, rng=rng)
                      for _ in range(6000)])
    np.testing.assert_allclose([(draws <= q).mean() for q in quantiles],
                               [.025, .5, .975], atol=.025, rtol=0)


def test_half_normal_bypasses_slice_and_previous_scale(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Half-normal shrinkage must not call slice sampling.")
    monkeypatch.setattr(implementation, "_slice_sample_real", forbidden)
    spec = bx.SharedShrinkage.half_normal({'level': .01})
    members = {'level': [SimpleNamespace(params_state={'s_level': s}) for s in (.003, -.02)]}
    left = SharedShrinkageState(spec, {'level': -200.}, members)
    right = SharedShrinkageState(spec, {'level': 200.}, members)
    metrics = left.update(np.random.default_rng(12))
    right.update(np.random.default_rng(12))
    assert left.log_multipliers == right.log_multipliers
    assert metrics == {'shared_shrinkage_gig_draws.level': 1.}
    assert left.update_method == 'gig'


@pytest.mark.parametrize('family', ['lognormal', 'half_t', 'half_cauchy'])
def test_other_hyperpriors_retain_slice_sampling(monkeypatch, family):
    called = []
    def slice_update(u, target, rng, *, width):
        assert np.isfinite(target(u))
        called.append(u)
        return .1, 7
    monkeypatch.setattr(implementation, '_slice_sample_real', slice_update)
    spec = bx.SharedShrinkage.from_sd({'level': .01}, hyperprior=family)
    sampler = SharedShrinkageState(spec, {'level': 0.}, {
        'level': [SimpleNamespace(params_state={'s_level': s}) for s in (.003, -.02)]})
    assert sampler.update(np.random.default_rng(1)) == {'shared_shrinkage_slice_evaluations.level': 7}
    assert called == [0.] and sampler.update_method == 'slice'


def test_tiny_coefficients_are_not_squared_or_floored():
    values = 1.e-190 * np.arange(1., 7.)
    assert np.sum(values**2) == 0.  # A naive sum of squares underflows.
    draw = _half_normal_gig_log_multiplier(values, anchor=.01, rng=np.random.default_rng(197))
    tau = .01 * np.exp(draw)
    assert 1.e-192 < tau < 1.e-188


@pytest.mark.parametrize('values', [[], [[1., 2.]], [np.nan], [np.inf]])
def test_invalid_coefficients_rejected(values):
    with pytest.raises(ValueError, match='nonempty vector of finite'):
        _half_normal_gig_log_multiplier(values, anchor=.01, rng=np.random.default_rng(1))


def test_exact_zero_conditional_is_not_replaced_with_a_variance_floor():
    with pytest.raises(ValueError, match='improper'):
        _half_normal_gig_log_multiplier(np.zeros(6), anchor=.01, rng=np.random.default_rng(1))


@pytest.mark.parametrize('initial', [{'initial_slope_sd': .001}, {'initial_slope_median': .001}])
def test_optional_initial_slope_units_are_preserved(monkeypatch, initial):
    spec = replace(bx.SharedShrinkage.half_normal({'level': .01}), **initial)
    values = np.array([.002, -.003])
    members = [SimpleNamespace(params_state={'s_level': s, 'beta0': s}) for s in values]
    seen = []
    def capture(coefficients, *, anchor, rng):
        seen.append((np.asarray(coefficients), anchor))
        return 0.
    monkeypatch.setattr(implementation, '_half_normal_gig_log_multiplier', capture)
    sampler = SharedShrinkageState(spec, dict(level=0., initial_slope=0.),
                                   dict(level=members, initial_slope=members))
    sampler.update(np.random.default_rng(1))
    np.testing.assert_allclose(seen[0][0], values)
    np.testing.assert_allclose(seen[1][0], values / spec.coefficient_sd('initial_slope', 1.))
    assert seen[1][1] == .001


def problem(families):
    components = [bx.LocalLinearTrend(), bx.DummySeasonal(4)]
    names = [f'y{i}' for i in range(len(families))]
    model = bx.MultiSeriesModel([
        bx.Channel(name, bx.Gaussian(scale=bx.SeasonalScale(4)) if family == 'gaussian'
                   else bx.GEV(scale=bx.SeasonalScale(4)), components,
                   tail='lower' if i == 1 and family == 'gev' else 'upper')
        for i, (name, family) in enumerate(zip(names, families))])
    data = pd.DataFrame(np.random.default_rng(9).normal(size=(20, len(names))), columns=names)
    priors = bx.MarginalPriors({name: bx.fs_priors(family, period=4)
                               for name, family in zip(names, families)},
        shrinkage=bx.SharedShrinkage.half_normal({'level': .01, 'slope': .0002, 'seasonal': .01}))
    return data, model, priors


@pytest.mark.parametrize('families', [('gaussian', 'gaussian'), ('gev', 'gev'), ('gaussian', 'gev')])
def test_api_fit_archive_restart_and_forecast(tmp_path, families):
    data, model, priors = problem(families)
    options = bx.MCMC(chains=2, chain_workers=1, warmup=2, draws=6, seed=197)
    fit = bx.fit(data, model, priors=priors, mcmc=options)
    assert fit.metadata['shrinkage_scale_update'] == 'gig'
    assert fit.metadata['bucex_version'] == bx.__version__
    assert np.isfinite(fit.state_draws).all()
    assert 'initial_slope' not in fit.priors.shrinkage.anchors
    metrics = fit.sampler_diagnostics['draw_metrics']
    for component in priors.shrinkage.anchors:
        assert np.all(fit.parameter_draws['shrinkage.shared.' + component] > 0)
        np.testing.assert_array_equal(metrics['shared_shrinkage_gig_draws.' + component], 1.)
        assert 'shared_shrinkage_slice_evaluations.' + component not in metrics
    fit.save(tmp_path / 'fit.bucex')
    restored = bx.load_fit(tmp_path / 'fit.bucex')
    assert restored.metadata['shrinkage_scale_update'] == 'gig'
    assert restored.priors.shrinkage == priors.shrinkage
    np.testing.assert_array_equal(restored.state_draws, fit.state_draws)
    for key in fit.parameter_draws:
        np.testing.assert_array_equal(restored.parameter_draws[key], fit.parameter_draws[key])
    restarted = bx.fit(data, model, priors=restored.priors, init=restored,
                       mcmc=bx.MCMC(chains=1, warmup=1, draws=2, seed=198))
    assert np.isfinite(restarted.forecast(8, draws=20, seed=197).observations).all()
    if families == ('gaussian', 'gev'):
        parallel = bx.fit(data, model, priors=priors, mcmc=replace(options, chain_workers=2))
        np.testing.assert_array_equal(parallel.state_draws, fit.state_draws)
        for key in fit.parameter_draws:
            np.testing.assert_array_equal(parallel.parameter_draws[key], fit.parameter_draws[key])


def test_individual_hierarchy_and_active_member_count(monkeypatch):
    # Check membership in a public fit where one channel's level is static.
    components = [bx.LocalLinearTrend(), bx.LocalLinearTrend(), bx.LocalLinearTrend(level_mode='static')]
    model = bx.MultiSeriesModel([bx.Channel(name, bx.Gaussian(), [component])
                                for name, component in zip(('a', 'b', 'fixed'), components)])
    data = pd.DataFrame(np.random.default_rng(71).normal(size=(10, 3)), columns=model.channel_names)
    prior = bx.MarginalPriors({name: bx.fs_priors('gaussian') for name in model.channel_names},
                             shrinkage=bx.SharedShrinkage.half_normal({'level': .01}))
    counts = []
    original = implementation._half_normal_gig_log_multiplier
    def observed(coefficients, **kwargs):
        counts.append(len(coefficients))
        return original(coefficients, **kwargs)
    monkeypatch.setattr(implementation, '_half_normal_gig_log_multiplier', observed)
    fit = bx.fit(data, model, priors=prior, mcmc=bx.MCMC(chains=1, warmup=1, draws=2, seed=197))
    assert set(counts) == {2}
    assert fit.metadata['shared_shrinkage_members']['level'] == ['a', 'b']
    counts.clear()
    single = bx.MultiSeriesModel([model.channels[0]])
    independent = bx.MarginalPriors({'a': prior.channels['a']},
        shrinkage=bx.IndependentShrinkage.half_normal({'level': .01}))
    fit = bx.fit(data[['a']], single, priors=independent,
                 mcmc=bx.MCMC(chains=1, warmup=1, draws=2, seed=197))
    assert set(counts) == {1} and fit.metadata['shrinkage_scale_update'] == 'gig'
    assert 'independent_shrinkage_gig_draws.level' in fit.sampler_diagnostics['draw_metrics']
