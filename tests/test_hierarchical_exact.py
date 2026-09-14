"""Regression contracts for exact non-Gaussian hierarchical inference."""
from __future__ import annotations

import importlib

import numpy as np
import pytest

import bucex as bx

hierarchy = importlib.import_module('bucex.inference.fit.hierarchical')


def _problem():
    rng = np.random.default_rng(16001)
    n = 20
    time = np.arange(n)
    seasonal = 0.2 * np.sin(2 * np.pi * time / 4)
    components = (bx.LocalLinearTrend(), bx.DummySeasonal(4))
    model = bx.MultiSeriesModel(channels=(
        bx.Channel('mean', bx.Gaussian(), components),
        bx.Channel('minimum', bx.GEV(), components, tail='lower'),
    ))
    values = np.column_stack((
        0.01 * time + seasonal + rng.normal(scale=0.2, size=n),
        -0.3 + 0.01 * time + seasonal - rng.gumbel(scale=0.25, size=n),
    ))
    return values, model


@pytest.mark.parametrize('workers', [1, 2])
def test_exact_hierarchy_reports_evaluated_density_and_mh_uncertainty(tmp_path, workers):
    values, model = _problem()
    fit = bx.fit(
        values, model,
        priors=bx.HierarchicalPrior(pool='both'),
        mcmc=bx.MCMC(draws=4, warmup=3, chains=1, seed=16002),
        laplace=bx.Laplace(mh_steps=2),
        hierarchical_sampler=bx.HierarchicalSampler(channel_workers=workers),
    )
    assert fit.plan.engine == 'laplace_mh'
    assert fit.plan.targets_exact_posterior
    assert fit.meta['laplace_mh_exact_invariant']
    assert fit.meta['model_selection_exact']
    assert fit.meta['restored_iterations'] == 0
    assert np.all(np.isnan(fit.log_posterior))
    assert not fit.meta['log_posterior_available']
    loglik = fit.draws_aux['log_likelihood']
    predictors = fit.eta_draws(combine_chains=False)
    for draw in range(fit.draws_per_chain):
        params = {k: np.asarray(v)[0, draw] for k, v in fit.parameter_draws.items()}
        expected = fit.compiled.observation_log_likelihood(fit.y, predictors[0, draw], params)
        assert np.isfinite(expected)  # includes transformed lower-tail support
        assert loglik[0, draw] == pytest.approx(expected)
    acceptance = fit.sampler_diagnostics['acceptance']['state_laplace_mh.minimum']
    assert np.all((acceptance >= 0) & (acceptance <= 1))
    assert np.all(np.isfinite(fit.draws_aux['laplace_mh_log_weight']))
    assert not any(key.startswith('particle_') for key in fit.draws_aux)
    archive = tmp_path / 'exact-hierarchy.bucex'
    fit.save(archive)
    restored = bx.FitResult.load(archive)
    np.testing.assert_array_equal(restored.draws_aux['log_likelihood'], loglik)
    np.testing.assert_array_equal(restored.eta_draws(), fit.eta_draws())
    combined = bx.combine_fits([fit, restored])
    assert combined.draws_aux['log_likelihood'].shape == (2, fit.draws_per_chain)


def test_numerical_failure_aborts_exact_hierarchy(monkeypatch):
    values, model = _problem()

    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError('injected factorization failure')

    monkeypatch.setattr(hierarchy, 'ncp_laplace_mh', fail)
    with pytest.raises(RuntimeError, match="channel 'minimum'.*aborted") as caught:
        bx.fit(
            values, model,
            mcmc=bx.MCMC(draws=1, warmup=0, chains=1, seed=16003),
            hierarchical_sampler=bx.HierarchicalSampler(initializer='data'),
        )
    assert isinstance(caught.value.__cause__, np.linalg.LinAlgError)


def test_support_rejected_proposals_are_valid_mh_rejections(monkeypatch):
    """An ordinary support rejection must not be confused with numerical failure."""
    values, model = _problem()
    fs_utils = importlib.import_module('bucex.inference.fit.fs_utils')
    real_correction = fs_utils.ncp_laplace_log_correction
    calls = 0

    def reject_new_proposals(*args, **kwargs):
        nonlocal calls
        calls += 1
        # mh_steps=2: each path update scores the valid incumbent once, then two
        # proposals. Force only proposals outside support, leaving the incumbent.
        return real_correction(*args, **kwargs) if calls % 3 == 1 else -np.inf

    monkeypatch.setattr(fs_utils, 'ncp_laplace_log_correction', reject_new_proposals)
    fit = bx.fit(
        values, model,
        mcmc=bx.MCMC(draws=2, warmup=0, chains=1, seed=16004),
        laplace=bx.Laplace(mh_steps=2),
        hierarchical_sampler=bx.HierarchicalSampler(initializer='data'),
    )
    assert fit.meta['restored_iterations'] == 0
    np.testing.assert_array_equal(fit.draws_aux['laplace_mh_acceptance'], 0.0)
    np.testing.assert_array_equal(fit.draws_aux['laplace_support_rejections'], 2.0)
    assert np.all(np.isfinite(fit.draws_aux['log_likelihood']))


def test_retired_particle_engine_is_not_public_and_cannot_be_selected():
    assert not hasattr(bx, 'pgas')
    assert not hasattr(bx, 'particle_filter')
    assert not hasattr(bx, 'Particles')
    values, model = _problem()
    for retired in ('pgas', 'particle'):
        with pytest.raises(ValueError, match='retired'):
            bx.plan(model, values, engine=retired)
    assert bx.plan(bx.Model(bx.GEV(), (bx.LocalLinearTrend(),)), values[:, 0]).engine == 'laplace_mh'


@pytest.mark.parametrize('family', ['gaussian', 'gev'])
def test_fs_likelihood_is_named_and_matches_saved_draws(family):
    values, _ = _problem()
    y = values[:, 0] if family == 'gaussian' else -values[:, 1]
    fit = bx.fit(
        y, family=family, period=4, priors='normal', parameterization='fs',
        mcmc=bx.MCMC(draws=3, warmup=2, chains=1, seed=16005),
    )
    assert fit.meta['log_likelihood_available']
    assert not fit.meta['log_posterior_available']
    assert np.all(np.isnan(fit.log_posterior))
    eta = fit.eta_draws()
    sigma = fit.parameter('sigma')
    xi = fit.parameter('xi') if family == 'gev' else [None] * fit.n_draws
    actual = fit.draws_aux['log_likelihood'].reshape(-1)
    for draw in range(fit.n_draws):
        expected = np.sum(fit.model.observation.logpdf(
            fit.y, eta[draw], sigma=float(sigma[draw]), xi=xi[draw],
        ))
        assert actual[draw] == pytest.approx(expected)
