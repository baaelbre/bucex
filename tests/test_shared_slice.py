"""Numerical reference checks for the supplementary exact state refresh."""
from types import SimpleNamespace

import numpy as np
import pytest

import bucex as bx
from bucex.inference.fit.shared_slice import gaussian_prior_blocks, elliptical_slice_block, _zero_mean_prior_draw


def _static_reference():
    model = bx.MultiSeriesModel(
        channels=(
            bx.Channel('a', bx.Gaussian(), (bx.LocalLevel(mode='static', initial_mean=2., initial_sd=1.3),)),
            bx.Channel('b', bx.Gaussian(), (bx.LocalLevel(mode='static', initial_mean=0., initial_sd=0.),)),
        ), shared=(bx.Shared('common', bx.LocalLevel()),),
    )
    y = np.column_stack((np.array([2.6, 3.2, 3.1, 2.9, 2.8, 3.2, 3.3, 3.]), np.zeros(8)))
    compiled = bx.compile_model(model, y)
    params = {'sd.shared.common.level': 0., 'sigma.a': .5, 'sigma.b': .5}
    path = np.broadcast_to(compiled.initial_mean, (9, compiled.state_dim)).copy()
    return compiled, y, params, path


def test_elliptical_slice_matches_nonzero_mean_gaussian_conditional():
    compiled, y, params, path = _static_reference()
    block = gaussian_prior_blocks(compiled)[0]
    rng = np.random.default_rng(3481)
    samples = []
    for iteration in range(3400):
        refresh = elliptical_slice_block(y, path, compiled, params, block, rng)
        path = refresh.path
        if iteration >= 400:
            samples.append(path[0, block.indices[0]])
    variance = 1 / (1 / 1.3**2 + len(y) / .5**2)
    mean = variance * (2. / 1.3**2 + y[:,0].sum() / .5**2)
    assert np.mean(samples) == pytest.approx(mean, abs=.017)
    assert np.var(samples, ddof=1) == pytest.approx(variance, rel=.12)
    # A static prior keeps every time equal without numerical repair.
    np.testing.assert_allclose(path[:,block.indices[0]], path[0,block.indices[0]])


def test_prior_auxiliary_retains_tiny_positive_initial_variance():
    compiled, _, params, _ = _static_reference()
    block = gaussian_prior_blocks(compiled)[0]
    from dataclasses import replace
    tiny = replace(block, initial_cov=np.array([[1e-20]]))
    auxiliary, stochastic = _zero_mean_prior_draw(tiny, compiled.process_vector(params), np.random.default_rng(2))
    assert stochastic
    assert auxiliary[0,0] != 0.0
    np.testing.assert_array_equal(auxiliary[:,0], np.repeat(auxiliary[0,0], len(auxiliary)))


def test_deterministic_block_is_not_reported_as_refreshed():
    compiled, y, params, path = _static_reference()
    block = gaussian_prior_blocks(compiled)[1]
    refresh = elliptical_slice_block(y, path, compiled, params, block, np.random.default_rng(2))
    assert refresh.path is path
    assert not refresh.moved and not refresh.stochastic and refresh.evaluations == 0


def test_exhausted_slice_budget_aborts_instead_of_restoring():
    compiled, y, params, path = _static_reference()
    block = gaussian_prior_blocks(compiled)[0]
    path[:, block.indices] = 3.
    class RejectingRNG:
        def normal(self, size):
            return np.zeros(size)
        def random(self):
            return .999
        def uniform(self, lower, upper):
            return np.pi
    with pytest.raises(FloatingPointError, match='exhausted 1 evaluations'):
        elliptical_slice_block(y, path, compiled, params, block, RejectingRNG(), maximum_evaluations=1)


def test_departure_contrasts_are_one_block_and_all_states_partitioned():
    channels = tuple(bx.Channel(name, bx.Gaussian(), (bx.LocalLevel(mode='static'),)) for name in ('a','b','c'))
    model = bx.MultiSeriesModel(channels, shared=(bx.Shared('common', bx.LocalLinearTrend()), bx.Departures('individual', bx.LocalLinearTrend())))
    compiled = bx.compile_model(model, np.zeros((8,3)))
    blocks = gaussian_prior_blocks(compiled)
    departure = [block for block in blocks if block.name == 'departure.individual']
    assert len(departure) == 1 and len(departure[0].indices) == 4
    np.testing.assert_array_equal(np.sort(np.concatenate([b.indices for b in blocks])), np.arange(compiled.state_dim))


def test_zero_uniform_never_accepts_outside_observation_support(monkeypatch):
    import bucex.inference.fit.shared_slice as module
    compiled, y, params, path = _static_reference()
    block = gaussian_prior_blocks(compiled)[0]
    densities = iter((0., -np.inf))
    monkeypatch.setattr(module, '_affected_log_likelihood', lambda *args: next(densities))
    class ZeroUniformRNG:
        def normal(self, size):
            return np.ones(size)
        def random(self):
            return 0.
        def uniform(self, lower, upper):
            return np.pi / 2
    with pytest.raises(FloatingPointError, match='exhausted'):
        elliptical_slice_block(y, path, compiled, params, block, ZeroUniformRNG(), maximum_evaluations=1)
