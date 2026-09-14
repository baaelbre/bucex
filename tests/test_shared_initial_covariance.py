"""Small positive Gaussian prior variances must remain stochastic."""
from __future__ import annotations

import numpy as np
import pytest

import bucex as bx
from bucex.inference.state.kalman import ffbs, kalman_filter, kalman_smoother


def _small_slope_problem():
    intercept = bx.LocalLevel(mode='static', initial_mean=0.0, initial_sd=10.0)
    model = bx.MultiSeriesModel(
        channels=(bx.Channel('a', bx.Gaussian(), (intercept,)),
                  bx.Channel('b', bx.Gaussian(), (intercept,))),
        shared=(bx.Shared('warming', bx.LocalLinearTrend(
            level_mode='static', trend_mode='static', initial_slope_sd=1e-6,
        )),),
    )
    y = np.zeros((6, 2))
    compiled = bx.compile_model(model, y)
    params = {'sigma.a': 1.0, 'sigma.b': 1.0}
    return compiled, y, params


def test_projection_preserves_small_positive_initial_slope_variance():
    compiled, y, params = _small_slope_problem()
    index = compiled.state_names.index('shared.warming.slope')
    assert compiled.initial_cov[index, index] == pytest.approx(1e-12)
    path = np.zeros((len(y) + 1, compiled.state_dim))
    path[0, index] = 1e-6
    for t in range(1, len(path)):
        path[t] = compiled.transition @ path[t - 1]
    projected = compiled.project_path(path, params)
    np.testing.assert_allclose(projected, path, rtol=1e-12, atol=1e-20)


def test_ffbs_small_slope_matches_analytic_gaussian_conditional():
    compiled, y, params = _small_slope_problem()
    index = compiled.state_names.index('shared.warming.slope')
    # No transition noise: stack H_t F^t to obtain an ordinary Gaussian linear
    # model for the initial state, including its deterministic zero coordinates.
    transition_power = np.eye(compiled.state_dim)
    rows = []
    for design in compiled.design():
        transition_power = compiled.transition @ transition_power
        rows.append(design @ transition_power)
    x = np.concatenate(rows, axis=0)
    covariance = compiled.initial_cov
    observation_cov = np.eye(y.size)
    analytic = covariance - covariance @ x.T @ np.linalg.solve(
        observation_cov + x @ covariance @ x.T, x @ covariance,
    )
    filtered = kalman_filter(y, compiled, params)
    smoothed = kalman_smoother(filtered, compiled)
    np.testing.assert_allclose(smoothed.covariance[0, index, index], analytic[index, index], rtol=1e-7)
    rng = np.random.default_rng(16006)
    slopes = np.array([
        ffbs(y, compiled, params, rng, filter_result=filtered)[0][0, index]
        for _ in range(1600)
    ])
    assert np.var(slopes, ddof=1) == pytest.approx(analytic[index, index], rel=0.15)
    assert abs(np.mean(slopes)) < 5 * np.sqrt(analytic[index, index] / slopes.size)


def test_plan_explains_private_trends_outside_departure_constraint():
    from dataclasses import replace

    compiled, y, _ = _small_slope_problem()
    model = replace(compiled.model, shared=(
        *compiled.model.shared, bx.Departures('departures', bx.LocalLevel()),
    ))
    assert not any('private' in warning for warning in bx.plan(model, y).warnings)
    dynamic_channel = replace(model.channels[0], components=(bx.LocalLevel(),))
    model = replace(model, channels=(dynamic_channel, model.channels[1]))
    warnings = bx.plan(model, y).warnings
    assert any("['a']" in warning and 'weighted mean' in warning for warning in warnings)
