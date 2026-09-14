from __future__ import annotations

import numpy as np
import pytest

import bucex as bx
import bucex.inference.fit.fs_gev as fs_gev_module
from bucex.inference.fit.fs_utils import (
    _ncp_exact_observation_loglik,
    build_ncp_laplace_approximation,
    build_ncp_system,
    infer_ncp_layout,
    ncp_laplace_mh,
)


def _integrated_slope_support_case():
    model = bx.Model(bx.GEV(), [bx.LocalLinearTrend()])
    layout = infer_ncp_layout(model)
    y = np.asarray([0.0, 5.0, 5.0, 5.0, 5.0])
    params_state = {
        "alpha0": 0.0,
        "beta0": 0.0,
        "s_level": 0.0,
        "q_level": 0.0,
        "s_trend": 1.0,
        "q_trend": 1.0,
    }
    params_obs = {"sigma": 1.0, "xi": -0.25}

    transition, _ = build_ncp_system(layout)
    current = np.zeros((y.size + 1, layout.ncp_state_dim))
    for time in range(1, y.size + 1):
        current[time] = transition @ current[time - 1]
        if time == 1:
            current[time, int(layout.idx_tilde_beta)] += 5.0
        elif time == 2:
            current[time, int(layout.idx_tilde_beta)] -= 5.0
    return model, layout, y, params_state, params_obs, current


def test_negative_xi_support_repair_handles_an_integrated_slope():
    model, layout, y, params_state, params_obs, current = (
        _integrated_slope_support_case()
    )
    zero = np.zeros_like(current)
    assert not np.isfinite(
        _ncp_exact_observation_loglik(
            y, zero, params_state, params_obs, model, layout
        )
    )
    assert np.isfinite(
        _ncp_exact_observation_loglik(
            y, current, params_state, params_obs, model, layout
        )
    )

    first = build_ncp_laplace_approximation(
        y,
        model,
        params_state,
        params_obs,
        layout,
        max_iterations=15,
    )
    second = build_ncp_laplace_approximation(
        y,
        model,
        params_state,
        params_obs,
        layout,
        max_iterations=15,
    )
    assert first.initial_support_repaired
    assert first.converged
    assert np.isfinite(first.objective)
    np.testing.assert_array_equal(first.mode_path, second.mode_path)

    transition, covariance = build_ncp_system(layout)
    residual = first.mode_path[1:] - first.mode_path[:-1] @ transition.T
    deterministic = np.flatnonzero(np.diag(covariance) == 0.0)
    np.testing.assert_array_equal(residual[:, deterministic], 0.0)

    result = ncp_laplace_mh(
        y,
        model,
        params_state,
        params_obs,
        layout,
        current,
        rng=np.random.default_rng(113),
        mh_steps=2,
        max_iterations=15,
    )
    assert result.exact_invariant
    assert result.approximation.initial_support_repaired
    assert np.all(np.isfinite(result.z_path))


def test_exact_fs_numerical_failure_aborts_without_retries(monkeypatch):
    model = bx.Model(
        bx.GEV(),
        [bx.LocalLinearTrend(trend_mode="off")],
    )
    simulation = bx.simulate(
        model,
        12,
        {"sd.level": 0.02, "sigma": 1.0, "xi": -0.20},
        initial_state=np.asarray([20.0]),
        seed=114,
    )
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise FloatingPointError("forced Laplace-MH construction failure")

    monkeypatch.setattr(fs_gev_module, "ncp_laplace_mh", fail_once)
    with pytest.raises(
        RuntimeError,
        match="stopped before recording a restored or duplicate posterior draw",
    ):
        bx.fit(
            simulation.y,
            model=model,
            priors="normal",
            engine="laplace_mh",
            parameterization="fs",
            mcmc=bx.MCMC(draws=1, warmup=1, chains=1, seed=115),
            laplace=bx.Laplace(max_iterations=5),
        )
    assert calls == 1
