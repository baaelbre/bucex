"""The initialized seasonal state must generate the season used to estimate sigma."""

import numpy as np
import pytest

import bucex as bx
from bucex.api.fit import _default_initial_values
from bucex.inference.fit.fs_utils import (
    baseline_mu_path, infer_ncp_layout, mu_from_ncp, static_seasonal_design,
)
from bucex.inference.fit.private_channel import _initial_channel_state


@pytest.mark.parametrize('period', [4, 12])
def test_initial_dummy_state_reproduces_chronological_seasons(period):
    phases = np.arange(period, dtype=float)
    effects = 5. * np.sin(2. * np.pi * phases / period)
    effects -= effects.mean()
    time = np.arange(20 * period)
    y = 7. + .015 * time + effects[time % period] + .4 * np.sin(time * 2.4)
    model = bx.Model(bx.GEV(), [bx.LocalLinearTrend(), bx.DummySeasonal(period)])

    state_params, obs_params = _default_initial_values(y, model)
    baseline = baseline_mu_path(len(y), state_params, infer_ncp_layout(model))
    np.testing.assert_allclose(obs_params['sigma'], np.std(y-baseline, ddof=1), rtol=1e-12)
    np.testing.assert_allclose(
        static_seasonal_design(period, period-1) @ state_params['gamma0_season'],
        [np.mean(y[time % period == phase])-np.mean(y) for phase in phases.astype(int)],
        atol=1e-12,
    )

    compiled = bx.compile_model(model, y)
    prior = bx.fs_priors('gev', period=period)
    private = _initial_channel_state('cold', 'gev', y, compiled, prior,
                                     np.random.default_rng(21), None)
    actual = mu_from_ncp(private.z_path, private.params_state, private.layout)
    np.testing.assert_allclose(private.params_obs['sigma'], np.std(y-actual, ddof=1), rtol=1e-12)
    np.testing.assert_allclose(private.params_state['gamma0_season'],
                               state_params['gamma0_season'], atol=1e-12)
    assert np.isfinite(np.sum(model.observation.logpdf(y, actual, private.params_obs)))
