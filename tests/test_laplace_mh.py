from __future__ import annotations

import numpy as np
import pytest

import bucex as bx
from bucex.inference.fit.fs_utils import (
    build_ncp_system,
    infer_ncp_layout,
    ncp_laplace_mh,
)


def _gev_sample(seed: int = 710) -> tuple[bx.Model, np.ndarray]:
    model = bx.Model(
        bx.GEV(),
        [bx.LocalLinearTrend(), bx.DummySeasonal(4)],
    )
    parameters = {
        "sd.level": 0.04,
        "sd.slope": 0.003,
        "sd.seasonal": 0.025,
        "sigma": 0.75,
        "xi": -0.18,
    }
    simulation = bx.simulate(
        model,
        24,
        parameters,
        initial_state=np.zeros(5),
        seed=seed,
    )
    return model, simulation.y


def test_version_plan_and_configuration_contract():
    model, values = _gev_sample()
    plan = bx.plan(model, values, engine="laplace_mh", parameterization="fs")
    assert bx.__version__ == "1.8.0"
    assert plan.engine == "laplace_mh"
    assert plan.targets_exact_posterior
    assert plan.approximation is None
    assert plan.proposal == "iterated_laplace_smoother"
    assert "Metropolis-Hastings" in plan.state_update
    assert bx.InferencePlan.from_dict(plan.to_dict()) == plan
    with pytest.raises(ValueError, match="mh_steps"):
        bx.Laplace(mh_steps=0)

    multiseries = bx.MultiSeriesModel(
        channels=(
            bx.Channel("a", bx.GEV(), (bx.LocalLinearTrend(),)),
            bx.Channel("b", bx.GEV(), (bx.LocalLinearTrend(),)),
        )
    )
    joint_plan = bx.plan(multiseries, np.column_stack((values, values)), engine="laplace_mh")
    assert joint_plan.targets_exact_posterior
    assert joint_plan.engine == "laplace_mh"


def test_ncp_laplace_mh_is_exact_for_a_quadratic_observation_layer():
    model = bx.Model(
        bx.Gaussian(),
        [bx.LocalLinearTrend(), bx.DummySeasonal(4)],
    )
    layout = infer_ncp_layout(model)
    params_state = {
        "alpha0": 0.0,
        "beta0": 0.0,
        "gamma0_season": np.zeros(3),
        "s_level": 0.10,
        "q_level": 0.10**2,
        "s_trend": 0.01,
        "q_trend": 0.01**2,
        "s_season": 0.05,
        "q_season": 0.05**2,
    }
    params_obs = {"sigma": 0.4}
    values = np.random.default_rng(711).normal(size=20)
    initial = np.zeros((values.size + 1, layout.ncp_state_dim))
    result = ncp_laplace_mh(
        values,
        model,
        params_state,
        params_obs,
        layout,
        initial,
        rng=np.random.default_rng(712),
        mh_steps=5,
    )

    # For a Gaussian observation layer the Laplace likelihood is exact, so the
    # correction is constant and every proposal must be accepted.
    assert result.exact_invariant
    assert np.all(result.accepted)
    np.testing.assert_allclose(result.log_acceptance_ratio, 0.0, atol=2e-13)

    # The integrated slope and dummy-seasonal lag states have no one-step
    # innovation. Their recursions must hold exactly after numerical support
    # projection, not merely within a loose tolerance.
    transition, covariance = build_ncp_system(layout)
    residual = result.z_path[1:] - result.z_path[:-1] @ transition.T
    deterministic = np.flatnonzero(np.diag(covariance) == 0.0)
    np.testing.assert_array_equal(residual[:, deterministic], 0.0)


@pytest.mark.parametrize("parameterization", ["fs", "disturbance"])
def test_public_laplace_mh_fit_is_exact_and_reports_diagnostics(
    parameterization, tmp_path
):
    model, values = _gev_sample(713)
    fit = bx.fit(
        values,
        model=model,
        priors="normal",
        engine="laplace_mh",
        parameterization=parameterization,
        mcmc=bx.MCMC(draws=2, warmup=1, chains=1, seed=714),
        laplace=bx.Laplace(max_iterations=12, mh_steps=2),
    )

    assert fit.plan.targets_exact_posterior
    assert fit.meta["laplace_mh_exact_invariant"]
    assert fit.config["laplace"]["mh_steps"] == 2
    assert np.all(np.isfinite(fit.state_draws))
    if parameterization == "fs":
        assert not fit.meta["log_posterior_available"]
        assert np.all(np.isfinite(fit.draws_aux["log_likelihood"]))
        assert np.all(np.isnan(fit.log_posterior))
    else:
        assert np.all(np.isfinite(fit.log_posterior))
    acceptance = fit.sampler_diagnostics["acceptance"]["state_laplace_mh"]
    assert acceptance.shape == (1,)
    assert np.all((0.0 <= acceptance) & (acceptance <= 1.0))
    engine = fit.diagnostics()["engine"]
    assert 0.0 <= engine["state_acceptance"] <= 1.0
    assert engine["mean_proposal_support_rejections"] >= 0.0
    if parameterization == "fs":
        assert 0.0 <= engine["initial_support_repair_rate"] <= 1.0
    else:
        assert "initial_support_repair_rate" not in engine
    archive = tmp_path / f"laplace_mh_{parameterization}.bucex"
    fit.save(archive)
    restored = bx.FitResult.load(archive)
    assert restored.plan == fit.plan
    assert restored.config["laplace"]["mh_steps"] == 2


def test_laplace_mh_uses_exact_structural_selection_updates():
    model, values = _gev_sample(715)
    fit = bx.fit(
        values,
        model=model,
        priors="ssvs",
        engine="laplace_mh",
        parameterization="fs",
        mcmc=bx.MCMC(draws=1, warmup=1, chains=1, seed=716),
        laplace=bx.Laplace(max_iterations=12),
    )
    assert fit.meta["model_selection_exact"]
    assert fit.meta["model_selection_basis"] == (
        "exact_gev_rjmh_with_laplace_independence_proposals"
    )
    assert "ssvs_model_mh" in fit.sampler_diagnostics["acceptance"]
