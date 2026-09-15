"""Independent likelihood and end-to-end contracts for the 1.6.3 backend."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import bucex as bx
from bucex.inference.fit.conditional_margin import ConditionalMargin, conditional_score_parameters


@pytest.mark.parametrize("observation,xi", [(bx.Gaussian(), None), (bx.GEV(), -.35), (bx.GEV(), .3), (bx.GEV(), 0.)])
def test_conditional_margin_is_a_density_with_correct_curvature(observation, xi):
    eta = np.array([-.3, .5, 1.])
    y = eta + np.array([.7, -.2, .3])
    params = {"sigma": np.array([.8, 1.3, 2.])}
    if xi is not None:
        params["xi"] = xi
    margin = ConditionalMargin(observation, np.array([.2, -.5, .7]), .42)
    z = margin.scores(y, eta, params)
    expected = observation.logpdf(y, eta, params) + norm.logpdf(z, margin.mean, np.sqrt(.42)) - norm.logpdf(z)
    np.testing.assert_allclose(margin.logpdf(y, eta, params), expected, atol=1e-12)
    eps = 2e-5
    plus, base, minus = (margin.logpdf(y, eta+shift, params) for shift in (eps, 0, -eps))
    np.testing.assert_allclose(margin.grad_eta(y, eta, params), (plus-minus)/(2*eps), atol=1e-7)
    np.testing.assert_allclose(margin.hess_eta(y, eta, params), (plus-2*base+minus)/eps**2, atol=1e-5)


def test_conditional_scores_reflect_minima():
    z = np.array([[.5, -.4], [1.2, 2.]])
    R = np.array([[1, .7], [.7, 1]])
    mean, variance = conditional_score_parameters(z, R, 1, sign=-1)
    np.testing.assert_allclose(mean, -.7*z[:,0])
    assert variance == pytest.approx(.51)


def _small_fit(copula=True, seasonal=True):
    components = [bx.LocalLinearTrend(), bx.DummySeasonal(4)]
    scale = bx.SeasonalScale(4) if seasonal else None
    model = bx.MultiSeriesModel([
        bx.Channel("mean", bx.Gaussian(scale=scale), components),
        bx.Channel("minimum", bx.GEV(scale=scale), components, tail="lower")],
        copula=bx.GaussianCopula() if copula else None)
    data = pd.DataFrame(np.random.default_rng(9).normal(size=(20,2)), columns=model.channel_names,
                        index=pd.date_range("2000-01-01", periods=20, freq="MS"))
    prior = bx.MarginalPriors({"mean": bx.ssvs_gaussian_priors(4), "minimum": bx.ssvs_gev_priors(4)})
    return bx.fit(data, model, priors=prior, parameterization="fs",
                  mcmc=bx.MCMC(chains=1, warmup=3, draws=4, seed=12))


def test_mixed_copula_seasonal_prediction_density_and_archive(tmp_path):
    fit = _small_fit()
    assert fit.plan.targets_exact_posterior
    assert fit.metadata["copula_feedback"] and fit.metadata["structural_ssvs"]
    assert not fit.metadata["shared_temporal_state"]
    for channel in fit.channel_names:
        np.testing.assert_allclose(fit.parameter(f"scale.seasonal.{channel}").sum(axis=-1), 0, atol=1e-14)
        assert fit.sigma_draws(channel=channel).shape == (4,20)
        assert len(fit.component_probabilities(channel=channel)) > 0
    future = fit.forecast(4, seed=41)
    replicated = fit.posterior_predictive(seed=42)
    for prediction in (future, replicated):
        assert np.all(np.isfinite(prediction.observations))
        assert np.all(np.isfinite(prediction.joint_conditional_log_density(prediction.observations[0])))
    path = tmp_path / "copula.bucex"
    fit.save(path)
    loaded = bx.load_fit(path)
    np.testing.assert_array_equal(fit.sigma_draws(channel="minimum"), loaded.sigma_draws(channel="minimum"))
    restarted = bx.fit(loaded.y * loaded.transform_sign, loaded.model, priors=loaded.priors,
                       dates=loaded.dates, init=loaded, mcmc=bx.MCMC(chains=1, warmup=1, draws=1, seed=44))
    assert np.all(np.isfinite(restarted.auxiliary_draws["log_likelihood"]))


def test_univariate_seasonal_api_dates_and_warm_start(tmp_path):
    y = pd.Series(np.random.default_rng(5).normal(size=24), index=pd.date_range("2001-03-01", periods=24, freq="MS"))
    model = bx.Model(bx.Gaussian(scale=bx.SeasonalScale()), [bx.LocalLinearTrend(), bx.DummySeasonal(12)])
    fit = bx.fit(y, model, priors=bx.ssvs_gaussian_priors(), mcmc=bx.MCMC(chains=1, draws=4, warmup=2, seed=22))
    assert not fit.is_multiseries_model
    assert list(fit.component_probabilities().index) == ["level", "slope", "seasonal"]
    assert np.isclose(fit.structural_model_probabilities()["probability"].sum(), 1.)
    assert len(fit.component_transition_summary()) == 3
    expected = fit.parameter("sigma") * np.exp(fit.parameter("scale.seasonal")[:,2])
    np.testing.assert_allclose(fit.sigma_draws()[:,0], expected)
    future = fit.forecast(12, seed=9)
    np.testing.assert_allclose(future.parameters["sigma_path"][:,0], expected)
    assert np.all(np.isfinite(future.log_score(np.zeros(12))))
    fit.save(tmp_path / "univariate.bucex")
    loaded = bx.load_fit(tmp_path / "univariate.bucex")
    bx.fit(y, loaded.model, priors=loaded.priors, init=loaded, mcmc=bx.MCMC(chains=1, draws=1, warmup=1, seed=23))


def test_fixed_identity_copula_equals_independent_private_kernel():
    fit = _small_fit(copula=False, seasonal=True)
    independent_copula = replace(fit.model, copula=bx.GaussianCopula(correlation=np.eye(2)))
    other = bx.fit(fit.y * fit.transform_sign, independent_copula, priors=fit.priors, dates=fit.dates,
                   parameterization="fs", mcmc=bx.MCMC(chains=1, warmup=3, draws=4, seed=12))
    np.testing.assert_allclose(other.state_draws, fit.state_draws, rtol=1e-11, atol=1e-11)
    for name in fit.parameter_draws:
        np.testing.assert_allclose(other.parameter_draws[name], fit.parameter_draws[name], rtol=1e-11, atol=1e-11)


def test_support_contract_rejects_unsupported_scale_and_missing_data():
    with pytest.raises(ValueError, match="phi"):
        bx.GEV(phi="rw", scale=bx.SeasonalScale())
    fit = _small_fit()
    y = fit.y.copy(); y[0,0] = np.nan
    with pytest.raises(ValueError, match="complete aligned"):
        bx.fit(y, fit.model, priors=fit.priors, mcmc=bx.MCMC(chains=1, draws=1, warmup=1))


def test_short_record_can_initialize_unobserved_seasonal_phases():
    y = pd.Series([1., .8, -.1, -.6, .2, .7],
                  index=pd.date_range("2001-03-01", periods=6, freq="MS"))
    model = bx.Model(bx.Gaussian(scale=bx.SeasonalScale()),
                     [bx.LocalLinearTrend(), bx.DummySeasonal(12)])
    fit = bx.fit(y, model, priors=bx.ssvs_gaussian_priors(),
                 mcmc=bx.MCMC(chains=1, warmup=1, draws=2, seed=79))
    assert np.all(np.isfinite(fit.state_draws))
    assert np.all(np.isfinite(fit.forecast(12, seed=80).observations))
