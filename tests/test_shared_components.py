"""Shared-state scientific contracts, independently of sampler implementation."""
from __future__ import annotations

import numpy as np
import pytest

import bucex as bx
from tests._gaussian_reference import dense_gaussian_predictor_posterior


def _model(*, seasonal=False, mixed=False, weights=None):
    observations = (bx.Gaussian(), bx.Gaussian(), bx.GEV() if mixed else bx.Gaussian())
    channels = []
    for name, observation, baseline in zip(("mean", "maximum", "minimum"), observations, (1., 3., -2.)):
        components = [bx.LocalLevel(mode="static", initial_mean=baseline, initial_sd=0.4)]
        if seasonal:
            components.append(bx.DummySeasonal(4, mode="static", initial_sd=0.3))
        channels.append(bx.Channel(name, observation, tuple(components),
                                   tail="lower" if mixed and name == "minimum" else None))
    kwargs = {} if weights is None else {"weights": dict(zip(("mean", "maximum", "minimum"), weights))}
    return bx.MultiSeriesModel(tuple(channels), shared=(
        bx.Shared("warming", bx.LocalLinearTrend(
            initial_level=0, initial_level_sd=0, initial_slope=0,
            initial_slope_sd=0.02)),
        bx.Departures("departure", bx.LocalLinearTrend(
            initial_level=0, initial_level_sd=0, initial_slope=0,
            initial_slope_sd=0.01), **kwargs),
    ))


def _prior(model):
    return bx.JointPriors(
        process={"shared.warming.level": bx.FixedSD(0.10),
                 "shared.warming.slope": bx.FixedSD(0.008),
                 "departure.departure.level": bx.FixedSD(0.035),
                 "departure.departure.slope": bx.FixedSD(0.003)},
        observation_sd={channel.name: bx.FixedSD(0.3) for channel in model.channels},
        shape={channel.name: bx.UniformPrior(-0.4, 0.2)
               for channel in model.channels if channel.family == "gev"},
    )


def _parameters(model, priors):
    result = {f"sd.{key}": prior.initial() for key, prior in priors.process.items()}
    result.update({f"sigma.{key}": prior.initial() for key, prior in priors.observation_sd.items()})
    result.update({f"xi.{key}": prior.initial() for key, prior in priors.shape.items()})
    return result


def test_shared_gaussian_posterior_matches_dense_conditioning():
    model = _model()
    prior = _prior(model)
    rng = np.random.default_rng(16001)
    values = np.array((1., 3., -2.)) + rng.normal(0, 0.4, (6, 3))
    compiled = bx.compile_model(model, values)
    parameters = _parameters(model, prior)
    expected_mean, expected_cov = dense_gaussian_predictor_posterior(compiled, values, parameters)
    fit = bx.fit(values, model, priors=prior,
                 mcmc=bx.MCMC(draws=1200, warmup=0, chains=1, seed=16002))
    draws = fit.eta_draws().reshape((1200, -1))
    assert fit.plan.engine == "ffbs"
    # Fixed parameters give independent FFBS draws. Bounds use Monte Carlo SEs,
    # not arbitrary fit tolerances. The whole covariance checks shared uncertainty.
    mean_error = np.abs(draws.mean(axis=0) - expected_mean)
    assert np.all(mean_error <= 6 * np.sqrt(np.diag(expected_cov) / len(draws)) + 1e-10)
    covariance_se = np.sqrt((np.outer(np.diag(expected_cov), np.diag(expected_cov))
                             + expected_cov ** 2) / (len(draws) - 1))
    assert np.all(np.abs(np.cov(draws, rowvar=False) - expected_cov)
                  <= 6 * covariance_se + 1e-10)


def test_inferred_shared_sd_matches_integrated_gaussian_posterior():
    """Check centered/NCP parameter updates against quadrature, without latent MCMC."""
    from scipy.integrate import quad

    model = bx.MultiSeriesModel(tuple(
        bx.Channel(name, bx.Gaussian(), (bx.LocalLevel(
            mode="static", initial_mean=0, initial_sd=0),))
        for name in ("a", "b")), shared=(
            bx.Shared("common", bx.LocalLevel(initial_mean=0, initial_sd=0)),))
    prior = bx.JointPriors(process={"shared.common.level": bx.HalfNormalSD(.5)},
                           observation_sd={"a": bx.FixedSD(.5), "b": bx.FixedSD(.5)})
    y = np.array([[.8, .7], [.7, .5]])
    # Integrate the random-walk state analytically. Per-time channel means
    # have covariance q² * min(t,s) + .5²/2 * I; channel differences are ancillary.
    def density(q):
        covariance = q*q*np.array([[1., 1.], [1., 2.]]) + .125*np.eye(2)
        means = y.mean(axis=1)
        return np.exp(-.5*(q/.5)**2 - .5*np.linalg.slogdet(covariance)[1]
                      - .5*means @ np.linalg.solve(covariance, means))
    normalizer = quad(density, 0, np.inf, epsabs=1e-11)[0]
    mean = quad(lambda q: q*density(q), 0, np.inf, epsabs=1e-11)[0] / normalizer
    variance = quad(lambda q: (q-mean)**2*density(q), 0, np.inf, epsabs=1e-11)[0] / normalizer
    fitted = bx.fit(y, model, priors=prior,
                    mcmc=bx.MCMC(draws=4000, warmup=1000, chains=2, seed=16006))
    samples = fitted.parameter("sd.shared.common.level", combine_chains=False)
    ess = bx.ess_bulk(samples)
    assert ess > 100
    assert abs(samples.mean() - mean) < 6*np.sqrt(variance/ess)
    expected_second = variance + mean*mean
    fourth = quad(lambda q: q**4*density(q), 0, np.inf, epsabs=1e-11)[0] / normalizer
    second_ess = bx.ess_bulk(samples*samples)
    assert abs(np.mean(samples*samples) - expected_second) < 6*np.sqrt(
        (fourth-expected_second**2)/second_ess)


@pytest.mark.parametrize("mixed", [False, True])
def test_shared_decomposition_orientation_risk_and_archive(tmp_path, mixed):
    weights = (0.2, 0.3, 0.5)
    model = _model(seasonal=True, mixed=mixed, weights=weights)
    rng = np.random.default_rng(16003)
    time = np.arange(16)
    values = np.array((1., 3., -2.)) + 0.025 * time[:, None]
    values = values + np.sin(2 * np.pi * time[:, None] / 4) * np.array((0.4, 0.2, 0.1))
    values = values + rng.normal(0, 0.1, (16, 3))
    fit = bx.fit(values, model, priors=_prior(model),
                 mcmc=bx.MCMC(draws=8, warmup=4, chains=1, seed=16004))
    shared = fit.shared_draws("warming")
    departures = np.stack([fit.departure_draws(name) for name in fit.channel_names], axis=-1)
    np.testing.assert_allclose(departures @ np.array(weights), 0, atol=1e-12)
    for index, name in enumerate(fit.channel_names):
        level = fit.channel_component_draws(name, component="level")
        seasonal = fit.channel_component_draws(name, component="seasonal")
        predictor = fit.channel_eta_draws(name, original_scale=True)
        np.testing.assert_allclose(predictor, level + seasonal, atol=1e-12)
        # The individual baseline is constant; time changes decompose completely.
        np.testing.assert_allclose(level[:, 1:] - level[:, :1],
                                   shared[:, 1:] - shared[:, :1]
                                   + departures[:, 1:, index] - departures[:, :1, index],
                                   atol=1e-12)
    if mixed:
        sigma = fit.parameter("sigma.minimum")[:, None]
        shape = fit.parameter("xi.minimum")[:, None]
        original_eta = fit.channel_eta_draws("minimum", original_scale=True)
        expected = 1 - bx.GEV().cdf(2., -original_eta, sigma=sigma, xi=shape)
        probability = fit.exceedance_probability_draws(-2., channel="minimum", return_labels=False)
        np.testing.assert_allclose(probability, expected, atol=1e-12)
    path = tmp_path / "shared.bucex"
    fit.save(path)
    restored = bx.FitResult.load(path)
    assert restored.model.to_dict() == fit.model.to_dict()
    np.testing.assert_array_equal(restored.eta_draws(), fit.eta_draws())
    np.testing.assert_array_equal(restored.shared_draws(), shared)
    np.testing.assert_array_equal(restored.departure_draws("minimum"), departures[:, :, 2])
    forecast = fit.forecast(4, draws=12, seed=16005)
    restored_forecast = restored.forecast(4, draws=12, seed=16005)
    np.testing.assert_array_equal(forecast.eta, restored_forecast.eta)
    assert forecast.observations.shape == (12, 4, 3)
    assert np.all(np.isfinite(forecast.observations))
    future_departures = np.stack([forecast.departure_draws(name)
                                  for name in fit.channel_names], axis=-1)
    np.testing.assert_allclose(future_departures @ np.array(weights), 0, atol=1e-12)
    for index, name in enumerate(fit.channel_names):
        np.testing.assert_allclose(forecast.eta[:, :, index],
                                   forecast.component_draws("level", channel=name)
                                   + forecast.component_draws("seasonal", channel=name),
                                   atol=1e-12)
    if mixed:
        engine = fit.diagnostics()["engine"]
        assert 0 <= engine["state_acceptance"] <= 1
