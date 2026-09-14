"""Joint predictions and scoring checked against known Gaussian distributions."""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm

import bucex as bx
from bucex.dependence import GaussianCopula, normal_scores
from bucex.inference.plan import InferencePlan


def _fit(*, correlation=.7, inferred=False, mixed=False, regression=False):
    channels = []
    for name, mean, observation, tail in (
        ('high', 1., bx.Gaussian(), None),
        ('low', -1., bx.GEV() if mixed else bx.Gaussian(), 'lower' if mixed else None),
    ):
        components = [bx.LocalLevel(mode='static', initial_mean=mean, initial_sd=0)]
        if regression:
            components.append(bx.Regression(1, initial_mean=(2.,), initial_sd=0))
        channels.append(bx.Channel(name, observation, tuple(components), tail=tail))
    copula = GaussianCopula() if inferred else GaussianCopula(correlation=((1., correlation), (correlation, 1.)))
    model = bx.MultiSeriesModel(tuple(channels), copula=copula)
    signs = np.array([(-1. if channel.tail == "lower" else 1.) for channel in channels])
    y = np.array([[1., -1.], [1., -1.]]) * signs
    exog = {'high': np.zeros((2, 1)), 'low': np.zeros((2, 1))} if regression else None
    compiled = bx.compile_model(model, y, exog=exog)
    parameters = {'sigma.high': np.ones((1, 2)), 'sigma.low': np.ones((1, 2))}
    if mixed:
        parameters['xi.low'] = np.full((1, 2), -.2)
    if inferred:
        parameters['copula.z.1.0'] = np.full((1, 2), np.arctanh(correlation))
    states = np.broadcast_to(compiled.initial_mean, (1, 2, 3, compiled.state_dim)).copy()
    return bx.FitResult(model=model, compiled=compiled, priors=None, y=y, state_draws=states,
                        parameter_draws=parameters, log_posterior=np.zeros((1, 2)),
                        plan=InferencePlan('mixed', 'laplace_mh', 'centered', False, 'exact', True, None),
                        exog=exog, transform_sign=signs)


def test_joint_predictive_covariance_and_gaussian_density(tmp_path):
    fit = _fit(inferred=True)
    predictive = fit.posterior_predictive(draws=12000, seed=16101)
    np.testing.assert_allclose(np.cov(predictive.observations[:, 0].T), [[1., .7], [.7, 1.]], atol=.035)
    observed = np.array([[.1, -.8], [2., -2.]])
    expected = multivariate_normal.logpdf(observed, mean=[1., -1.], cov=[[1., .7], [.7, 1.]])
    np.testing.assert_allclose(predictive.joint_conditional_log_density(observed),
                               np.broadcast_to(expected, (12000, 2)), atol=1e-12)
    np.testing.assert_allclose(predictive.joint_log_score(observed), -expected, atol=1e-12)
    future = fit.forecast(1, draws=12000, seed=16102)
    np.testing.assert_allclose(np.cov(future.observations[:, 0].T), [[1., .7], [.7, 1.]], atol=.035)
    np.testing.assert_allclose(future.compound_probability({'high': ('>', 1), 'low': ('>', -1)}),
                               [.25 + np.arcsin(.7)/(2*np.pi)], atol=.015)
    path = tmp_path / 'copula.bucex'
    fit.save(path)
    restored = bx.FitResult.load(path)
    np.testing.assert_allclose(restored.copula_correlation_draws(), np.broadcast_to([[1., .7], [.7, 1.]], (2, 2, 2)))
    np.testing.assert_array_equal(restored.forecast(3, seed=4).observations, fit.forecast(3, seed=4).observations)
    assert restored.model.copula == fit.model.copula
    assert restored.warm_start()['parameters']['copula.z.1.0'] == np.arctanh(.7)


def test_mixed_minimum_copula_orientation_preserved():
    fit = _fit(correlation=.75, mixed=True)
    predictive = fit.posterior_predictive(draws=10000, seed=16103)
    internal_y = predictive.observations * np.array([1., -1.])
    internal_eta = predictive.eta * np.array([1., -1.])
    z = normal_scores(internal_y, internal_eta, fit.model.channels,
                      {'sigma.high': 1., 'sigma.low': 1., 'xi.low': -.2})
    np.testing.assert_allclose(np.corrcoef(z[:, 0].T), [[1., .75], [.75, 1.]], atol=.025)
    # The lower extreme's ORIGINAL CDF is increasing in original Celsius.
    threshold = np.full(2, -1.)
    expected = 1 - bx.GEV().cdf(-threshold, 1., sigma=1., xi=-.2)
    np.testing.assert_allclose(predictive.pit(threshold, channel='low'), expected)
    y = np.array([[1., -1.], [0., -2.]])
    expected_log = predictive.conditional_log_density(y[:, 0], channel='high') + predictive.conditional_log_density(y[:, 1], channel='low')
    actual = predictive.joint_conditional_log_density(y)
    assert np.all(np.isfinite(actual)) and not np.allclose(actual, expected_log)
    # Marginal support violations have joint density zero, not NaN or clipping.
    unsupported = np.array([[1., -100.], [0., -2.]])
    assert np.all(np.isneginf(predictive.joint_conditional_log_density(unsupported)[:, 0]))


def test_joint_score_averages_joint_density_before_mixing():
    fit = _fit(correlation=0)
    prediction = fit.posterior_predictive(seed=16104)
    # Two posterior components with opposing means. Conditional independence
    # is insufficient to factor the posterior-predictive density.
    prediction.eta[:] = np.array([[[2., 2.], [2., 2.]], [[-2., -2.], [-2., -2.]]])
    observed = np.array([[2., 2.], [2., -2.]])
    conditional = np.stack([multivariate_normal.logpdf(observed, mean=mean, cov=np.eye(2))
                            for mean in ([2., 2.], [-2., -2.])])
    correct = -(logsumexp(conditional, axis=0) - np.log(2))
    np.testing.assert_allclose(prediction.joint_log_score(observed), correct)
    wrong = prediction.log_score(observed[:, 0], channel='high') + prediction.log_score(observed[:, 1], channel='low')
    assert not np.allclose(correct, wrong)


def test_copula_regression_forecast_requires_and_uses_future_covariates(tmp_path):
    fit = _fit(correlation=.2, regression=True)
    x = {'high': np.array([[1.], [3.]]), 'low': np.array([[2.], [4.]])}
    future = fit.forecast(2, exog_future=x, seed=16105)
    np.testing.assert_allclose(future.eta, np.broadcast_to([[3., 3.], [7., 7.]], (2, 2, 2)))
    path = tmp_path / 'regression.bucex'
    fit.save(path)
    restored = bx.FitResult.load(path)
    np.testing.assert_array_equal(restored.forecast(2, exog_future=x, seed=16105).observations, future.observations)


def test_joint_score_marginalizes_missing_channels():
    fit = _fit(correlation=.8)
    prediction = fit.posterior_predictive(seed=16106)
    observed = np.array([[np.nan, -1.5], [1.5, np.nan]])
    expected = -norm.logpdf([-.5, .5])
    np.testing.assert_allclose(prediction.joint_log_score(observed), expected)
    import pytest
    with pytest.raises(ValueError, match="at least one"):
        prediction.joint_log_score(np.full((2, 2), np.nan))
