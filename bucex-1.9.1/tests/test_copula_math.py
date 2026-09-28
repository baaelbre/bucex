"""Independent distribution checks for residual copula calculations."""
import json

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr, ndtri
from scipy.stats import genextreme, multivariate_normal, norm

from bucex import Channel, Gaussian, GEV, LocalLevel

from bucex.dependence import (
    GaussianCopula, copula_log_likelihood, copula_observation_derivatives, correlation_from_unconstrained,
    gaussian_copula_logpdf, log_lkj_unconstrained, normal_scores,
    quantiles_from_normal_scores, sample_normal_scores, sample_uniforms,
    unconstrained_from_correlation,
)


def _channels():
    return [Channel("mean", Gaussian(), (LocalLevel(),)),
            Channel("maximum", GEV(), (LocalLevel(),)),
            Channel("minimum", GEV(), (LocalLevel(),), tail="lower")]


def _params(xi=-0.2):
    return {"sigma.mean": 1.3, "sigma.maximum": 1.5, "sigma.minimum": 0.8,
            "xi.maximum": xi, "xi.minimum": xi}


def test_lkj_transform_matches_finite_difference_jacobian():
    k = 4
    theta = np.array([0.2, -0.4, 0.3, 0.1, -0.3, 0.6])
    lower = np.tril_indices(k, -1)
    step = 1e-5
    jacobian = np.column_stack([
        (correlation_from_unconstrained(theta + step * np.eye(6)[j], k)[lower]
         - correlation_from_unconstrained(theta - step * np.eye(6)[j], k)[lower]) / (2 * step)
        for j in range(theta.size)])
    _, log_jacobian = np.linalg.slogdet(jacobian)
    zero = np.zeros_like(theta)
    for eta in [0.5, 1.0, 2.0, 5.0]:
        logdet = np.linalg.slogdet(correlation_from_unconstrained(theta, k))[1]
        direct_difference = (eta - 1) * logdet + log_jacobian
        transformed_difference = log_lkj_unconstrained(theta, k, eta) - log_lkj_unconstrained(zero, k, eta)
        assert transformed_difference == pytest.approx(direct_difference, abs=2e-8)
    np.testing.assert_allclose(unconstrained_from_correlation(correlation_from_unconstrained(theta, k)), theta)


@pytest.mark.parametrize("eta", [0.5, 1.0, 2.0, 5.0])
def test_bivariate_lkj_transformed_density_normalizes(eta):
    integral, error = quad(lambda theta: np.exp(log_lkj_unconstrained([theta], 2, eta)), -np.inf, np.inf)
    assert integral == pytest.approx(1.0, abs=1e-8)
    assert error < 1e-7


def test_copula_declaration_prior_and_fixed_serialization():
    names = ("x", "y", "z")
    spec = GaussianCopula(eta=2.5)
    assert spec.parameter_names(names) == ("copula.z.1.0", "copula.z.2.0", "copula.z.2.1")
    np.testing.assert_array_equal(spec.correlation_matrix(spec.initial_parameters(names), names), np.eye(3))
    assert np.isfinite(spec.log_prior(spec.initial_parameters(names), names))
    for spec in [spec, GaussianCopula(correlation=[[1, -0.4], [-0.4, 1]])]:
        loaded = GaussianCopula.from_dict(json.loads(json.dumps(spec.to_dict())))
        assert loaded == spec
    fixed = GaussianCopula(correlation=np.eye(2))
    assert fixed.initial_parameters(("x", "y")) == {}
    assert fixed.log_prior({}, ("x", "y")) == 0
    with pytest.raises(ValueError, match="shape"):
        fixed.parameter_names(names)
    with pytest.raises(ValueError, match="positive definite"):
        GaussianCopula(correlation=[[1, 1], [1, 1]])


def test_lkj_prior_draw_correlation_marginal_moments():
    generator = np.random.default_rng(1521)
    spec = GaussianCopula(eta=2)
    draws = np.array([spec.sample_correlation(("x", "y", "z"), generator) for _ in range(6000)])
    lower = draws[:, 2, 0]
    assert abs(lower.mean()) < 0.02
    # Under LKJ(eta), any off-diagonal variance is 1/(2*eta+K-1).
    assert lower.var() == pytest.approx(1 / 6, abs=0.012)


def test_copula_density_is_multivariate_normal_density_ratio_with_missing_data():
    correlation = np.array([[1, 0.4, -0.2], [0.4, 1, 0.3], [-0.2, 0.3, 1]])
    scores = np.array([[0.2, -1.1, 0.8], [-0.4, np.nan, 1.5], [np.nan, 0.4, np.nan], [np.nan]*3])
    result = gaussian_copula_logpdf(scores, correlation)
    expected = []
    for row in scores:
        mask = np.isfinite(row)
        expected.append(0.0 if mask.sum() < 2 else
                        multivariate_normal.logpdf(row[mask], cov=correlation[np.ix_(mask, mask)])
                        - norm.logpdf(row[mask]).sum())
    np.testing.assert_allclose(result, expected, atol=1e-14)
    np.testing.assert_array_equal(gaussian_copula_logpdf(scores, np.eye(3)), np.zeros(4))


@pytest.mark.parametrize("xi", [-0.5, -0.2, 0, 1e-10, 0.2, 0.5])
def test_mixed_normal_scores_and_inverse_respect_minimum_orientation(xi):
    scores = np.array([[-1.0, 0.4, 1.2], [2.0, -1.5, -0.5], [0, 0, 0]])
    eta = np.array([[10, 20, -5]] * len(scores))
    channels, params = _channels(), _params(xi)
    values = quantiles_from_normal_scores(scores, eta, channels, params)
    expected = np.column_stack([
        eta[:, 0] + params["sigma.mean"] * scores[:, 0],
        genextreme.ppf(ndtr(scores[:, 1]), -xi, loc=eta[:, 1], scale=params["sigma.maximum"]),
        genextreme.ppf(ndtr(-scores[:, 2]), -xi, loc=eta[:, 2], scale=params["sigma.minimum"])])
    np.testing.assert_allclose(values, expected, atol=2e-12)
    np.testing.assert_allclose(normal_scores(values, eta, channels, params), scores, atol=2e-12)
    # For a minimum, F_original(y) = 1-F_internal(-y), so normal scores flip sign.
    internal_cdf = genextreme.cdf(values[:, 2], -xi, loc=eta[:, 2], scale=params["sigma.minimum"])
    np.testing.assert_allclose(scores[:, 2], ndtri(1 - internal_cdf), atol=2e-12)


def test_normal_score_tail_arithmetic_does_not_clip_probabilities():
    channels = [Channel("maximum", GEV(), (LocalLevel(),))]
    params = {"sigma.maximum": 1.0, "xi.maximum": 0.0}
    # exp(-exp(-800)) rounds to one, although its normal score is finite.
    score = normal_scores([[800.0]], [[0.0]], channels, params)
    assert score[0, 0] > 39
    reconstructed = quantiles_from_normal_scores(score, [[0.0]], channels, params)
    assert reconstructed[0, 0] == pytest.approx(800.0, rel=1e-12)
    direct_score = np.array([[40.0]])
    observation = quantiles_from_normal_scores(direct_score, [[0]], channels, params)
    assert np.isfinite(observation).all()
    np.testing.assert_allclose(normal_scores(observation, [[0]], channels, params), direct_score)
    with pytest.raises(FloatingPointError, match="no clipping"):
        normal_scores([[-1000]], [[0]], channels, params)
    assert copula_log_likelihood([[-1000]], [[0]], channels, params, [[1]]) == -np.inf


def test_invalid_support_is_rejected_and_missing_margin_remains_missing():
    channels, params = _channels(), _params(-0.2)
    y = np.array([[0.0, 0.0, np.nan]])
    eta = np.zeros_like(y)
    assert np.isnan(normal_scores(y, eta, channels, params)[0, 2])
    y[0, 1] = 100
    with pytest.raises(ValueError, match="support"):
        normal_scores(y, eta, channels, params)
    assert copula_log_likelihood(y, eta, channels, params, np.eye(3)) == -np.inf


def test_single_observation_and_multiple_leading_dimensions():
    channels, params = _channels(), _params()
    scores = np.array([0.1, -0.2, 0.3])
    eta = np.zeros(3)
    observed = quantiles_from_normal_scores(scores, eta, channels, params)
    np.testing.assert_allclose(normal_scores(observed, eta, channels, params), scores)
    assert isinstance(gaussian_copula_logpdf(scores, np.eye(3)), float)
    many = np.broadcast_to(scores, (2, 4, 3))
    observed = quantiles_from_normal_scores(many, eta, channels, params)
    np.testing.assert_allclose(normal_scores(observed, eta, channels, params), many)


def test_correlated_draws_have_correct_margins_and_correlation():
    correlation = np.array([[1, 0.7], [0.7, 1]])
    scores = sample_normal_scores(correlation, 20000, np.random.default_rng(903))
    np.testing.assert_allclose(np.cov(scores.T), correlation, atol=0.025)
    assert sample_uniforms(np.eye(2), (3, 4), np.random.default_rng(10)).shape == (3, 4, 2)
    first = sample_normal_scores(np.eye(2), 5, np.random.default_rng(20))
    second = np.random.default_rng(20).normal(size=(5, 2))
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("xi", [-.5, -.2, 0.0, .2, .5])
def test_joint_copula_derivatives_match_full_likelihood_finite_differences(xi):
    channels, params = _channels(), _params(xi)
    correlation = np.array([[1, .5, -.2], [.5, 1, .4], [-.2, .4, 1]])
    eta = np.array([[.2, 1.4, -.8], [.1, 1.1, -.4], [.4, .9, -.1], [0, 0, 0]])
    scores = np.array([[.8, -.6, 1.2], [-.3, .9, .4], [.7, -.9, -.6], [0, 0, 0]])
    observed = quantiles_from_normal_scores(scores, eta, channels, params)
    observed[1, 0] = np.nan
    observed[2, :2] = np.nan
    observed[3, :] = np.nan
    gradient, hessian = copula_observation_derivatives(observed, eta, channels, params, correlation)
    step = 2e-4
    for time in range(eta.shape[0]):
        def objective(location):
            marginal = sum(channel.observation.logpdf(observed[time, j], location[j],
                sigma=params[f"sigma.{channel.name}"], xi=params.get(f"xi.{channel.name}"))
                for j, channel in enumerate(channels) if np.isfinite(observed[time, j]))
            return marginal + copula_log_likelihood(observed[time], location, channels, params, correlation)
        location = eta[time]
        expected_gradient = np.zeros(3)
        expected_hessian = np.zeros((3, 3))
        unit = np.eye(3) * step
        for j in range(3):
            expected_gradient[j] = (objective(location + unit[j])-objective(location-unit[j]))/(2*step)
            expected_hessian[j, j] = (objective(location+unit[j])-2*objective(location)+objective(location-unit[j]))/step**2
            for k in range(j):
                value = (objective(location+unit[j]+unit[k])-objective(location+unit[j]-unit[k])
                    -objective(location-unit[j]+unit[k])+objective(location-unit[j]-unit[k]))/(4*step**2)
                expected_hessian[j, k] = expected_hessian[k, j] = value
        np.testing.assert_allclose(gradient[time], expected_gradient, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(hessian[time], expected_hessian, atol=3e-6, rtol=3e-6)
        missing = np.isnan(observed[time])
        np.testing.assert_array_equal(gradient[time, missing], 0)
        np.testing.assert_array_equal(hessian[time, missing], 0)
        np.testing.assert_array_equal(hessian[time, :, missing], 0)
    # Single-observation API is equivalent to the corresponding batch row.
    single_g, single_h = copula_observation_derivatives(observed[0], eta[0], channels, params, correlation)
    np.testing.assert_array_equal(single_g, gradient[0])
    np.testing.assert_array_equal(single_h, hessian[0])


def test_joint_gaussian_derivatives_are_exact_observed_covariance_precision():
    channels = [Channel(name, Gaussian(), (LocalLevel(),)) for name in ("x", "y", "z")]
    params = {"sigma.x": .8, "sigma.y": 1.5, "sigma.z": 2.0}
    correlation = np.array([[1, .5, -.2], [.5, 1, .4], [-.2, .4, 1]])
    scales = np.array([params[f"sigma.{c.name}"] for c in channels])
    covariance = correlation * np.outer(scales, scales)
    observed = np.array([[.2, -.8, .4], [np.nan, .3, -.5]])
    eta = np.zeros_like(observed)
    gradient, hessian = copula_observation_derivatives(observed, eta, channels, params, correlation)
    for t, row in enumerate(observed):
        mask = np.isfinite(row)
        precision = np.linalg.inv(covariance[np.ix_(mask, mask)])
        np.testing.assert_allclose(gradient[t, mask], precision @ row[mask], atol=1e-14)
        np.testing.assert_allclose(hessian[t][np.ix_(mask, mask)], -precision, atol=1e-14)
