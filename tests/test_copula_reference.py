"""Independent targets for copula inference, including parameter conditionals.

These references do not call BUCEX copula densities or matrix transforms to
compute their expected posterior moments.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.stats import multivariate_normal

import bucex as bx


def _constant_model(*, correlation=None, eta=2.0, initial_sd=(1.2, 0.8)):
    names = tuple(chr(ord("a") + i) for i in range(len(initial_sd)))
    model = bx.MultiSeriesModel(tuple(
        bx.Channel(name, bx.Gaussian(), (bx.LocalLevel(
            mode="static", initial_mean=0.0, initial_sd=sd),))
        for name, sd in zip(names, initial_sd)
    ), copula=bx.GaussianCopula(eta=eta, correlation=correlation))
    return model


def _fixed_priors(names, sigma):
    return bx.JointPriors(process={}, observation_sd={
        name: bx.FixedSD(value) for name, value in zip(names, sigma)
    })


def test_correlated_gaussian_location_posterior_matches_independent_precision():
    """The Gaussian-copula state proposal must reproduce the exact conditional."""
    rho = 0.55
    correlation = np.array([[1., rho], [rho, 1.]])
    sigma = np.array([0.6, 0.9])
    observation_covariance = sigma[:, None] * correlation * sigma[None, :]
    model = _constant_model(correlation=correlation)
    y = np.array([[0.8, -0.4], [0.2, 0.3], [0.7, -0.1]])
    prior_precision = np.diag(1.0 / np.square([1.2, 0.8]))
    noise_precision = np.linalg.inv(observation_covariance)
    covariance = np.linalg.inv(prior_precision + len(y) * noise_precision)
    mean = covariance @ noise_precision @ y.sum(axis=0)
    fit = bx.fit(y, model, priors=_fixed_priors(model.channel_names, sigma),
                 engine="laplace_mh", mcmc=bx.MCMC(
                     draws=3000, warmup=300, chains=1, seed=161101))
    samples = fit.eta_draws(combine_chains=False)[:, :, 0, :]
    assert fit.plan.engine == "laplace_mh"
    np.testing.assert_allclose(fit.sampler_diagnostics["draw_metrics"]["laplace_mh_acceptance"], 1.0, atol=1e-12)
    for index in range(2):
        ess = bx.ess_bulk(samples[..., index])
        assert ess > 100
        assert abs(samples[..., index].mean() - mean[index]) < (
            6 * np.sqrt(covariance[index, index] / ess))
    # Check cross-channel covariance through the variance of the sum and
    # difference. A factorized likelihood has the wrong contrast uncertainty.
    for contrast in (np.array([1., 1.]), np.array([1., -1.])):
        centered = samples @ contrast - mean @ contrast
        variance = float(contrast @ covariance @ contrast)
        ess = bx.ess_bulk(centered**2)
        assert ess > 100
        assert abs(np.mean(centered**2) - variance) < (
            6 * variance * np.sqrt(2 / ess))
    # The retained likelihood must be the multivariate-normal likelihood,
    # including its scale determinant; this checks the advertised density.
    expected = np.array([
        multivariate_normal.logpdf(y, mean=draw, cov=observation_covariance).sum()
        for draw in samples[0, :20]
    ])
    np.testing.assert_allclose(
        fit.auxiliary_draws["log_likelihood"][0, :20], expected,
        atol=1e-10, rtol=1e-10)


def test_copula_observation_scale_posterior_matches_direct_quadrature():
    """A marginal-only sigma update would fail this correlated conditional."""
    rho, fixed_sigma, prior_scale = 0.65, 0.7, 0.9
    model = _constant_model(correlation=((1., rho), (rho, 1.)), initial_sd=(0., 0.))
    priors = bx.JointPriors(process={}, observation_sd={
        "a": bx.HalfNormalSD(prior_scale), "b": bx.FixedSD(fixed_sigma)})
    y = np.array([[.8, .45], [-.4, -.2], [.25, .4], [.7, .2]])
    def log_density(scale):
        if scale <= 0:
            return -np.inf
        first, second = y[:, 0] / scale, y[:, 1] / fixed_sigma
        quadratic = np.sum(first**2 - 2*rho*first*second + second**2) / (1-rho**2)
        return -len(y)*np.log(scale) - .5*quadratic - .5*(scale/prior_scale)**2
    stabilizer = log_density(.5)
    def moment(order):
        return quad(lambda scale: scale**order * np.exp(log_density(scale)-stabilizer),
                    0, np.inf, epsabs=1e-11, epsrel=1e-10)[0]
    normalizer = moment(0)
    first, second, fourth = (moment(order)/normalizer for order in (1, 2, 4))
    fit = bx.fit(y, model, priors=priors, engine="laplace_mh",
                 mcmc=bx.MCMC(draws=7000, warmup=500, chains=1, seed=161102))
    samples = fit.parameter("sigma.a", combine_chains=False)
    ess = bx.ess_bulk(samples)
    assert ess > 100
    assert abs(samples.mean() - first) < 6*np.sqrt((second-first**2)/ess)
    ess_squared = bx.ess_bulk(samples**2)
    assert ess_squared > 100
    assert abs(np.mean(samples**2) - second) < 6*np.sqrt((fourth-second**2)/ess_squared)
    assert "conjugate" not in fit.sampler_diagnostics["update_methods"]["sigma.a"]


def test_unobserved_cross_channel_pairs_retain_the_lkj_prior():
    """Singleton masks contain no correlation information: posterior equals LKJ."""
    eta, dimensions = 2.0, 3
    model = _constant_model(eta=eta, initial_sd=(0.,)*dimensions)
    y = np.full((2*dimensions, dimensions), np.nan)
    for index in range(dimensions):
        y[index, index], y[index+dimensions, index] = 0.2, -0.1
    fit = bx.fit(y, model, priors=_fixed_priors(model.channel_names, (1.,)*dimensions),
                 engine="laplace_mh", mcmc=bx.MCMC(
                     draws=5000, warmup=500, chains=1, seed=161103))
    matrices = fit.auxiliary_draws["copula_correlation"]
    # Under LKJ(eta), every ordinary off-diagonal correlation has the same
    # scaled-beta marginal, E[r²]=1/(2eta+K-1) and E[r⁴]=3/((2eta+K-1)(2eta+K+1)).
    second = 1/(2*eta+dimensions-1)
    fourth = 3/((2*eta+dimensions-1)*(2*eta+dimensions+1))
    for row, column in ((1, 0), (2, 0), (2, 1)):
        samples = matrices[..., row, column]
        ess = bx.ess_bulk(samples)
        assert ess > 100
        assert abs(samples.mean()) < 6*np.sqrt(second/ess)
        squared_ess = bx.ess_bulk(samples**2)
        assert squared_ess > 100
        assert abs(np.mean(samples**2)-second) < 6*np.sqrt((fourth-second**2)/squared_ess)
    assert np.all(np.linalg.eigvalsh(matrices) > 0)


def test_fixed_copula_forecast_has_declared_gaussian_covariance():
    """Forecasting must use a joint residual draw, including unequal scales."""
    rho, sigma = -0.6, np.array([0.4, 1.2])
    covariance = np.outer(sigma, sigma)*np.array([[1., rho], [rho, 1.]])
    model = _constant_model(correlation=((1., rho), (rho, 1.)), initial_sd=(0., 0.))
    fit = bx.fit(np.array([[.1, -.1], [.2, -.2]]), model,
                 priors=_fixed_priors(model.channel_names, sigma),
                 mcmc=bx.MCMC(draws=2, warmup=0, chains=1, seed=161104))
    samples = fit.forecast(1, draws=12000, seed=161105).observations[:, 0, :]
    covariance_se = np.sqrt((np.outer(np.diag(covariance), np.diag(covariance))
                             + covariance**2)/(len(samples)-1))
    np.testing.assert_array_less(np.abs(np.cov(samples, rowvar=False)-covariance),
                                 6*covariance_se)


def test_full_covariance_kalman_matches_dense_missing_data_conditioning():
    """Independent Gaussian conditioning checks covariance masking and logdet."""
    from bucex.inference.state.kalman import kalman_filter, kalman_smoother
    from bucex.inference.state.laplace import (
        build_laplace_approximation, gaussian_approximation_log_likelihood,
    )

    y = np.array([[.1, np.nan], [.3, .2], [np.nan, -.4], [.2, .1]])
    initial_sd = np.array([.5, .7])
    innovation_sd = np.array([.15, .2])
    sigma = np.array([.5, .8])
    correlation = np.array([[1., .65], [.65, 1.]])
    noise = np.outer(sigma, sigma) * correlation
    model = bx.MultiSeriesModel(tuple(
        bx.Channel(name, bx.Gaussian(), (bx.LocalLevel(
            initial_mean=0., initial_sd=sd),))
        for name, sd in zip(("a", "b"), initial_sd)
    ), copula=bx.GaussianCopula(correlation=correlation))
    compiled = bx.compile_model(model, y)
    params = {f"sd.{key}": value for key, value in zip(compiled.noise_names, innovation_sd)}
    params.update({"sigma.a": sigma[0], "sigma.b": sigma[1]})
    # Two independent random walks, then same-time correlated observation noise.
    # Observations use states t=1,...,T, after one transition from x0.
    times = np.arange(1, len(y)+1)
    prior = np.kron(np.ones((len(y), len(y))), np.diag(initial_sd**2))
    prior += np.kron(np.minimum.outer(times, times), np.diag(innovation_sd**2))
    observed_covariance = prior + np.kron(np.eye(len(y)), noise)
    present = np.flatnonzero(np.isfinite(y.ravel()))
    observed_covariance = observed_covariance[np.ix_(present, present)]
    cross = prior[:, present]
    known_y = y.ravel()[present]
    expected_mean = cross @ np.linalg.solve(observed_covariance, known_y)
    expected_covariance = prior - cross @ np.linalg.solve(observed_covariance, cross.T)
    expected_log_density = multivariate_normal.logpdf(known_y, cov=observed_covariance)
    filtered = kalman_filter(y, compiled, params,
        observation_variance=np.broadcast_to(noise, (len(y), 2, 2)))
    # The exported helper must honor the compiled copula even without an
    # explicit covariance override; the two routes have the same target.
    default_filter = kalman_filter(y, compiled, params)
    np.testing.assert_allclose(default_filter.log_likelihood, expected_log_density, atol=1e-11)
    np.testing.assert_allclose(default_filter.filtered_mean, filtered.filtered_mean, atol=1e-11)
    smoothed = kalman_smoother(filtered, compiled)
    np.testing.assert_allclose(smoothed.mean[1:].ravel(), expected_mean, atol=1e-11)
    for t in range(len(y)):
        np.testing.assert_allclose(smoothed.covariance[t+1],
            expected_covariance[2*t:2*t+2, 2*t:2*t+2], atol=1e-11)
    np.testing.assert_allclose(filtered.log_likelihood, expected_log_density, atol=1e-11)
    # For Gaussian margins the full-curvature Laplace approximation is exact.
    approximation = build_laplace_approximation(y, compiled, params)
    np.testing.assert_allclose(approximation.mode_path[1:].ravel(), expected_mean, atol=1e-9)
    proposal_likelihood = gaussian_approximation_log_likelihood(
        approximation.mode_path, approximation, compiled, params)
    true_likelihood = 0.
    for t in range(len(y)):
        mask = np.isfinite(y[t])
        true_likelihood += multivariate_normal.logpdf(y[t, mask],
            mean=approximation.mode_path[t+1, mask], cov=noise[np.ix_(mask, mask)])
    np.testing.assert_allclose(proposal_likelihood, true_likelihood, atol=1e-9)
