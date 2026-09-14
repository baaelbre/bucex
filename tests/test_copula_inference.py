"""Full-likelihood contract of the copula state-space sampler."""
import numpy as np
import pytest

import bucex as bx
from bucex.dependence import GaussianCopula
from bucex.inference.fit.shared import _channel_log_likelihood, _innovation_geometry, _log_parameter_prior, _log_state_density
from bucex.inference.fit.shared_slice import gaussian_prior_blocks, _affected_log_likelihood


def _model(correlation=None, mixed=False, shared=False):
    channels = (
        bx.Channel('a', bx.Gaussian(), (bx.LocalLevel(mode='static', initial_mean=0., initial_sd=1.),)),
        bx.Channel('b', bx.GEV() if mixed else bx.Gaussian(),
                   (bx.LocalLevel(mode='static', initial_mean=1. if mixed else 0., initial_sd=1.),)),
    )
    return bx.MultiSeriesModel(channels=channels, copula=GaussianCopula(correlation=correlation),
        shared=(bx.Shared('common', bx.LocalLevel()),) if shared else ())


def _priors(mixed=False, shared=False, unknown_sigma=False):
    return bx.JointPriors(
        {'shared.common.level': bx.HalfNormalSD(.1)} if shared else {},
        {'a': bx.InverseGammaVariance(3., .3) if unknown_sigma else bx.FixedSD(.5), 'b': bx.FixedSD(.7)},
        {'b': bx.UniformPrior(-.4, -.1)} if mixed else {},
    )


def test_copula_without_factor_is_full_joint_model_and_roundtrips():
    model = _model()
    assert model.requires_joint_inference and not model.has_shared_states
    assert not model.supports_fs_parameterization
    assert bx.MultiSeriesModel.from_dict(model.to_dict()) == model
    fit = bx.fit(np.array([[.1, .2], [-.2, -.4], [.3, .1]]), model=model, priors=_priors(),
        mcmc=bx.MCMC(draws=3, warmup=2, chains=1, seed=14))
    assert fit.plan.engine == 'laplace_mh'
    assert fit.plan.targets_exact_posterior
    assert fit.metadata['residual_copula']
    assert not fit.metadata['shared_temporal_state']
    assert fit.auxiliary_draws['copula_correlation'].shape == (1, 3, 2, 2)
    assert 'copula.z.1.0' in fit.parameter_draws
    inverse, _ = _innovation_geometry(fit.compiled)
    for draw in range(3):
        params = {name: float(value[0, draw]) for name, value in fit.parameter_draws.items()}
        path = fit.state_draws[0, draw]
        eta = fit.compiled.eta(path, params=params)
        expected = fit.compiled.observation_log_likelihood(fit.y, eta, params)
        expected += _log_parameter_prior(params, fit.priors, fit.compiled)
        expected += _log_state_density(path, params, fit.compiled, inverse)
        assert fit.log_posterior[0, draw] == pytest.approx(expected)


def test_copula_mixed_sigma_shape_and_ess_targets_include_all_channels():
    fit = bx.fit(np.array([[.1, 1.2], [-.2, .8], [.3, 1.5]]), model=_model(mixed=True, shared=True),
        priors=_priors(mixed=True, shared=True), mcmc=bx.MCMC(draws=2, warmup=1, chains=1, seed=6))
    params = {name: float(value[0, 0]) for name, value in fit.parameter_draws.items()}
    params['copula.z.1.0'] = np.arctanh(.65)
    eta = fit.compiled.eta(fit.state_draws[0, 0], params=params)
    for key, shift in [('sigma.a', .15), ('xi.b', .03)]:
        candidate = {**params, key: params[key] + shift}
        full_delta = fit.compiled.observation_log_likelihood(fit.y, eta, candidate) - fit.compiled.observation_log_likelihood(fit.y, eta, params)
        # Observation-parameter MH must use the complete joint density even
        # when only one channel parameter is proposed.
        actual = _channel_log_likelihood(fit.y, eta, fit.compiled, candidate, 0) - _channel_log_likelihood(fit.y, eta, fit.compiled, params, 0)
        assert actual == pytest.approx(full_delta)
    for block in gaussian_prior_blocks(fit.compiled):
        np.testing.assert_array_equal(block.channels, [0, 1])
        assert _affected_log_likelihood(fit.y, eta, fit.compiled, params, block.channels) == pytest.approx(fit.compiled.observation_log_likelihood(fit.y, eta, params))


def test_gaussian_sigma_is_not_conjugate_under_copula(monkeypatch):
    import bucex.inference.fit.shared as sampler
    def forbidden(*args, **kwargs):
        raise AssertionError('An observation sigma is not conjugate with a residual copula.')
    monkeypatch.setattr(sampler, '_draw_variance', forbidden)
    fit = bx.fit(np.array([[.1, .2], [-.2, -.4], [.3, .1]]), model=_model(correlation=((1., .5), (.5, 1.))),
        priors=_priors(unknown_sigma=True), mcmc=bx.MCMC(draws=2, warmup=2, chains=1, seed=8))
    assert fit.sampler_diagnostics['update_methods']['sigma.a'] == 'log-SD MH'
    assert np.ptp(fit.parameter_draws['sigma.a']) > 0


def test_identity_copula_recovers_exact_gaussian_proposal_acceptance():
    fit = bx.fit(np.array([[.1, .2], [-.2, -.4], [.3, .1]]), model=_model(correlation=((1., 0.), (0., 1.))),
        priors=_priors(), mcmc=bx.MCMC(draws=4, warmup=1, chains=1, seed=18),
        shared_sampler=bx.SharedSampler(elliptical_slice_steps=0))
    np.testing.assert_allclose(fit.sampler_diagnostics['draw_metrics']['laplace_mh_acceptance'], 1.)
    np.testing.assert_allclose(fit.auxiliary_draws['copula_correlation'], np.broadcast_to(np.eye(2), (1,4,2,2)))
    params = {name: value[0,0] for name,value in fit.parameter_draws.items()}
    eta = fit.compiled.eta(fit.state_draws[0,0], params=params)
    assert fit.compiled.observation_log_likelihood(fit.y,eta,params) == pytest.approx(fit.compiled.marginal_observation_log_likelihood(fit.y,eta,params))


def test_copula_ffbs_rejected_and_hierarchical_prior_not_silently_ignored():
    y = np.array([[.1,.2],[.3,.4]])
    with pytest.raises(ValueError, match='incompatible'):
        bx.fit(y, model=_model(), priors=_priors(), engine='ffbs', mcmc=bx.MCMC(draws=1,warmup=0,chains=1))
    with pytest.raises(TypeError, match='JointPriors'):
        bx.fit(y, model=_model(), priors={'some_hierarchy': 1}, mcmc=bx.MCMC(draws=1,warmup=0,chains=1))


def test_strong_gaussian_copula_has_exact_full_covariance_proposal():
    fit = bx.fit(np.array([[.1, .2], [-.2, -.4], [.3, .1]]),
        model=_model(correlation=((1., .97), (.97, 1.))), priors=_priors(),
        mcmc=bx.MCMC(draws=6, warmup=1, chains=1, seed=18),
        shared_sampler=bx.SharedSampler(elliptical_slice_steps=0))
    np.testing.assert_allclose(fit.sampler_diagnostics['draw_metrics']['laplace_mh_acceptance'], 1.)
    assert np.max(np.abs(fit.sampler_diagnostics['draw_metrics']['laplace_mh_mean_log_acceptance_ratio'])) < 1e-8


def test_full_curvature_stabilization_preserves_score_and_missing_margins():
    from bucex.inference.state.laplace import _multivariate_pseudo_data
    y = np.array([[1.,2.], [3.,np.nan]])
    eta = np.array([[.1,.2], [.3,.4]])
    gradient = np.array([[2.,-3.], [4.,0.]])
    # The first information matrix is indefinite; both its eigen floor and
    # the shift bound must preserve the exact Taylor score.
    hessian = np.array([[[1.,2.],[2.,-.1]],[[-.5,0.],[0.,0.]]])
    pseudo, covariance = _multivariate_pseudo_data(y,eta,gradient,hessian,
        curvature_floor=1e-6, maximum_variance=1e8, shift_limit=2.)
    for t in range(2):
        observed = np.flatnonzero(np.isfinite(y[t]))
        difference = pseudo[t,observed] - eta[t,observed]
        np.testing.assert_allclose(np.linalg.solve(covariance[t][np.ix_(observed,observed)],difference),gradient[t,observed],atol=1e-8)
        assert np.linalg.norm(difference) <= 2. + 1e-10
    assert np.isnan(pseudo[1,1])


def test_gaussian_copula_exact_covariance_is_not_modified_by_newton_limits():
    from bucex.models.compiler import compile_model
    from bucex.inference.state.laplace import build_laplace_approximation
    model = _model(correlation=((1., .95), (.95, 1.)))
    y = np.array([[100., 70.], [90., 60.]])
    compiled = compile_model(model,y)
    params = {'sigma.a': .5, 'sigma.b': .7}
    approx = build_laplace_approximation(y,compiled,params,shift_limit=.001,curvature_floor=100.,maximum_variance=.001)
    expected = np.array([[.25,.95*.5*.7],[.95*.5*.7,.49]])
    np.testing.assert_allclose(approx.pseudo_y,y)
    np.testing.assert_allclose(approx.pseudo_variance,np.broadcast_to(expected,(2,2,2)))
