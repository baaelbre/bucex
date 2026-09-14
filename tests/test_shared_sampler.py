"""Target-density and failure-policy checks for joint shared-state inference."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import invgamma, norm

from bucex.inference.fit.shared import (
    _centered_sd_target, _draw_variance, _innovation_geometry, _log_state_density,
    _noncentered_scale_step,
)
from bucex.priors import FixedSD, HalfNormalSD, InverseGammaVariance, JointPriors


def test_centered_logsd_target_has_change_of_variable_jacobian():
    prior = HalfNormalSD(0.7)
    count, ss = 9, 2.7
    first, second = 0.2, 0.8
    actual = _centered_sd_target(np.log(first), prior, count, ss) - _centered_sd_target(np.log(second), prior, count, ss)
    expected = prior.logpdf(first) - prior.logpdf(second)
    expected += -(count - 1) * np.log(first / second) - 0.5 * ss * (first**-2 - second**-2)
    assert actual == pytest.approx(expected)


def test_inverse_gamma_update_counts_all_contrast_innovations():
    prior = InverseGammaVariance(3.0, 0.5)
    # Four dates, two independent contrast innovations sharing one SD.
    epsilon = np.arange(8, dtype=float).reshape(4, 2) / 10
    count, ss = epsilon.size, float(np.square(epsilon).sum())
    rng = np.random.default_rng(42)
    variance = np.array([_draw_variance(prior, count, ss, rng)**2 for _ in range(12000)])
    target = invgamma(a=prior.shape + count / 2, scale=prior.scale + ss / 2)
    assert variance.mean() == pytest.approx(target.mean(), rel=0.035)
    assert np.quantile(variance, 0.9) == pytest.approx(target.ppf(0.9), rel=0.04)


def test_repeated_innovation_names_and_affine_density_volume():
    compiled = SimpleNamespace(
        loading=np.array([[2.0, 0], [0, 3.0], [0, 0]]),
        noise_names=("departure.level",),
        innovation_names=("departure.level", "departure.level"),
        transition=np.eye(3), initial_mean=np.zeros(3), initial_cov=np.zeros((3, 3)),
        process_vector=lambda params: np.repeat(params['sd.departure.level'], 2),
    )
    inverse, groups = _innovation_geometry(compiled)
    assert groups['departure.level'].tolist() == [0, 1]
    path = np.array([[0., 0., 0.], [0.4, -0.6, 0.]])
    density = _log_state_density(path, {'sd.departure.level': 0.5}, compiled, inverse)
    # Two independent state coordinates have SDs 2*.5 and 3*.5.
    expected = norm.logpdf(0.4, scale=1.) + norm.logpdf(-0.6, scale=1.5)
    assert density == pytest.approx(expected)


def test_fixed_zero_is_exactly_deterministic():
    compiled = SimpleNamespace(
        loading=np.eye(1), noise_names=('level',), innovation_names=('level',),
        transition=np.eye(1), initial_mean=np.zeros(1), initial_cov=np.zeros((1,1)),
        process_vector=lambda params: np.array([params['sd.level']]),
    )
    inverse, _ = _innovation_geometry(compiled)
    assert _log_state_density(np.zeros((3,1)), {'sd.level': 0.}, compiled, inverse) == 0.
    assert _log_state_density(np.array([[0.], [1.], [1.]]), {'sd.level': 0.}, compiled, inverse) == -np.inf


def test_joint_priors_roundtrip_retains_fixed_scales():
    prior = JointPriors(process={'shared.level': HalfNormalSD(.03)}, observation_sd={'mean': FixedSD(1.)})
    restored = JointPriors.from_dict(prior.to_dict())
    assert restored == prior


def test_noncentered_scale_move_updates_reused_columns_together():
    compiled = SimpleNamespace(
        loading=np.eye(2), transition=np.eye(2),
        eta=lambda path, params=None: path[1:],
        observation_log_likelihood=lambda y, eta, params: 0.,
    )
    path = np.array([[1., 3.], [2., 5.], [4., 6.]])
    class DeterministicRNG:
        def normal(self, scale):
            return np.log(2.)
        def random(self):
            return 1e-50
    params = {'sd.departure.level': .2}
    changed, accepted = _noncentered_scale_step(
        np.zeros((2,2)), path, params, 'departure.level', HalfNormalSD(1.), .1,
        compiled, np.eye(2), {'departure.level': np.array([0,1])}, DeterministicRNG(),
    )
    assert accepted
    assert params['sd.departure.level'] == pytest.approx(.4)
    np.testing.assert_allclose(changed[0], path[0])
    np.testing.assert_allclose(np.diff(changed, axis=0), 2*np.diff(path, axis=0))


def _mixed_shared_fixture():
    import bucex as bx
    from scipy.stats import genextreme
    model = bx.MultiSeriesModel(
        channels=(
            bx.Channel('mean', bx.Gaussian(), (bx.LocalLevel(mode='static', initial_mean=0., initial_sd=1.),)),
            bx.Channel('tail', bx.GEV(), (bx.LocalLevel(mode='static', initial_mean=1., initial_sd=1.),)),
        ),
        shared=(bx.Shared('common', bx.LocalLevel()), bx.Departures('difference', bx.LocalLevel())),
    )
    rng = np.random.default_rng(17)
    t = np.arange(12) / 40
    y = np.column_stack((t + rng.normal(0, .1, 12), t + 1 + genextreme.rvs(c=.25, scale=.2, size=12, random_state=rng)))
    priors = bx.JointPriors(
        process={'shared.common.level': bx.InverseGammaVariance(3., .002), 'departure.difference.level': bx.HalfNormalSD(.04)},
        observation_sd={'mean': bx.InverseGammaVariance(3., .02), 'tail': bx.HalfNormalSD(.4)},
        shape={'tail': bx.UniformPrior(-.4, -.1)},
    )
    return model, y, priors


def test_shared_mixed_posterior_support_diagnostics_and_reproducibility():
    import bucex as bx
    from bucex.inference.fit.shared import _log_parameter_prior
    model, y, priors = _mixed_shared_fixture()
    settings = dict(model=model, priors=priors, mcmc=bx.MCMC(draws=3, warmup=2, chains=1, seed=42), laplace=bx.Laplace(max_iterations=12))
    first, second = bx.fit(y, **settings), bx.fit(y, **settings)
    np.testing.assert_array_equal(first.state_draws, second.state_draws)
    assert np.all(np.isfinite(first.log_posterior))
    inverse, _ = _innovation_geometry(first.compiled)
    for draw in range(first.draws_per_chain):
        params = {name: values[0,draw] for name, values in first.parameter_draws.items()}
        path = first.state_draws[0,draw]
        ll = first.compiled.observation_log_likelihood(first.y, first.compiled.eta(path, params=params), params)
        expected = ll + _log_parameter_prior(params, first.priors) + _log_state_density(path, params, first.compiled, inverse)
        assert first.log_posterior[0,draw] == pytest.approx(expected)
    engine = first.diagnostics()['engine']
    assert np.isfinite(engine['state_acceptance'])
    assert np.isfinite(engine['median_relative_change'])


def test_shared_numerical_state_failure_aborts(monkeypatch):
    import bucex as bx
    import bucex.inference.fit.shared as shared
    model, y, priors = _mixed_shared_fixture()
    def fail(*args, **kwargs):
        raise FloatingPointError('injected numerical failure')
    monkeypatch.setattr(shared, 'laplace_mh', fail)
    with pytest.raises(RuntimeError, match='no posterior result was returned.*injected'):
        bx.fit(y, model=model, priors=priors, mcmc=bx.MCMC(draws=1, warmup=0, chains=1))


def test_generic_laplace_rejects_incumbent_outside_initial_support():
    import bucex as bx
    from bucex.inference.state.laplace import laplace_mh
    model, y, priors = _mixed_shared_fixture()
    result = bx.fit(y, model=model, priors=priors, mcmc=bx.MCMC(draws=1, warmup=0, chains=1, seed=42))
    path = result.state_draws[0,0].copy()
    params = {name: value[0,0] for name, value in result.parameter_draws.items()}
    # The common initial level is fixed at zero, so a shift is off prior support.
    index = result.state_names.index('shared.common.level')
    path[:,index] += 1.
    with pytest.raises(ValueError, match='Gaussian prior support'):
        laplace_mh(result.y, result.compiled, params, path, np.random.default_rng(4))


def test_shared_initializer_rejects_misspelled_outer_fields():
    from bucex.inference.fit.shared import _initial_parameters
    with pytest.raises(ValueError, match='put scalar values in parameters'):
        _initial_parameters(JointPriors({}, {'mean': FixedSD(1.)}), {'sigma.mean': 2.})


def test_laplace_stabilization_preserves_score_in_flat_tail():
    from bucex.inference.state.laplace import _pseudo_data
    compiled = SimpleNamespace(observation_derivatives=lambda y, eta, params: (np.array([[2., -3.]]), np.array([[1e-3, -1e-10]])))
    eta = np.array([[4., 5.]])
    pseudo, variance = _pseudo_data(np.zeros((1,2)), eta, compiled, {}, curvature_floor=1e-6, maximum_variance=1e8, shift_limit=10.)
    np.testing.assert_allclose((pseudo - eta) / variance, [[2., -3.]])
    assert np.max(abs(pseudo - eta)) <= 10.


def test_shared_regression_fit_forecast_and_archive(tmp_path):
    import bucex as bx
    rng = np.random.default_rng(10)
    exog = {'a': np.linspace(-1., 1., 12)[:,None]}
    y = np.column_stack((2*exog['a'][:,0], np.ones(12))) + rng.normal(0,.1,(12,2))
    model = bx.MultiSeriesModel(
        channels=(
            bx.Channel('a', bx.Gaussian(), (
                bx.LocalLevel(mode='static', initial_mean=0., initial_sd=1.),
                bx.Regression(1, initial_mean=(0.,), initial_sd=2.),
            )),
            bx.Channel('b', bx.Gaussian(), (bx.LocalLevel(mode='static', initial_mean=1., initial_sd=1.),)),
        ), shared=(bx.Shared('common', bx.LocalLevel()),),
    )
    prior = bx.JointPriors({'shared.common.level': bx.FixedSD(.02)}, {'a': bx.FixedSD(.1), 'b': bx.FixedSD(.1)})
    fit = bx.fit(y, model=model, exog=exog, priors=prior, mcmc=bx.MCMC(draws=5, warmup=1, chains=1, seed=3))
    np.testing.assert_allclose(fit.exog['a'], exog['a'])
    destination = tmp_path / 'regression.bucex'
    fit.save(destination)
    restored = bx.load_fit(destination)
    np.testing.assert_allclose(restored.exog['a'], exog['a'])
    np.testing.assert_allclose(restored.eta_draws(), fit.eta_draws())
    with pytest.raises(ValueError, match='exog_future'):
        fit.forecast(len(y), seed=42)
    future = {'a': np.array([[1.2], [1.4]])}
    original = fit.forecast(2, exog_future=future, seed=42)
    loaded = restored.forecast(2, exog_future=future, seed=42)
    np.testing.assert_allclose(original.eta, loaded.eta)
    # The same future state simulation gives different predictors under changed
    # known covariates; fitted regression coefficients are propagated by draw.
    shifted = fit.forecast(2, exog_future={'a': future['a'] + 1.}, seed=42)
    coefficients = fit.state_draws[:, :, -1, fit.state_names.index('channel.a.regression[1]')].reshape(-1)
    np.testing.assert_allclose(shifted.eta[:,:,0] - original.eta[:,:,0], coefficients[:,None] * np.ones((1,2)))
