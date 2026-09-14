"""Scientific checks for prior integration, moments and annual aggregation."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
import bucex as bx


def test_ssvs_prior_retains_atoms_and_sd_variance_scaling():
    prior = bx.ssvs_gev_priors(innovation_slab_sd={"level": .2, "trend": .003, "season": .4})
    scaled = bx.innovation_prior_variant(prior, multipliers={"trend": 2})
    original = bx.draw_structural_prior(prior, 50000, seed=19)
    changed = bx.draw_structural_prior(scaled, 50000, seed=19)
    np.testing.assert_array_equal(changed["sd.slope"], 2*original["sd.slope"])
    np.testing.assert_array_equal(changed["sd.level"], original["sd.level"])
    assert abs(np.mean(original["sd.level"] == 0)-.5) < .01
    assert abs(np.mean(original["sd.slope"] == 0)-2/3) < .01
    assert np.mean(changed["sd.slope"]**2) == pytest.approx(4*np.mean(original["sd.slope"]**2))


def test_lasso_prior_integrates_gamma_and_observation_variance():
    prior = bx.ssvs_gaussian_priors(sigma2_prior=bx.InverseGammaPrior(8, 14))
    prior = bx.innovation_prior_variant(prior, profile="manuscript_lasso", lasso_shape=8, lasso_rate=7)
    samples = bx.draw_structural_prior(prior, 150000, seed=33)
    # E[s^2] = E[sigma^2] E[tau] = (14/7) * (2*7/7) = 4.
    assert np.mean(samples["sd.level"]**2) == pytest.approx(4, rel=.035)
    assert np.mean(samples["sigma"]**2) == pytest.approx(2, rel=.01)


def _forecast(family="gaussian", xi=None, tail="upper", draws=3):
    eta = np.repeat(np.arange(draws, dtype=float)[:, None], 12, axis=1)
    observation = bx.Gaussian() if family == "gaussian" else bx.GEV(xi_bounds=(-1, 1))
    parameters = {"sigma": np.full(draws, 2.0)}
    if xi is not None:
        parameters["xi"] = np.full(draws, xi)
    return bx.Forecast(observations=eta.copy(), eta=eta, states=eta[..., None],
        parameters=parameters, dates=pd.date_range("2025-01-01", periods=12, freq="MS").to_numpy(),
        family=family, tail=tail, observation_model=observation, transform_sign=-1 if tail == "lower" else 1,
        period=12, state_names=("level",), component_designs={"univariate.level": np.ones(1)})


def test_forecast_total_variance_uses_conditional_mean_and_noise():
    forecast = _forecast()
    table = bx.forecast_uncertainty(forecast)
    assert set(table.nominal) == {.9, .95, .99}
    assert set(table.target) == {"level", "location", "observation"}
    np.testing.assert_allclose(table.estimated_conditional_mean_variance, 2/3)
    np.testing.assert_allclose(table.estimated_expected_observation_variance, 4)
    np.testing.assert_allclose(table.estimated_total_observation_variance, 4+2/3)


def test_gev_boundary_shape_has_infinite_variance_but_valid_quantiles():
    table = bx.forecast_uncertainty(_forecast("gev", xi=.5))
    assert not table.all_retained_conditional_second_moments_finite.any()
    assert np.isinf(table.estimated_total_observation_variance).all()
    assert np.isfinite(table.estimated_conditional_mean_variance).all()
    assert np.isfinite(table.upper_quantile).all()


def test_gev_parameter_uncertainty_changes_conditional_mean_variance():
    forecast = _forecast("gev", xi=0)
    forecast.eta[:] = 0
    forecast.states[:] = 0
    forecast.parameters["sigma"] = np.array([1., 2., 3.])
    table = bx.forecast_uncertainty(forecast)
    np.testing.assert_allclose(table.latent_location_variance, 0)
    np.testing.assert_allclose(table.estimated_conditional_mean_variance, np.euler_gamma**2*2/3)
    np.testing.assert_allclose(table.estimated_expected_observation_variance, np.pi**2/6*14/3)


def test_annual_cdf_integrates_product_over_common_posterior_draw():
    forecast = _forecast(draws=2)
    forecast.parameters["sigma"][:] = 1
    forecast.eta[:] = -norm.ppf(np.array([.2, .8]))[:, None]
    result = bx.annual_aggregation_check(forecast, 0).iloc[0]
    assert result.analytic_predictive_cdf == pytest.approx((.2**12+.8**12)/2)
    assert result.product_of_posterior_mean_cdfs == pytest.approx(.5**12)
    assert result.months == 12


def test_annual_aggregation_rejects_partial_calendar_year():
    forecast = _forecast()
    forecast.dates = pd.date_range("2025-03-01", periods=12, freq="MS").to_numpy()
    with pytest.raises(ValueError, match="complete"):
        bx.annual_aggregation_check(forecast, 0)
