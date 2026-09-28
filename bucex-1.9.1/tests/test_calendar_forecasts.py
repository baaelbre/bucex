"""Independent probability and calendar checks for aggregate predictions."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
import bucex as bx


def forecast(start="2024-01-01", periods=12, *, family="gaussian", tail="upper"):
    dates = pd.date_range(start, periods=periods, freq="MS").to_numpy()
    eta = np.vstack([np.linspace(1, 3, periods), np.linspace(2, 5, periods)])
    model = bx.Gaussian() if family == "gaussian" else bx.GEV()
    parameters = {"sigma": np.array([1., 2.]), "sigma_path": np.tile(np.linspace(.5, 2, periods), (2, 1))}
    if family == "gev":
        parameters["xi"] = np.zeros(2)
    return bx.Forecast(observations=eta.copy(), eta=eta, states=eta[..., None], parameters=parameters,
        dates=dates, family=family, tail=tail, observation_model=model,
        transform_sign=-1. if tail == "lower" else 1., period=12, state_names=("level",))


def test_gaussian_annual_mean_day_weights_and_conditional_noise_variance():
    f = forecast()
    aggregate = f.aggregate()
    weights = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])/366
    expected_mean = f.eta @ weights
    expected_variance = f.parameters["sigma_path"]**2 @ weights**2
    np.testing.assert_allclose(aggregate.observations[:, 0], expected_mean)
    mean, variance = aggregate.conditional_moments()
    np.testing.assert_allclose(mean[:, 0], expected_mean)
    np.testing.assert_allclose(variance[:, 0], expected_variance)
    np.testing.assert_allclose(aggregate.probability_draws(3)[:, 0], norm.sf(3, expected_mean, np.sqrt(expected_variance)))
    assert not np.allclose(expected_variance, (f.parameters['sigma_path'] @ weights)**2)
    np.testing.assert_allclose(f.aggregate(weighting="equal").observations[:, 0], f.eta.mean(axis=1))


def test_partial_2026_is_omitted_and_djf_uses_ending_year():
    f = forecast("2026-09-01", 24)
    annual = f.aggregate()
    assert annual.periods.year.tolist() == [2027]
    all_years = f.aggregate(include_partial=True)
    assert all_years.periods.complete.tolist() == [False, True, False]
    assert all_years.periods.months.tolist() == [4, 12, 8]
    seasons = f.aggregate(frequency="season")
    winter = seasons.periods.query("window == 'DJF'").iloc[0]
    assert winter.label == "DJF 2027"
    assert winter.start == pd.Timestamp("2026-12-01")
    assert winter.end == pd.Timestamp("2027-02-01")
    custom = f.aggregate(months=(12, 1, 2))
    assert custom.periods.year.tolist() == [2027, 2028]
    np.testing.assert_array_equal(custom.observations, seasons.observations[:, seasons.periods.window == "DJF"])


def test_no_complete_year_returns_empty_not_a_fake_annual_value():
    f = forecast("2026-09-01", 12)
    aggregate = f.aggregate()
    assert aggregate.observations.shape == (2, 0)
    assert aggregate.summary().empty
    assert aggregate.risk_summary(2).empty


@pytest.mark.parametrize("tail,reduction", [("upper", "max"), ("lower", "min")])
def test_gev_aggregate_risks_use_original_orientation_and_common_draw(tail, reduction):
    f = forecast(family="gev", tail=tail)
    # Known Gumbel CDF, including the reflected lower-tail response orientation.
    threshold = 2.5
    sign = f.transform_sign
    w_cdf = np.exp(-np.exp(-(sign*threshold-sign*f.eta)/f.parameters['sigma_path']))
    original_cdf = w_cdf if sign == 1 else 1-w_cdf
    aggregate = f.aggregate()
    assert aggregate.reduction == reduction
    expected = 1-np.prod(original_cdf, axis=1) if tail == "upper" else 1-np.prod(1-original_cdf, axis=1)
    np.testing.assert_allclose(aggregate.probability_draws(threshold)[:, 0], expected)
    np.testing.assert_allclose(aggregate.observations[:, 0], getattr(np, reduction)(f.observations, axis=1))
    row = aggregate.risk_summary(threshold).iloc[0]
    assert row['mean'] == pytest.approx(np.mean(expected))
    # Observation probabilities multiply WITHIN one parameter/state draw.
    wrong = 1-np.prod(original_cdf.mean(axis=0)) if tail == "upper" else 1-np.prod(1-original_cdf.mean(axis=0))
    assert abs(np.mean(expected)-wrong) > 1e-5


def test_joint_and_separate_marginal_aggregation_agree():
    f = forecast()
    joint = replace(f, observations=np.stack([f.observations, f.observations+1], -1),
        eta=np.stack([f.eta, f.eta+1], -1), channel_names=("a", "b"),
        tail=("upper", "upper"), transform_sign=np.ones(2), family="multiseries",
        observation_model={"a":bx.Gaussian(), "b":bx.Gaussian()},
        parameters={f"{key}.{channel}":value for key, value in f.parameters.items() for channel in ("a", "b")})
    np.testing.assert_allclose(joint.aggregate(channel="a").observations, f.aggregate().observations)
    np.testing.assert_allclose(joint.aggregate(channel="a").probability_draws(2), f.aggregate().probability_draws(2))
    np.testing.assert_allclose(joint.aggregate(channel="b").observations, f.aggregate().observations+1)
    with pytest.raises(ValueError, match="Choose channel"):
        joint.aggregate()


def test_mean_risk_is_distinct_from_at_least_one_exceedance():
    f = forecast()
    f.eta[:] = 0
    f.parameters['sigma_path'][:] = 1
    mean_risk = f.aggregate(weighting="equal").probability_draws(1)
    any_risk = f.aggregate(reduction="max").probability_draws(1)
    np.testing.assert_allclose(mean_risk, norm.sf(np.sqrt(12)))
    np.testing.assert_allclose(any_risk, 1-norm.cdf(1)**12)
    assert any_risk.min() > 100*mean_risk.max()


@pytest.mark.parametrize("dates", [np.arange(12), pd.date_range("2024-01-01", periods=12, freq="D"),
                                  pd.date_range("2024-01-01", periods=12, freq="2MS")])
def test_nonmonthly_calendar_inputs_are_rejected(dates):
    with pytest.raises(ValueError, match="monthly"):
        replace(forecast(), dates=np.asarray(dates)).aggregate()


def test_custom_months_and_risk_direction_validation():
    f = forecast()
    for months in ((1, 1), (0, 1), (12, 2), (1.5,), ()):
        with pytest.raises(ValueError, match="months"):
            f.aggregate(months=months)
    with pytest.raises(ValueError, match="direction"):
        f.probability_draws(2, direction="any")
    with pytest.raises(ValueError, match="level"):
        f.aggregate().summary(1)


def test_month_risk_summaries_integrate_probabilities_not_simulated_events():
    f = forecast(periods=24)
    table = f.risk_summary(2, phase=7)
    assert pd.DatetimeIndex(table.time).month.tolist() == [7, 7]
    indices = [6, 18]
    expected = norm.sf((2-f.eta[:, indices])/f.parameters['sigma_path'][:, indices]).mean(axis=0)
    np.testing.assert_allclose(table['mean'], expected)
    assert not np.array_equal(table['mean'], f.tail_probability(2)[indices])
