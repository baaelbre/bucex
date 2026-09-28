"""Calendar labels and risk aggregation are independent of seasonal model choice."""
import numpy as np
import pandas as pd
import pytest

import bucex as bx


def _monthly_fit(start, periods):
    dates = pd.date_range(start, periods=periods, freq="MS")
    y = pd.Series(np.random.default_rng(16100).normal(size=periods), index=dates)
    model = bx.Model(bx.Gaussian(), (bx.LocalLevel(),))
    return bx.fit(y, model, priors="normal", parameterization="centered",
                  mcmc=bx.MCMC(draws=5, warmup=2, chains=1, seed=16101))


def test_annual_risk_uses_only_complete_calendar_years_without_seasonality():
    fit = _monthly_fit("2000-03-01", 24)
    monthly = fit.exceedance_probability_draws(.5, return_labels=False)
    yearly, labels = fit.exceedance_probability_draws(.5, annual=True)
    np.testing.assert_array_equal(labels, [2001])
    mask = pd.DatetimeIndex(fit.dates).year == 2001
    np.testing.assert_allclose(yearly[:, 0], 1 - np.prod(1-monthly[:, mask], axis=1))
    _, partial_labels = fit.exceedance_probability_draws(.5, annual=True, include_partial=True)
    np.testing.assert_array_equal(partial_labels, [2000, 2001, 2002])
    periods, period_labels = fit.return_period_draws(.5, annual=True)
    np.testing.assert_array_equal(period_labels, labels)
    np.testing.assert_allclose(periods, 1/yearly)


def test_annual_risk_requires_explicit_partial_year_opt_in():
    fit = _monthly_fit("2000-03-01", 5)
    with pytest.raises(ValueError, match="[Cc]omplete"):
        fit.exceedance_probability_draws(.5, annual=True)
    partial, labels = fit.exceedance_probability_draws(.5, annual=True, include_partial=True)
    assert partial.shape == (5, 1)
    np.testing.assert_array_equal(labels, [2000])
