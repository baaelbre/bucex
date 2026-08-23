from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import bucex as bx


def _seasonal_fit():
    model = bx.Model(
        bx.Gaussian(),
        (
            bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
            bx.DummySeasonal(period=4, mode="dynamic"),
        ),
    )
    simulation = bx.simulate(
        model,
        24,
        {
            "sigma": 0.25,
            "sd.level": 0.03,
            "sd.slope": 0.002,
            "sd.seasonal": 0.02,
        },
        initial_state=[2.0, 0.01, -0.2, 0.0, 0.2],
        seed=1201,
    )
    return bx.fit(
        simulation.y,
        model=model,
        priors="normal",
        engine="ffbs",
        parameterization="centered",
        mcmc=bx.MCMC(draws=4, warmup=2, chains=1, seed=1202),
    )


def test_fitted_trajectory_and_level_accept_one_based_phase_selection():
    fit = _seasonal_fit()
    predictor_figure, predictor_axis = fit.plot("predictor", phase=2)
    level_figure, level_axis = fit.plot("level", phase=2)

    assert len(predictor_axis.lines[0].get_xdata()) == 6
    assert len(level_axis.lines[0].get_xdata()) == 6
    assert "phase 2" in predictor_axis.get_title()
    assert "phase 2" in level_axis.get_title()

    plt.close(predictor_figure)
    plt.close(level_figure)


def test_forecast_supports_phase_and_seasonally_adjusted_level_targets():
    fit = _seasonal_fit()
    forecast = fit.forecast(8, draws=4, seed=1203)

    np.testing.assert_array_equal(forecast.phases, np.array([1, 2, 3, 4, 1, 2, 3, 4]))
    phase_summary = forecast.summary(phase=2)
    assert phase_summary["phase"].tolist() == [2, 2]
    assert phase_summary["time"].tolist() == [25, 29]

    level_draws = forecast.component_draws("level")
    level_summary = forecast.summary(target="level")
    np.testing.assert_allclose(
        level_summary["median"].to_numpy(), np.median(level_draws, axis=0)
    )
    assert "eta_median" not in level_summary

    phase_axis = forecast.plot(
        phase=2,
        phase_label="second phase",
        history=fit.observed,
        history_dates=np.arange(fit.n_time),
        history_points=12,
    )
    level_axis = forecast.plot(
        target="level",
        history=np.median(fit.state_original("level"), axis=0),
        history_dates=np.arange(fit.n_time),
        history_points=12,
    )
    assert "second phase" in phase_axis.get_title()
    assert level_axis.get_title() == "Latent level forecast"
    assert "latent level median" in {line.get_label() for line in level_axis.lines}

    plt.close(phase_axis.figure)
    plt.close(level_axis.figure)


def test_phase_selection_rejects_nonseasonal_or_out_of_range_requests():
    fit = bx.fit(
        np.linspace(0.0, 1.0, 12),
        family="gaussian",
        trend="local_level",
        priors="normal",
        mcmc=bx.MCMC(draws=2, warmup=1, chains=1, seed=1204),
    )
    forecast = fit.forecast(3, draws=2, seed=1205)

    try:
        forecast.summary(phase=1)
    except ValueError as error:
        assert "seasonal period" in str(error)
    else:
        raise AssertionError("A nonseasonal forecast accepted phase=.")

