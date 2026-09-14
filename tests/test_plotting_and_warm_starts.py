from __future__ import annotations


import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
from scipy.stats import genextreme

import bucex as bx


def test_loess_smooth_is_robust_and_preserves_missing_positions():
    x = np.linspace(0.0, 10.0, 81)
    truth = 2.0 + 0.5 * x
    y = truth.copy()
    y[40] += 30.0
    y[7] = np.nan
    smooth = bx.loess_smooth(x, y, fraction=0.25, robust_iterations=2)
    assert np.isnan(smooth[7])
    assert np.nanmedian(np.abs(smooth - truth)) < 0.05
    assert abs(smooth[40] - truth[40]) < 0.5


def test_tail_comparisons_are_stationary_and_probability_matched():
    models = [
        bx.Model(bx.GEV(), (bx.LocalLevel(mode="static"),), name=f"xi={xi}")
        for xi in (-0.3, 0.0, 0.3)
    ]
    simulations = [
        bx.simulate(
            model,
            48,
            {"sigma": 1.5, "xi": xi},
            initial_state=[25.0],
            seed=2601,
        )
        for model, xi in zip(models, (-0.3, 0.0, 0.3))
    ]
    for simulation in simulations:
        np.testing.assert_allclose(simulation.eta, 25.0)
        np.testing.assert_allclose(simulation.states[:, 0], 25.0)
        assert simulation.model.noise_names == ()

    probabilities = [
        genextreme.cdf(simulation.y, c=-xi, loc=25.0, scale=1.5)
        for simulation, xi in zip(simulations, (-0.3, 0.0, 0.3))
    ]
    for probability in probabilities[1:]:
        np.testing.assert_allclose(probability, probabilities[0], atol=1e-12)


def test_phase_specific_season_plot_uses_only_the_seasonal_effect():
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
        {"sigma": 0.25, "sd.level": 0.03, "sd.slope": 0.002, "sd.seasonal": 0.02},
        initial_state=[2.0, 0.01, -0.2, 0.0, 0.2],
        seed=262,
    )
    fit = bx.fit(
        simulation.y,
        model=model,
        priors="normal",
        engine="ffbs",
        parameterization="centered",
        mcmc=bx.MCMC(draws=2, warmup=1, chains=1, seed=263),
    )
    figure, axis = fit.plot(
        "season",
        labels=("one", "two", "three", "four"),
        show_interval=False,
    )
    assert len(axis.lines) == 4
    assert [line.get_label() for line in axis.lines] == ["one", "two", "three", "four"]
    assert all(len(line.get_xdata()) == 6 for line in axis.lines)
    seasonal_name = next(name for name in fit.state_names if name.startswith("seasonal["))
    seasonal_median = np.median(fit.state_original(seasonal_name), axis=0)
    np.testing.assert_allclose(axis.lines[0].get_ydata(), seasonal_median[0::4])
    assert axis.get_title() == ""
    assert axis.get_ylabel() == "seasonal effect"
    plt.close(figure)


def test_level_and_slope_have_separate_scientific_scales():
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
        {"sigma": 0.25, "sd.level": 0.03, "sd.slope": 0.002, "sd.seasonal": 0.02},
        initial_state=[2.0, 0.01, -0.2, 0.0, 0.2],
        seed=1001,
    )
    fit = bx.fit(
        simulation.y,
        model=model,
        priors="normal",
        engine="ffbs",
        parameterization="centered",
        mcmc=bx.MCMC(draws=2, warmup=1, chains=1, seed=1002),
    )

    level_figure, level_axis = fit.plot("level")
    clean_level_figure, clean_level_axis = fit.plot("level", show_observed=False)
    slope_figure, slope_axis = fit.plot("slope", truth=np.zeros(24))
    assert any(isinstance(item, PathCollection) for item in level_axis.collections)
    assert not any(
        isinstance(item, PathCollection) for item in clean_level_axis.collections
    )
    assert not any(isinstance(item, PathCollection) for item in slope_axis.collections)
    assert "slope per observation interval" == slope_axis.get_ylabel()
    assert r"$\beta_t$" in {line.get_label() for line in slope_axis.lines}
    plt.close(level_figure)
    plt.close(clean_level_figure)
    plt.close(slope_figure)


def test_posterior_predictive_and_forecast_share_the_public_plot_api():
    values = np.linspace(0.0, 1.0, 20)
    fit = bx.fit(
        values,
        family="gaussian",
        trend="local_level",
        priors="normal",
        mcmc=bx.MCMC(draws=3, warmup=1, chains=1, seed=1010),
    )
    predictive = fit.posterior_predictive(draws=4, seed=1011)
    assert predictive.observations.shape == (4, 20)
    predictive_axis = predictive.plot(observed=fit.observed)
    assert predictive_axis.get_title() == ""

    forecast = fit.forecast(4, draws=4, seed=1012)
    np.testing.assert_array_equal(forecast.dates, np.arange(20, 24))
    forecast_axis = forecast.plot(
        history=fit.observed,
        history_dates=np.arange(fit.n_time),
        history_points=5,
    )
    assert forecast_axis.get_title() == ""
    plt.close(predictive_axis.figure)
    plt.close(forecast_axis.figure)


def test_slope_can_condition_on_dynamic_draws_and_overlay_fixed_median():
    fit = bx.fit(
        np.linspace(0.0, 1.0, 24),
        family="gaussian",
        period=4,
        priors="ssvs",
        parameterization="fruehwirth_schnatter",
        mcmc=bx.MCMC(draws=4, warmup=2, chains=1, seed=1013),
    )
    fit.parameter_draws["state_trend"][0] = np.array([1, 2, 1, 2])
    figure, axis = fit.plot(
        "slope",
        scale="decade",
        unit="slope / unit per decade",
        condition_on="dynamic",
        show_fixed=True,
    )
    labels = {line.get_label() for line in axis.lines}
    assert r"$\hat{\beta}_t$ (dynamic)" in labels
    assert r"$\hat{\beta}_0$" in labels
    assert axis.get_title() == ""
    assert axis.get_ylabel() == "slope / unit per decade"
    assert not any(collection.get_label().startswith("fixed") for collection in axis.collections)
    plt.close(figure)


def test_laplace_fit_is_a_full_path_warm_start_for_laplace_mh():
    model = bx.Model(
        bx.GEV(),
        (
            bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
            bx.DummySeasonal(period=4, mode="dynamic"),
        ),
    )
    simulation = bx.simulate(
        model,
        24,
        {"sigma": 1.0, "xi": -0.2, "sd.level": 0.04, "sd.slope": 0.001, "sd.seasonal": 0.03},
        initial_state=[20.0, 0.0, -0.2, 0.0, 0.2],
        seed=264,
    )
    priors = bx.ssvs_gev_priors(period=4, alpha_mean=20.0)
    laplace = bx.fit(
        simulation.y,
        model=model,
        priors=priors,
        engine="laplace",
        parameterization="fruehwirth_schnatter",
        mcmc=bx.MCMC(draws=1, warmup=1, chains=1, seed=265),
    )
    exact = bx.fit(
        simulation.y,
        model=model,
        priors=laplace.priors,
        engine="laplace_mh",
        parameterization="fruehwirth_schnatter",
        mcmc=bx.MCMC(draws=1, warmup=1, chains=1, seed=266),
        init=laplace,
    )
    warm = laplace.warm_start()
    assert warm["__centered_path__"].shape == (laplace.n_time + 1, 5)
    assert warm["__warm_start__"]["source_engine"] == "laplace"
    assert exact.meta["warm_start_source_engine"] == "laplace"
    assert not laplace.plan.targets_exact_posterior
    assert exact.plan.targets_exact_posterior
