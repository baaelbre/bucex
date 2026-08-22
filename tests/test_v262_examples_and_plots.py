from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np

import bucex as bx


def test_v101_version_and_workflow_removal():
    assert bx.__version__ == "1.0.1"
    assert not hasattr(bx, "make_structural_scenarios")
    assert not hasattr(bx, "PresentationWorkflow")


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


def test_matched_gev_models_keep_the_same_latent_path():
    models = [
        bx.Model(bx.GEV(), (bx.LocalLevel(mode="dynamic"),), name=f"xi={xi}")
        for xi in (-0.3, 0.0, 0.3)
    ]
    simulations = [
        bx.simulate(
            model,
            48,
            {"sigma": 1.5, "xi": xi, "sd.level": 0.08},
            initial_state=[25.0],
            seed=2601,
        )
        for model, xi in zip(models, (-0.3, 0.0, 0.3))
    ]
    for simulation in simulations[1:]:
        np.testing.assert_allclose(simulation.eta, simulations[0].eta)


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
    assert axis.get_title() == "Posterior seasonality"
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
    assert "true slope" in {line.get_label() for line in slope_axis.lines}
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
    assert predictive_axis.get_title() == "Posterior predictive check"

    forecast = fit.forecast(4, draws=4, seed=1012)
    np.testing.assert_array_equal(forecast.dates, np.arange(20, 24))
    forecast_axis = forecast.plot(
        history=fit.observed,
        history_dates=np.arange(fit.n_time),
        history_points=5,
    )
    assert forecast_axis.get_title() == "Posterior predictive forecast"
    plt.close(predictive_axis.figure)
    plt.close(forecast_axis.figure)


def test_slope_can_condition_on_dynamic_draws_and_overlay_fixed_uncertainty():
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
    assert "dynamic slope median" in labels
    assert "fixed slope median" in labels
    assert axis.get_title() == "Posterior slope"
    assert axis.get_ylabel() == "slope / unit per decade"
    plt.close(figure)


def test_laplace_fit_is_a_full_path_warm_start_for_pgas():
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
    pgas = bx.fit(
        simulation.y,
        model=model,
        priors=laplace.priors,
        engine="pgas",
        parameterization="fruehwirth_schnatter",
        mcmc=bx.MCMC(draws=1, warmup=1, chains=1, seed=266),
        particles=bx.Particles(n=16, proposal="guided"),
        init=laplace,
    )
    warm = laplace.warm_start()
    assert warm["__centered_path__"].shape == (laplace.n_time + 1, 5)
    assert warm["__warm_start__"]["source_engine"] == "laplace"
    assert pgas.meta["warm_start_source_engine"] == "laplace"
    assert not laplace.plan.targets_exact_posterior
    assert pgas.plan.targets_exact_posterior


def test_hpc_surface_matches_the_eight_examples():
    root = Path(__file__).resolve().parents[1]
    runner_directory = root / "bash_scripts"
    submit_directory = root / "job_scripts"
    runners = {path.name for path in runner_directory.glob("run_*.sh")}
    submissions = {path.name for path in submit_directory.glob("submit_*.pbs")}
    expected_stems = {
        "00_uccle_record",
        "01_tail_simulations",
        "02_structural_simulations",
        "03_simulation_laplace",
        "04_simulation_pgas",
        "05_uccle_laplace",
        "06_uccle_pgas",
        "07_centered_ig_random_walk_gev",
    }
    assert {f"run_{stem}.sh" for stem in expected_stems} <= runners
    assert {f"submit_{stem}.pbs" for stem in expected_stems} <= submissions
    assert "run_laplace_sensitivity_all.sh" in runners
    assert "submit_laplace_sensitivity_fits.pbs" in submissions
    assert "submit_laplace_sensitivity_combine.pbs" in submissions
    assert not (submit_directory / "common.sh").exists()
    assert not (submit_directory / "submit_all.sh").exists()
    for stem in expected_stems:
        runner = (runner_directory / f"run_{stem}.sh").read_text(encoding="utf-8")
        submission = (submit_directory / f"submit_{stem}.pbs").read_text(encoding="utf-8")
        assert "set -euo pipefail" in runner
        assert f"examples/{stem}.py" in runner
        assert "BUCEX_RESULTS_ROOT" in runner
        assert "BUCEX_RUN_ID" in runner
        assert "#PBS -N" in submission
        assert f"bash_scripts/run_{stem}.sh" in submission
