from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bucex as bx


ROOT = Path(__file__).resolve().parents[1]


def _fit(*, dated: bool):
    model = bx.Model(
        bx.Gaussian(),
        (
            bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
            bx.DummySeasonal(period=4, mode="dynamic"),
        ),
    )
    simulation = bx.simulate(
        model,
        12,
        {
            "sigma": 0.25,
            "sd.level": 0.03,
            "sd.slope": 0.002,
            "sd.seasonal": 0.02,
        },
        initial_state=[2.0, 0.01, -0.2, 0.0, 0.2],
        seed=1521,
    )
    values = (
        pd.Series(
            simulation.y,
            index=pd.date_range("2000-01-01", periods=12, freq="QS"),
        )
        if dated
        else simulation.y
    )
    return bx.fit(
        values,
        model=model,
        priors="normal",
        engine="ffbs",
        parameterization="centered",
        mcmc=bx.MCMC(draws=4, warmup=2, chains=1, seed=1522),
    )


def test_dated_patterns_select_complete_calendar_years():
    fit = _fit(dated=True)
    figure, axis = fit.plot(
        "seasonal_patterns",
        years=[2000, 2002],
        labels=("winter", "spring", "summer", "autumn"),
        show_interval=False,
    )
    seasonal_name = next(
        name for name in fit.state_names if name.startswith("seasonal[")
    )
    median = np.median(fit.state_original(seasonal_name), axis=0)
    line_2000 = next(line for line in axis.lines if line.get_label() == "2000")
    line_2002 = next(line for line in axis.lines if line.get_label() == "2002")
    np.testing.assert_allclose(line_2000.get_ydata(), median[:4])
    np.testing.assert_allclose(line_2002.get_ydata(), median[8:12])
    assert [tick.get_text() for tick in axis.get_xticklabels()] == [
        "winter",
        "spring",
        "summer",
        "autumn",
    ]
    assert {line.get_label() for line in axis.lines} >= {"2000", "2002"}
    assert axis.get_xlabel() == "month"
    plt.close(figure)


def test_undated_patterns_accept_named_cycles_and_simulation_truth():
    fit = _fit(dated=False)
    truth = np.arange(fit.n_time, dtype=float)
    figure, axis = bx.plot_seasonal_patterns(
        fit,
        cycles=["first", "last"],
        show_interval=True,
        truth=truth,
    )
    labels = {line.get_label() for line in axis.lines}
    assert {
        "cycle 1 posterior",
        "cycle 1 truth",
        "cycle 3 posterior",
        "cycle 3 truth",
    } <= labels
    last_truth = next(
        line for line in axis.lines if line.get_label() == "cycle 3 truth"
    )
    np.testing.assert_allclose(last_truth.get_ydata(), truth[-4:])
    assert axis.get_xlabel() == "seasonal phase"
    plt.close(figure)


def test_pattern_selector_rejects_wrong_time_scale():
    dated = _fit(dated=True)
    undated = _fit(dated=False)
    for fit, kwargs, text in (
        (dated, {"cycles": [1]}, "years"),
        (undated, {"years": [2000]}, "cycles"),
        (dated, {"years": [1999]}, "absent or incomplete"),
    ):
        try:
            bx.plot_seasonal_patterns(fit, **kwargs)
        except ValueError as error:
            assert text in str(error)
        else:
            raise AssertionError("An invalid seasonal-pattern selector was accepted.")


def test_relevant_jsons_expose_seasonal_pattern_selection():
    paths = [
        ROOT / "examples" / "config" / "simulation.json",
        *sorted((ROOT / "examples" / "config" / "simulations").glob("*.json")),
        ROOT / "examples" / "config" / "uccle.json",
        *sorted((ROOT / "examples" / "config" / "uccle").glob("*.json")),
        *sorted((ROOT / "examples" / "config" / "uccle_gaussian").glob("*.json")),
        *sorted((ROOT / "examples" / "config" / "phi").glob("simulation_*.json")),
        *sorted((ROOT / "examples" / "config" / "phi" / "uccle").glob("*.json")),
    ]
    for path in paths:
        settings = bx.load_config(path)["figures"]["seasonal_patterns"]
        assert set(settings) >= {"years", "cycles", "show_interval"}
        if path.name == "uccle.json" or any(
            part in {"uccle", "uccle_gaussian"} for part in path.parts
        ):
            assert settings["years"] == [1892, 2022]
            assert settings["cycles"] == []
        else:
            assert settings["years"] == []
            assert settings["cycles"] == ["first", "last"]


def test_all_fitting_examples_write_the_new_plot():
    scripts = (
        "03_simulation_laplace.py",
        "04_simulation_pgas.py",
        "05_uccle_laplace.py",
        "06_uccle_pgas.py",
        "08_simulation_laplace_mh.py",
        "09_uccle_laplace_mh.py",
        "10_simulation_phi.py",
        "11_uccle_phi.py",
        "12_uccle_gaussian.py",
    )
    for script in scripts:
        source = (ROOT / "examples" / script).read_text(encoding="utf-8")
        assert '"seasonal_patterns"' in source
        assert "seasonal_patterns.{extension}" in source


def test_quick_replot_script_is_generic_and_names_the_two_requested_runs():
    source = (ROOT / "examples" / "replot_seasonal_patterns.py").read_text(
        encoding="utf-8"
    )
    assert "combined.bucex" in source
    assert "20260825_222926_03-tnx_27536126_combined" in source
    assert "20260825_222924_01-txx_27536124_combined" in source
