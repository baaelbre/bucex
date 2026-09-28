from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bucex as bx


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
