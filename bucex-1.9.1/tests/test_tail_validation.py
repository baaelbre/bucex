"""Directional extremes must not be inferred from central coverage alone."""
import numpy as np
import pandas as pd
import pytest
import bucex as bx

from research.monthly.tail_validation import summarize_tail_validation


def test_central_misses_and_directional_events_use_correct_sides():
    cases = []
    observations = (("2020-12-01", -3., 12), ("2021-03-01", 3., 16))
    for horizon, (time, observed, origin) in enumerate(observations, 1):
        for nominal in (.90, .95, .99):
            cases.append(dict(time=time, channel="TXx", horizon=horizon, origin=origin,
                              kind="central_interval", nominal=nominal, observed=observed,
                              lower=-2., upper=2., width=4.))
        for nominal, quantile in ((.01, -2.), (.05, -2.), (.95, 2.), (.99, 2.)):
            cases.append(dict(time=time, channel="TXx", horizon=horizon, origin=origin,
                              kind="cdf_quantile", nominal=nominal,
                              observed=observed, quantile=quantile))
    risks = pd.DataFrame([
        dict(time="2020-12-01", channel="TXx", horizon=1, origin=12,
             observed_event=False, event_probability=.2, threshold=2.,
             direction=">", brier=.04, log_score=-np.log(.8)),
        dict(time="2021-03-01", channel="TXx", horizon=2, origin=16,
             observed_event=True, event_probability=.6, threshold=2.,
             direction=">", brier=.16, log_score=-np.log(.6))])
    summary = summarize_tail_validation(pd.DataFrame(cases), risks, steps_per_year=4)
    central = summary["central_coverage.csv"]
    pooled = central.loc[(central.stratum == "overall") & (central.nominal == .99)].iloc[0]
    assert (pooled.n_cases, pooled.n_dates, pooled.n_origins) == (2, 2, 2)
    assert (pooled.below, pooled.inside, pooled.above) == (1, 0, 1)
    assert pooled.expected_misses == pytest.approx(.02)
    assert set(central.loc[central.stratum == "season", "group"]) == {"DJF", "MAM"}

    tails = summary["directional_tails.csv"]
    overall = tails[tails.stratum == "overall"].set_index("nominal")
    assert overall.loc[.01, "observed_events"] == 1
    assert overall.loc[.99, "observed_events"] == 1
    assert overall.loc[.99, "expected_events"] == pytest.approx(.02)
    assert overall.loc[.95, "expected_events"] == pytest.approx(.10)

    thresholds = summary["threshold_events.csv"]
    risk = thresholds[thresholds.stratum == "overall"].iloc[0]
    assert risk.observed_events == 1
    assert risk.expected_events == pytest.approx(.8)
    assert risk.mean_brier == pytest.approx(.1)
    assert risk.n_origins == 2


def test_invalid_threshold_probability_is_not_reported_as_calibration():
    case = pd.DataFrame([dict(time="2020-12-01", channel="TXx", horizon=1,
                              origin=12, kind="central_interval", nominal=.99,
                              observed=1., lower=0., upper=2., width=2.),
                         *[dict(time="2020-12-01", channel="TXx", horizon=1,
                                origin=12, kind="cdf_quantile", nominal=q,
                                observed=1., quantile=0.) for q in (.01, .05, .95, .99)]])
    risk = pd.DataFrame([dict(time="2020-12-01", channel="TXx", horizon=1,
                              origin=12, observed_event=True, event_probability=1.1,
                              threshold=0., direction=">", brier=0., log_score=0.)])
    with pytest.raises(ValueError, match="Threshold probabilities"):
        summarize_tail_validation(case, risk, steps_per_year=4)


def test_independent_validation_writes_matching_analytic_event_scores(tmp_path, monkeypatch):
    from research.monthly.validate import validate

    config = bx.load_config("research/seasonal/config/smoke.json")
    config["analysis"] = "independent"
    config["priors"]["shared_shrinkage"] = None
    config["data"]["series"] = ["TXm"]
    config["mcmc"].update(chains=2, chain_workers=1, warmup=1, draws=4, progress=False)
    config["validation"] = dict(training_ends=["2023-11"], horizon=4, draws=8,
                                save_fits=False)
    dates = pd.date_range("2021-03-01", periods=22, freq="3MS")
    values = pd.DataFrame({"TXm": 10 + np.sin(np.arange(22))}, index=dates)
    values.attrs["frequency"] = "seasonal"
    monkeypatch.setattr(bx, "load_uccle_multiseries", lambda **kwargs: values.copy())
    result = validate(config, directory=tmp_path / "check") / "TXm"
    cases = pd.read_csv(result / "threshold_cases.csv")
    scores = pd.read_csv(result / "scores.csv")
    assert (result / "directional_tails.csv").exists()
    merged = scores[scores.score.eq("exceedance_log")].merge(
        cases, on=["origin", "channel", "horizon", "time"], validate="one_to_one")
    np.testing.assert_allclose(merged.value, merged.log_score)
