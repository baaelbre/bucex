"""Keep the archived reference distinct from the current calibrated wide fit."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import bucex as bx
from research.seasonal.prepare import exploratory_structure


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "research" / "seasonal" / "config"


def test_archived_reference_and_new_final_config_have_distinct_priors():
    reference = bx.load_config(CONFIG / "reference_20260923.json")
    final = bx.load_config(CONFIG / "final.json")
    assert reference["reference_run"]["archive"] == "uccle_copula_20260923T222238_406159Z.zip"
    assert reference["reference_run"]["archive_sha256"] == (
        "2e337e0fbe09f8587ab1782e10c861507b35b73677163b7904e7d1e16d1ac64a"
    )
    assert reference["analysis"] == "copula"
    assert reference["data"]["frequency"] == "seasonal"
    assert reference["model"]["period"] == 4
    assert reference["copula"] == {"structure": "seasons", "eta": 1.0, "prior_sd": 0.25}
    assert reference["priors"]["innovation_median"] == {
        "level": 0.014451332204168785,
        "trend": 0.00010883964876422239,
        "season": 0.008343480538225555,
    }
    assert reference["priors"]["initial_slope_sd"] == 0.0046387735335118195
    assert reference["priors"]["initial_slope_median"] is None
    assert reference["mcmc"]["warmup"] == 2000 and reference["mcmc"]["draws"] == 4000
    assert final['priors']['innovation_sd']==pytest.approx({'level':.01,'trend':.0001,'season':.01})
    assert final['priors'].get('independent_shrinkage') is None
    assert final['priors']['shared_shrinkage']['scale_parameterization']=='normal_sd'
    assert not final['priors']['shared_shrinkage']['pool_initial_slope']
    assert final['priors']['initial_slope_sd']==pytest.approx(.01)
    assert final["priors"]["seasonal_initial_sd"] == 20
    assert final["forecast_horizon"] == 120
    assert final["forecast_draws"] == 12000
    assert final["mcmc"]["warmup"] == 6000 and final["mcmc"]["draws"] == 20000
    assert final["mcmc"]["chains"] == final["mcmc"]["chain_workers"] == 2
    assert "reference_run" not in final


def test_pre2019_config_is_genuinely_prospective():
    config = bx.load_config(CONFIG / "pre2019.json")
    assert config["data"]["end"] == "2019-05"
    assert config["forecast_horizon"] == 1
    assert config["additional_risks"]["TXx"] == [36.6, 39.7]
    assert config["contrasts"]["comparison"] == ["1989-03", "2019-02"]


def test_exploratory_seasonal_figure_has_six_records_and_one_overlay():
    dates = pd.date_range("1990-03-01", periods=16, freq="3MS")
    phase = np.arange(len(dates)) % 4
    data = pd.DataFrame({name: i + 3 * np.sin(phase * np.pi / 2) + .1 * np.arange(len(dates))
                         for i, name in enumerate(("TXm", "TNm", "TXx", "TXn", "TNx", "TNn"))},
                        index=dates)
    figure, cycles, smooths = exploratory_structure(
        data, reference=("1990-03", "1992-02"), comparison=("1992-03", "1994-02")
    )
    assert len(figure.axes) == 7
    assert set(cycles.season) == {"DJF", "MAM", "JJA", "SON"}
    assert cycles.groupby(["series", "period"]).size().eq(4).all()
    assert len(smooths) == len(data) * len(data.columns)
    assert np.isfinite(smooths[["anomaly", "smooth"]]).all().all()
    plt.close(figure)
