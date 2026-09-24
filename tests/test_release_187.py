"""Lock the final seasonal manuscript declaration for BUCEX 1.8.7."""
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bucex as bx
from research.seasonal.prepare import exploratory_structure


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "research" / "seasonal" / "config"


def test_reference_and_final_configs_lock_successful_run_settings():
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
    assert reference["mcmc"]["warmup"] == 2000 and reference["mcmc"]["draws"] == 4000
    assert final["mcmc"]["warmup"] == 3000 and final["mcmc"]["draws"] == 8000
    assert final["mcmc"]["chains"] == final["mcmc"]["chain_workers"] == 4

    # The final run changes runtime/output declarations, never the scientific
    # model that generated the successful reference archive.
    scientific_reference, scientific_final = deepcopy(reference), deepcopy(final)
    for item in (scientific_reference, scientific_final):
        item.pop("_comment", None)
        item.pop("mcmc", None)
        item.pop("output", None)
    assert scientific_final == scientific_reference


def test_pre2019_config_is_genuinely_prospective():
    config = bx.load_config(CONFIG / "pre2019.json")
    assert config["data"]["end"] == "2019-05"
    assert config["forecast_horizon"] == 1
    assert config["additional_risks"]["TXx"] == [39.7]
    assert config["contrasts"]["comparison"] == ["1989-03", "2019-02"]


def test_exploratory_seasonal_figure_has_paired_cycle_and_smooth_panels():
    dates = pd.date_range("1990-03-01", periods=16, freq="3MS")
    phase = np.arange(len(dates)) % 4
    data = pd.DataFrame({name: i + 3 * np.sin(phase * np.pi / 2) + .1 * np.arange(len(dates))
                         for i, name in enumerate(("TXm", "TNm", "TXx", "TXn", "TNx", "TNn"))},
                        index=dates)
    figure, cycles, smooths = exploratory_structure(
        data, reference=("1990-03", "1992-02"), comparison=("1992-03", "1994-02")
    )
    assert len(figure.axes) == 12
    assert set(cycles.season) == {"DJF", "MAM", "JJA", "SON"}
    assert cycles.groupby(["series", "period"]).size().eq(4).all()
    assert len(smooths) == len(data) * len(data.columns)
    assert np.isfinite(smooths[["anomaly", "smooth"]]).all().all()
    plt.close(figure)
