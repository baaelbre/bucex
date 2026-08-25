from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "examples" / "config" / "uccle"
CONFIGS = (
    ("01_txx.json", "TXx"),
    ("02_txn.json", "TXn"),
    ("03_tnx.json", "TNx"),
    ("04_tnn.json", "TNn"),
)


@pytest.mark.parametrize(("filename", "series"), CONFIGS)
def test_each_uccle_config_is_complete_calibrated_and_selects_one_series(
    filename: str,
    series: str,
):
    config = json.loads((CONFIG_DIR / filename).read_text(encoding="utf-8"))

    assert set(config) == {
        "schema_version",
        "data",
        "model",
        "priors",
        "mcmc",
        "inference",
        "figures",
        "runtime",
        "output",
    }
    assert config["data"] == {
        "data_dir": None,
        "start": "1892-01-01",
        "end": None,
        "series": [series],
        "period": 12,
    }
    assert config["model"] == {
        "observation_family": "gev",
        "time_varying_parameter": "location",
        "trend_component": "local_linear_trend",
        "level_mode": "dynamic",
        "trend_mode": "dynamic",
        "seasonal_component": "dummy",
        "seasonal_mode": "dynamic",
    }
    assert config["priors"]["alpha_mean"] == "transformed_series_median"
    assert config["priors"]["alpha_sd"] == pytest.approx(3.2)
    assert config["priors"]["beta_mean"] == pytest.approx(0.0)
    assert config["priors"]["beta_sd"] == pytest.approx(0.0025)
    assert config["priors"]["seasonal_initial_sd"] == pytest.approx(2.25)
    assert config["priors"]["sigma2"] == {"a": 2.0, "b": 2.0}
    assert config["priors"]["xi_bounds"] == [-0.5, 0.5]
    assert config["priors"]["innovation_slab_sd"] == {
        "level": 0.02,
        "trend": 0.00005,
        "season": 0.02,
    }
    assert config["priors"]["level_dynamic_probability"] == pytest.approx(0.5)
    assert config["priors"]["trend_probabilities"] == [0.2, 0.4, 0.4]
    assert config["priors"]["season_probabilities"] == [0.0, 0.5, 0.5]
    assert config["mcmc"] == {
        "draws": 1000,
        "warmup": 1000,
        "chains": 4,
        "seed": 56000,
    }
    assert config["inference"]["parameterization"] == "fruehwirth_schnatter"
    assert config["inference"]["asis"] is False
    assert config["inference"]["laplace_mh_steps"] == 1
    assert config["figures"]["predictive_draws"] == 500
    assert config["figures"]["focus_month"] == 7
    assert config["figures"]["forecast_history"] == 360
    assert config["figures"]["forecast_horizon"] == 120
    assert config["runtime"] == {
        "progress": True,
        "chain_only": False,
        "combine_runs": [],
    }
    assert config["output"] == {
        "results_root": "results",
        "run_id": None,
        "overwrite": False,
    }


def test_uccle_presets_differ_only_by_selected_series():
    configs = [
        json.loads((CONFIG_DIR / filename).read_text(encoding="utf-8"))
        for filename, _ in CONFIGS
    ]
    for config in configs:
        config = deepcopy(config)
        config["data"]["series"] = ["SERIES"]
        assert config == {
            **deepcopy(configs[0]),
            "data": {**configs[0]["data"], "series": ["SERIES"]},
        }


def test_all_uccle_fitting_examples_read_the_model_and_prior_from_json():
    for number in ("05", "06", "09"):
        path = next((ROOT / "examples").glob(f"{number}_uccle_*.py"))
        source = path.read_text(encoding="utf-8")
        assert 'MODEL_SETTINGS = CONFIG["model"]' in source
        assert 'ALPHA_PRIOR_MEAN = PRIOR_SETTINGS["alpha_mean"]' in source
        assert 'MODEL_SETTINGS["level_mode"]' in source
        assert 'MODEL_SETTINGS["trend_mode"]' in source
        assert 'MODEL_SETTINGS["seasonal_mode"]' in source
