from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "examples" / "config" / "simulations"

CONFIGS = (
    ("01_stationary.json", "stationary"),
    ("02_linear_trend.json", "linear"),
    ("03_random_walk.json", "random_walk"),
    ("04_local_linear_trend.json", "llt"),
    ("05_changing_seasonality.json", "dynamic_season"),
    ("06_llt_fixed_seasonality.json", "llt_season"),
)

REQUIRED_SECTIONS = {
    "schema_version",
    "simulation",
    "priors",
    "mcmc",
    "inference",
    "figures",
    "runtime",
    "output",
}


@pytest.mark.parametrize(("filename", "scenario_key"), CONFIGS)
def test_each_simulation_config_is_complete_and_selects_one_truth(
    filename: str,
    scenario_key: str,
):
    config = json.loads((CONFIG_DIR / filename).read_text(encoding="utf-8"))

    assert set(config) == REQUIRED_SECTIONS
    assert config["schema_version"] == 1
    assert config["simulation"]["scenario_keys"] == [scenario_key]
    assert config["simulation"]["n_time"] == 1000
    assert config["simulation"]["period"] == 4
    assert config["output"] == {
        "results_root": "results",
        "run_id": None,
        "overwrite": False,
    }
    assert config["runtime"]["chain_only"] is False
    assert config["runtime"]["combine_runs"] == []
    assert config["inference"]["parameterization"] == "fruehwirth_schnatter"
    assert config["inference"]["asis"] is False
    assert config["inference"]["pgas_proposal"] == "guided"
    assert config["figures"]["interval_probability"] == pytest.approx(0.9)
    assert config["mcmc"] == {
        "draws": 1000,
        "warmup": 1000,
        "chains": 4,
        "seed": 13081997,
    }
    assert sum(config["priors"]["trend_probabilities"]) == pytest.approx(1.0)
    assert sum(config["priors"]["season_probabilities"]) == pytest.approx(1.0)


def test_stationary_and_linear_presets_retain_the_successful_joint_run():
    for filename in ("01_stationary.json", "02_linear_trend.json"):
        config = json.loads((CONFIG_DIR / filename).read_text(encoding="utf-8"))
        assert config["simulation"]["seed"] == 13041997
        assert config["priors"]["beta_sd"] == pytest.approx(0.006)
        assert config["priors"]["innovation_slab_sd"] == {
            "level": 0.03,
            "trend": 0.00015,
            "season": 0.05,
        }


def test_random_walk_preset_retain_dedicated_retry_settings():
    config = json.loads(
        (CONFIG_DIR / "03_random_walk.json").read_text(encoding="utf-8")
    )
    assert config["simulation"]["random_walk_sd"] == pytest.approx(0.02)
    assert config["priors"]["innovation_slab_sd"] == {
        "level": 0.02,
        "trend": 0.00015,
        "season": 0.03,
    }


@pytest.mark.parametrize(
    "filename",
    (
        "04_local_linear_trend.json",
        "05_changing_seasonality.json",
        "06_llt_fixed_seasonality.json",
    ),
)
def test_llt_and_seasonal_presets_retain_successful_baseline(filename: str):
    config = json.loads((CONFIG_DIR / filename).read_text(encoding="utf-8"))
    assert config["simulation"]["local_level_sd"] == pytest.approx(0.01)
    assert config["simulation"]["local_slope_sd"] == pytest.approx(0.0008)
    assert config["priors"]["innovation_slab_sd"] == {
        "level": 0.03,
        "trend": 0.003,
        "season": 0.05,
    }
