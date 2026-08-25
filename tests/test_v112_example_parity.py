from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_example(number: str, filename: str, monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("BUCEX_"):
            monkeypatch.delenv(key, raising=False)
    path = ROOT / "examples" / filename
    spec = importlib.util.spec_from_file_location(f"bucex_example_{number}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scenario_contract(module):
    result = []
    for scenario in module.SCENARIOS:
        result.append(
            {
                "name": scenario["name"],
                "key": scenario["key"],
                "model": scenario["model"].to_dict(),
                "params": scenario["params"],
                "initial_state": np.asarray(scenario["initial_state"]).tolist(),
                "seed": scenario["seed"],
                "structural_truth": scenario["structural_truth"],
            }
        )
    return result


def _runner_text(filename: str) -> str:
    return (ROOT / "bash_scripts" / filename).read_text(encoding="utf-8")


def test_simulation_laplace_mh_is_method_matched_to_laplace(monkeypatch):
    simulation = _load_example("02", "02_structural_simulations.py", monkeypatch)
    laplace = _load_example("03", "03_simulation_laplace.py", monkeypatch)
    pgas = _load_example("04", "04_simulation_pgas.py", monkeypatch)
    laplace_mh = _load_example("08", "08_simulation_laplace_mh.py", monkeypatch)

    simulation_settings = (
        "N_TIME",
        "PERIOD",
        "SIGMA",
        "XI",
        "INITIAL_LEVEL",
        "LINEAR_SLOPE",
        "RANDOM_WALK_SD",
        "LOCAL_LEVEL_SD",
        "LOCAL_SLOPE_SD",
        "LOCAL_INITIAL_SLOPE",
        "DYNAMIC_SEASON_AMPLITUDE",
        "FIXED_SEASON_AMPLITUDE",
        "SEASONAL_SD",
        "SIMULATION_SEED",
    )
    for setting in simulation_settings:
        expected = getattr(simulation, setting)
        assert getattr(laplace, setting) == expected
        assert getattr(pgas, setting) == expected
        assert getattr(laplace_mh, setting) == expected
    assert _scenario_contract(simulation) == _scenario_contract(laplace)
    assert _scenario_contract(pgas) == _scenario_contract(laplace)
    assert _scenario_contract(laplace_mh) == _scenario_contract(laplace)

    fit_settings = (
        "DRAWS",
        "WARMUP",
        "CHAINS",
        "SEED",
        "FIGURE_FORMATS",
        "PREDICTIVE_DRAWS",
        "FORECAST_HORIZON",
        "FORECAST_HISTORY",
        "FOCUS_PHASE",
    )
    for setting in fit_settings:
        assert getattr(pgas, setting) == getattr(laplace, setting)
        assert getattr(laplace_mh, setting) == getattr(laplace, setting)
    assert pgas.FIT_MODEL.to_dict() == laplace.FIT_MODEL.to_dict()
    assert laplace_mh.FIT_MODEL.to_dict() == laplace.FIT_MODEL.to_dict()
    assert pgas.PRIOR_SETTINGS == laplace.PRIOR_SETTINGS
    assert laplace_mh.PRIOR_SETTINGS == laplace.PRIOR_SETTINGS
    assert len(laplace_mh.SCENARIOS) == 6

    source = (ROOT / "examples" / "08_simulation_laplace_mh.py").read_text(
        encoding="utf-8"
    )
    for directory in ("simulations", "fits", "tables", "figures"):
        assert f'OUTPUT_DIR / "{directory}"' in source
    assert 'engine="laplace_mh"' in source
    assert "bx.Laplace(mh_steps=MH_STEPS)" in source


def test_uccle_laplace_mh_is_method_matched_to_laplace(monkeypatch):
    laplace = _load_example("05", "05_uccle_laplace.py", monkeypatch)
    pgas = _load_example("06", "06_uccle_pgas.py", monkeypatch)
    laplace_mh = _load_example("09", "09_uccle_laplace_mh.py", monkeypatch)

    settings = (
        "START",
        "END",
        "SERIES",
        "PERIOD",
        "DRAWS",
        "WARMUP",
        "CHAINS",
        "SEED",
        "FIGURE_FORMATS",
        "PREDICTIVE_DRAWS",
        "FORECAST_HORIZON",
        "FORECAST_HISTORY",
        "FOCUS_MONTH",
        "MONTH_LABELS",
    )
    for setting in settings:
        assert getattr(pgas, setting) == getattr(laplace, setting)
        assert getattr(laplace_mh, setting) == getattr(laplace, setting)
    assert pgas.MODEL.to_dict() == laplace.MODEL.to_dict()
    assert laplace_mh.MODEL.to_dict() == laplace.MODEL.to_dict()
    assert pgas.PRIOR_SETTINGS == laplace.PRIOR_SETTINGS
    assert laplace_mh.PRIOR_SETTINGS == laplace.PRIOR_SETTINGS


def test_scientific_settings_are_authoritative_json_files():
    config_dir = ROOT / "examples" / "config"
    simulation = json.loads((config_dir / "simulation.json").read_text())
    uccle = json.loads((config_dir / "uccle.json").read_text())
    centered = json.loads((config_dir / "centered_ig.json").read_text())
    assert simulation["schema_version"] == 1
    assert uccle["schema_version"] == 1
    assert centered["schema_version"] == 1
    assert simulation["priors"]["innovation_slab_sd"] == {
        "level": 0.1,
        "trend": 0.0008,
        "season": 0.07,
    }
    assert uccle["priors"]["innovation_slab_sd"] == {
        "level": 0.03,
        "trend": 0.0001,
        "season": 0.05,
    }

    simulation_scripts = (
        "02_structural_simulations.py",
        "03_simulation_laplace.py",
        "04_simulation_pgas.py",
        "08_simulation_laplace_mh.py",
    )
    uccle_scripts = (
        "05_uccle_laplace.py",
        "06_uccle_pgas.py",
        "09_uccle_laplace_mh.py",
    )
    for filename in simulation_scripts:
        source = (ROOT / "examples" / filename).read_text(encoding="utf-8")
        assert 'DEFAULT_CONFIG_FILE = EXAMPLE_ROOT / "config" / "simulation.json"' in source
        assert "CONFIG = bx.load_config(SETTINGS_PATH)" in source
    for filename in uccle_scripts:
        source = (ROOT / "examples" / filename).read_text(encoding="utf-8")
        assert 'DEFAULT_CONFIG_FILE = EXAMPLE_ROOT / "config" / "uccle.json"' in source
        assert "CONFIG = bx.load_config(SETTINGS_PATH)" in source
    centered_source = (ROOT / "examples" / "07_centered_ig.py").read_text()
    assert 'DEFAULT_CONFIG_FILE = EXAMPLE_ROOT / "config" / "centered_ig.json"' in centered_source
    assert "CONFIG = bx.load_config(SETTINGS_PATH)" in centered_source

    source = (ROOT / "examples" / "09_uccle_laplace_mh.py").read_text(
        encoding="utf-8"
    )
    for directory in ("fits", "tables", "figures"):
        assert f'OUTPUT_DIR / "{directory}"' in source
    assert 'engine="laplace_mh"' in source
    assert "bx.Laplace(mh_steps=MH_STEPS)" in source


def test_example_figure_titles_do_not_name_the_inference_engine():
    forbidden = ("(Laplace)", "(Laplace-MH)", "(Laplace_MH)", "(PGAS)")
    for path in sorted((ROOT / "examples").glob("[0-9][0-9]_*.py")):
        source = path.read_text(encoding="utf-8")
        assert not any(label in source for label in forbidden), path


def test_laplace_mh_runners_share_the_authoritative_json_files():
    simulation = _runner_text("run_03_simulation_laplace.sh")
    simulation_mh = _runner_text("run_08_simulation_laplace_mh.sh")
    uccle = _runner_text("run_05_uccle_laplace.sh")
    uccle_mh = _runner_text("run_09_uccle_laplace_mh.sh")

    assert "examples/config/simulation.json" in simulation
    assert "examples/config/simulation.json" in simulation_mh
    assert "examples/config/uccle.json" in uccle
    assert "examples/config/uccle.json" in uccle_mh
    assert "BUCEX_" not in simulation + simulation_mh + uccle + uccle_mh


def test_no_temporary_simulation_debug_overrides_remain():
    source = (ROOT / "examples" / "03_simulation_laplace.py").read_text(
        encoding="utf-8"
    )
    assert "all_great_except_for_RW" not in source
    assert "only the random walk" not in source
    assert source.count("SCENARIOS =") == 1
    assert source.count("OUTPUT_DIR =") == 1
