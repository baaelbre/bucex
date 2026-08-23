from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import re

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


def _runner_defaults(filename: str) -> dict[str, str]:
    text = (ROOT / "bash_scripts" / filename).read_text(encoding="utf-8")
    return dict(
        re.findall(r'^([A-Z][A-Z0-9_]*)="\$\{[0-9]+:-([^}]*)\}"$', text, re.MULTILINE)
    )


def test_simulation_laplace_mh_is_method_matched_to_laplace(monkeypatch):
    laplace = _load_example("03", "03_simulation_laplace.py", monkeypatch)
    laplace_mh = _load_example("08", "08_simulation_laplace_mh.py", monkeypatch)

    settings = (
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
        "DRAWS",
        "WARMUP",
        "CHAINS",
        "SEED",
        "FIGURE_FORMATS",
        "PREDICTIVE_DRAWS",
        "FORECAST_HORIZON",
        "FORECAST_HISTORY",
    )
    for setting in settings:
        assert getattr(laplace_mh, setting) == getattr(laplace, setting)
    assert laplace_mh.FIT_MODEL.to_dict() == laplace.FIT_MODEL.to_dict()
    assert laplace_mh.PRIOR_SETTINGS == laplace.PRIOR_SETTINGS
    assert _scenario_contract(laplace_mh) == _scenario_contract(laplace)
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
    )
    for setting in settings:
        assert getattr(laplace_mh, setting) == getattr(laplace, setting)
    assert laplace_mh.MODEL.to_dict() == laplace.MODEL.to_dict()
    assert laplace_mh.PRIOR_SETTINGS == laplace.PRIOR_SETTINGS

    source = (ROOT / "examples" / "09_uccle_laplace_mh.py").read_text(
        encoding="utf-8"
    )
    for directory in ("fits", "tables", "figures"):
        assert f'OUTPUT_DIR / "{directory}"' in source
    assert 'engine="laplace_mh"' in source
    assert "bx.Laplace(mh_steps=MH_STEPS)" in source


def test_laplace_mh_runners_match_their_laplace_counterparts():
    simulation = _runner_defaults("run_03_simulation_laplace.sh")
    simulation_mh = _runner_defaults("run_08_simulation_laplace_mh.sh")
    uccle = _runner_defaults("run_05_uccle_laplace.sh")
    uccle_mh = _runner_defaults("run_09_uccle_laplace_mh.sh")

    assert {key: simulation_mh[key] for key in simulation} == simulation
    assert {key: uccle_mh[key] for key in uccle} == uccle
    assert simulation_mh["MH_STEPS"] == "1"
    assert uccle_mh["MH_STEPS"] == "1"


def test_no_temporary_simulation_debug_overrides_remain():
    source = (ROOT / "examples" / "03_simulation_laplace.py").read_text(
        encoding="utf-8"
    )
    assert "all_great_except_for_RW" not in source
    assert "only the random walk" not in source
    assert source.count("SCENARIOS =") == 1
    assert source.count("OUTPUT_DIR =") == 1
