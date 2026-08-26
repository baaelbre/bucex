from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_example(script: str, config: dict, tmp_path: Path) -> Path:
    config_path = tmp_path / f"{Path(script).stem}.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    environment = dict(os.environ)
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / script),
            "--config",
            str(config_path),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result_directories = list(
        (Path(config["output"]["results_root"]) / Path(script).stem).iterdir()
    )
    assert len(result_directories) == 1
    return result_directories[0]


def test_phi_jsons_are_annotated_and_location_truth_is_not_hidden():
    simulation = json.loads(
        (
            ROOT / "examples/config/phi/simulation_linear.json"
        ).read_text(encoding="utf-8")
    )
    uccle = json.loads(
        (
            ROOT / "examples/config/phi/uccle/01_txx_linear.json"
        ).read_text(encoding="utf-8")
    )
    assert simulation["_comment"]
    assert simulation["simulation"]["_comment"]
    assert simulation["simulation"]["location"]["_comment"]
    assert simulation["simulation"]["location"]["trend_component"] == (
        "local_linear_trend"
    )
    assert simulation["simulation"]["location"]["level_mode"] == "dynamic"
    assert simulation["simulation"]["location"]["trend_mode"] == "dynamic"
    assert simulation["model"]["location"]["structure"] == "ssvs"
    assert simulation["simulation"]["phi"]["mode"] == "stationary"
    assert simulation["model"]["phi"] == "linear"
    assert uccle["model"]["location"] == simulation["model"]["location"]
    assert (ROOT / "examples/config/phi/README.md").is_file()


def test_phi_examples_keep_one_readable_main_and_no_configuration_helpers():
    for filename in ("10_simulation_phi.py", "11_uccle_phi.py"):
        source = (ROOT / "examples" / filename).read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        assert functions == ["main"]
        assert 'config["model"]' in source
        assert 'model_settings["location"]' in source
        assert 'figures["predictive_draws"]' in source


def test_simulation_phi_writes_the_established_full_report(tmp_path: Path):
    config = json.loads(
        (
            ROOT / "examples/config/phi/simulation_linear.json"
        ).read_text(encoding="utf-8")
    )
    config = deepcopy(config)
    config["simulation"]["n_time"] = 16
    config["mcmc"].update(draws=2, warmup=1, chains=1)
    config["runtime"]["progress"] = False
    config["figures"].update(
        formats=["png"],
        dpi=50,
        predictive_draws=2,
        forecast_horizon=4,
        forecast_history=8,
    )
    config["output"].update(
        results_root=str(tmp_path / "results"),
        run_id="v141_simulation_report",
        overwrite=True,
    )
    result = _run_example("10_simulation_phi.py", config, tmp_path)
    table_dir = result / "tables/phi_linear"
    figure_dir = result / "figures/phi_linear"
    assert {
        "parameters.csv",
        "diagnostics.csv",
        "algorithm.csv",
        "trajectory.csv",
        "selection.csv",
        "models.csv",
        "switching.csv",
        "posterior_predictive.csv",
        "forecast.csv",
        "forecast_phase_01.csv",
        "forecast_level.csv",
        "phi.csv",
        "phi_models.csv",
        "summary.json",
    } <= {path.name for path in table_dir.iterdir()}
    assert {
        "trajectory.png",
        "trajectory_phase_01.png",
        "posterior_predictive.png",
        "forecast.png",
        "forecast_phase_01.png",
        "forecast_level.png",
        "level.png",
        "level_no_observations.png",
        "slope.png",
        "selection.png",
        "process_sd.png",
        "gev.png",
        "season.png",
        "seasonal_patterns.png",
        "phi.png",
    } <= {path.name for path in figure_dir.iterdir()}
    assert (result / "fits/phi_linear/combined.bucex").is_file()
    assert (result / "simulations/phi_linear.csv").is_file()


def test_uccle_phi_writes_the_established_full_report(tmp_path: Path):
    config = json.loads(
        (
            ROOT / "examples/config/phi/uccle/01_txx_linear.json"
        ).read_text(encoding="utf-8")
    )
    config = deepcopy(config)
    config["data"].update(start="1892-01-01", end="1893-12-31")
    config["mcmc"].update(draws=2, warmup=1, chains=1)
    config["runtime"]["progress"] = False
    config["figures"].update(
        formats=["png"],
        dpi=50,
        predictive_draws=2,
        forecast_horizon=12,
        forecast_history=12,
    )
    config["figures"]["seasonal_patterns"]["years"] = [1892, 1893]
    config["output"].update(
        results_root=str(tmp_path / "results"),
        run_id="v141_uccle_report",
        overwrite=True,
    )
    result = _run_example("11_uccle_phi.py", config, tmp_path)
    table_dir = result / "tables/TXx"
    figure_dir = result / "figures/TXx"
    assert {
        "parameters.csv",
        "diagnostics.csv",
        "algorithm.csv",
        "trajectory.csv",
        "selection.csv",
        "models.csv",
        "switching.csv",
        "posterior_predictive.csv",
        "forecast.csv",
        "forecast_july.csv",
        "forecast_level.csv",
        "phi.csv",
        "phi_models.csv",
        "summary.json",
    } <= {path.name for path in table_dir.iterdir()}
    assert {
        "trajectory.png",
        "trajectory_july.png",
        "posterior_predictive.png",
        "forecast.png",
        "forecast_july.png",
        "forecast_level.png",
        "level.png",
        "level_no_observations.png",
        "slope.png",
        "selection.png",
        "process_sd.png",
        "gev.png",
        "season.png",
        "seasonal_patterns.png",
        "phi.png",
    } <= {path.name for path in figure_dir.iterdir()}
    assert (result / "fits/TXx/combined.bucex").is_file()
    assert (result / "tables/selection_all.csv").is_file()
    assert (result / "tables/laplace_mh_diagnostics.csv").is_file()
    assert (result / "tables/phi_summary_all.csv").is_file()
