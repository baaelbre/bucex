"""Run one scenario and one chain of the Laplace-MH simulation study.

This file is an HPC worker, not an eleventh scientific example.  The PBS array
maps one task to one ``(scenario, chain)`` pair.  Every task imports the model,
simulation, prior, and fitting definitions from ``08_simulation_laplace_mh``
and writes only its own fit below the shared run directory.
"""
from __future__ import annotations

from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if (SOURCE_ROOT / "bucex").is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import bucex as bx


def _load_scientific_example():
    path = Path(__file__).with_name("08_simulation_laplace_mh.py")
    spec = importlib.util.spec_from_file_location("bucex_example08_shared", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the scientific example from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _truth_matches(fit: bx.FitResult, expected: dict[str, object]) -> bool:
    return fit.metadata.get("truth") == expected


def main() -> None:
    shared_run_dir_text = os.environ.get("BUCEX_SHARED_RUN_DIR", "").strip()
    if not shared_run_dir_text:
        raise ValueError("BUCEX_SHARED_RUN_DIR is required for an array task.")
    scenario_key = os.environ.get("BUCEX_SCENARIO_KEY", "").strip()
    if not scenario_key:
        raise ValueError("BUCEX_SCENARIO_KEY is required for an array task.")
    chain_index = int(os.environ.get("BUCEX_CHAIN_INDEX", "0"))
    if chain_index < 1:
        raise ValueError("BUCEX_CHAIN_INDEX must be a positive integer.")
    base_seed = int(os.environ.get("BUCEX_SEED", "13081997"))
    overwrite = os.environ.get("BUCEX_OVERWRITE", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    progress = os.environ.get("BUCEX_PROGRESS", "1").lower() not in {
        "0",
        "false",
        "no",
    }

    example = _load_scientific_example()
    if scenario_key not in example.ALL_SCENARIO_INDEX:
        raise ValueError(
            f"Unknown scenario {scenario_key!r}; available keys are "
            + ", ".join(example.ALL_SCENARIO_INDEX)
            + "."
        )
    scenario_number = example.ALL_SCENARIO_INDEX[scenario_key]
    scenario = example.ALL_SCENARIOS[scenario_number]
    chain_seed = base_seed + chain_index - 1 + 100 * scenario_number
    shared_run_dir = Path(shared_run_dir_text)
    chain_root = shared_run_dir / "tasks" / f"chain{chain_index:02d}"
    fit_path = chain_root / "fits" / scenario_key / "combined.bucex"
    manifest_path = chain_root / "manifests" / f"{scenario_key}.json"

    table, truth = example.simulate_scenario_table(scenario)
    if fit_path.is_file() and not overwrite:
        fit = bx.FitResult.load(fit_path)
        valid = (
            fit.plan.engine == "laplace_mh"
            and fit.n_time == example.N_TIME
            and fit.n_chains == 1
            and fit.draws_per_chain == example.DRAWS
            and fit.family == example.FIT_MODEL.family
            and fit.model.period == example.FIT_MODEL.period
            and fit.state_names == example.FIT_MODEL.state_names
            and fit.compiled.noise_names == example.FIT_MODEL.noise_names
            and fit.metadata.get("prior_settings") == example.PRIOR_SETTINGS
            and _truth_matches(fit, truth)
            and fit.metadata.get("array_chain_index") == chain_index
            and fit.metadata.get("array_chain_seed") == chain_seed
        )
        if not valid:
            raise ValueError(
                f"Existing task fit does not match this request: {fit_path}. "
                "Use a new RUN_ID or set OVERWRITE=1."
            )
        print(f"Reusing completed array task: {fit_path}")
        return

    fit = example.fit_scenario(
        scenario,
        table,
        truth,
        seed=chain_seed,
        chains=1,
        progress=progress,
    )
    fit.metadata.update(
        {
            "array_task": True,
            "array_scenario_key": scenario_key,
            "array_scenario_index": scenario_number + 1,
            "array_chain_index": chain_index,
            "array_chain_seed": chain_seed,
            "shared_run_directory": str(shared_run_dir),
        }
    )
    fit_path.parent.mkdir(parents=True, exist_ok=True)
    fit.save(fit_path)

    manifest = {
        "status": "complete",
        "completed_at": datetime.now().astimezone().isoformat(),
        "bucex_version": bx.__version__,
        "scenario": scenario["name"],
        "scenario_key": scenario_key,
        "scenario_index": scenario_number + 1,
        "chain": chain_index,
        "seed": chain_seed,
        "n_time": example.N_TIME,
        "period": example.PERIOD,
        "draws": example.DRAWS,
        "warmup": example.WARMUP,
        "mh_steps": example.MH_STEPS,
        "fit": str(fit_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "Completed Laplace-MH array task: "
        f"scenario={scenario_key}, chain={chain_index}, fit={fit_path}"
    )


if __name__ == "__main__":
    main()
