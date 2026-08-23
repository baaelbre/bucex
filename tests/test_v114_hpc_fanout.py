from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess

import bucex as bx


ROOT = Path(__file__).resolve().parents[1]


def _load_example(name: str):
    path = ROOT / "examples" / "08_simulation_laplace_mh.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v114_version_and_scenario_filter(monkeypatch):
    assert bx.__version__ == "1.2.0"
    monkeypatch.delenv("BUCEX_SCENARIO_KEYS", raising=False)
    complete = _load_example("bucex_example08_complete")
    assert [scenario["key"] for scenario in complete.SCENARIOS] == [
        "stationary",
        "linear",
        "random_walk",
        "llt",
        "dynamic_season",
        "llt_season",
    ]

    monkeypatch.setenv("BUCEX_SCENARIO_KEYS", os.pathsep.join(("stationary", "llt")))
    selected = _load_example("bucex_example08_selected")
    assert [scenario["key"] for scenario in selected.SCENARIOS] == [
        "stationary",
        "llt",
    ]
    assert selected.ALL_SCENARIO_INDEX["llt"] == 3


def test_pbs_array_maps_all_24_scenario_chain_pairs_once():
    command = r'''
source config/laplace_mh_simulation_array.sh
for task in $(seq 1 24); do
  laplace_mh_map_task "$task" 4 "$LMH_DEFAULT_SCENARIO_KEYS"
  printf '%s,%s\n' "$LMH_TASK_SCENARIO" "$LMH_TASK_CHAIN"
done
'''
    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    pairs = [tuple(line.split(",")) for line in completed.stdout.splitlines()]
    assert len(pairs) == 24
    assert len(set(pairs)) == 24
    assert pairs[0] == ("stationary", "1")
    assert pairs[3] == ("stationary", "4")
    assert pairs[4] == ("linear", "1")
    assert pairs[-1] == ("llt_season", "4")


def test_array_resources_defaults_and_shared_layout_contract():
    fit_pbs = (
        ROOT / "job_scripts" / "submit_08_simulation_laplace_mh_array.pbs"
    ).read_text(encoding="utf-8")
    final_pbs = (
        ROOT / "job_scripts" / "submit_08_simulation_laplace_mh_finalize.pbs"
    ).read_text(encoding="utf-8")
    submitter = (
        ROOT / "bash_scripts" / "qsub_08_simulation_laplace_mh.sh"
    ).read_text(encoding="utf-8")
    worker = (
        ROOT / "examples" / "_08_simulation_laplace_mh_task.py"
    ).read_text(encoding="utf-8")
    finalizer = (
        ROOT / "bash_scripts" / "run_08_simulation_laplace_mh_finalize.sh"
    ).read_text(encoding="utf-8")

    assert "#PBS -l walltime=06:00:00" in fit_pbs
    assert "#PBS -l nodes=1:ppn=1" in fit_pbs
    assert "#PBS -l walltime=01:00:00" in final_pbs
    assert 'DRAWS="${DRAWS:-1000}"' in submitter
    assert 'WARMUP="${WARMUP:-1000}"' in submitter
    assert 'N_TASKS=$((LMH_N_SCENARIOS * CHAINS))' in submitter
    assert '-t "1-${N_TASKS}%${MAX_CONCURRENT}"' in submitter
    assert 'depend=${DEPENDENCY_KIND}:${FIT_ID}' in submitter

    assert 'shared_run_dir / "tasks" / f"chain{chain_index:02d}"' in worker
    assert '"fits" / scenario_key / "combined.bucex"' in worker
    assert '"manifests" / f"{scenario_key}.json"' in worker
    assert 'chain_dir="${SHARED_RUN_DIR}/tasks/chain${chain_label}"' in finalizer
    assert 'export BUCEX_COMBINE_RUNS="${COMBINE_RUNS}"' in finalizer


def test_shared_run_directory_has_the_final_chain_signature():
    command = r'''
source config/laplace_mh_simulation_array.sh
signature=$(laplace_mh_run_signature 1000 4 1000 1000 4 1)
laplace_mh_run_directory results run42 "$signature"
'''
    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == (
        "results/08_simulation_laplace_mh/"
        "run42__n1000p4_d1000w1000c4m1"
    )
