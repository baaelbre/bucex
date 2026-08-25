from __future__ import annotations

from pathlib import Path

import bucex as bx


ROOT = Path(__file__).resolve().parents[1]


def test_v130_parallel_runner_uses_json_chain_copies_and_public_combine_paths():
    assert bx.__version__ == "1.3.0"
    source = (ROOT / "hpc" / "run_example.py").read_text(encoding="utf-8")
    assert 'chain_config["mcmc"]["chains"] = 1' in source
    assert 'chain_config["runtime"]["chain_only"] = True' in source
    assert 'combined_config["runtime"]["combine_runs"]' in source
    assert "subprocess.Popen" in source
    assert "--config" in source
    assert "BUCEX_SCENARIO_KEYS" not in source


def test_v130_hpc_surface_has_one_runner_and_one_pbs_file_per_example():
    expected = {
        "00_uccle_record",
        "01_tail_simulations",
        "02_structural_simulations",
        "03_simulation_laplace",
        "04_simulation_pgas",
        "05_uccle_laplace",
        "06_uccle_pgas",
        "07_centered_ig",
        "08_simulation_laplace_mh",
        "09_uccle_laplace_mh",
    }
    assert {
        path.stem.removeprefix("run_")
        for path in (ROOT / "bash_scripts").glob("run_*.sh")
    } == expected
    assert {
        path.stem.removeprefix("submit_")
        for path in (ROOT / "job_scripts").glob("submit_*.pbs")
    } == expected

    obsolete = (
        ROOT / "examples" / "_08_simulation_laplace_mh_task.py",
        ROOT / "config" / "laplace_mh_simulation_array.sh",
        ROOT / "validation" / "run_hpc_fanout_smoke.py",
    )
    assert not any(path.exists() for path in obsolete)


def test_v130_pbs_allocates_parallel_chain_workers_and_passes_one_config():
    for path in sorted((ROOT / "job_scripts").glob("submit_*.pbs")):
        source = path.read_text(encoding="utf-8")
        assert "#PBS -l nodes=1:ppn=" in source
        assert 'CONFIG="${CONFIG:-examples/config/' in source
        assert '"${CONFIG}" "${PBS_NP:-' in source
        assert "BUCEX_DRAWS" not in source
        assert "BUCEX_WARMUP" not in source
