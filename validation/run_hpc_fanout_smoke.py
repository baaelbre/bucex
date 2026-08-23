#!/usr/bin/env python3
"""Real two-task/one-finalizer smoke test for the 1.2.0 PBS workflow."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import bucex as bx


def run(work_dir: Path) -> dict[str, object]:
    if bx.__version__ != "1.2.0":
        raise RuntimeError(f"Expected bucex 1.2.0, found {bx.__version__}.")
    started = time.perf_counter()
    work_dir = work_dir.resolve()
    run_id = f"fanout_smoke_{os.getpid()}"
    common = [
        "16",  # n_time
        "4",  # period
        "13081997",  # simulation seed
        "1",  # draws
        "1",  # warmup
        "2",  # total chains
        "13081997",  # MCMC seed
        str(work_dir),
        run_id,
        "0",  # overwrite
        "1",  # MH steps
        "stationary",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "BUCEX_PYTHON": sys.executable,
            "BUCEX_PROGRESS": "0",
            "MPLBACKEND": "Agg",
        }
    )

    processes = []
    for chain in (1, 2):
        command = [
            "bash",
            "bash_scripts/run_08_simulation_laplace_mh_task.sh",
            "stationary",
            str(chain),
            *common,
        ]
        processes.append(
            subprocess.Popen(
                command,
                cwd=SOURCE_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
    task_outputs = []
    for process in processes:
        output, _ = process.communicate()
        task_outputs.append(output)
        if process.returncode != 0:
            raise RuntimeError(f"Fan-out task failed:\n{output}")

    final_command = [
        "bash",
        "bash_scripts/run_08_simulation_laplace_mh_finalize.sh",
        *common,
    ]
    final = subprocess.run(
        final_command,
        cwd=SOURCE_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    run_dir = (
        work_dir
        / "08_simulation_laplace_mh"
        / f"{run_id}__n16p4_d1w1c2m1"
    )
    fit_path = run_dir / "fits" / "stationary" / "combined.bucex"
    fit = bx.FitResult.load(fit_path)
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    required = [
        fit_path,
        run_dir / "simulations" / "stationary.csv",
        run_dir / "tables" / "stationary" / "parameters.csv",
        run_dir / "figures" / "stationary" / "trajectory.png",
        run_dir / "tasks" / "chain01" / "manifests" / "stationary.json",
        run_dir / "tasks" / "chain02" / "manifests" / "stationary.json",
    ]
    if not all(path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise RuntimeError("Missing fan-out smoke artifacts:\n" + "\n".join(missing))
    if fit.n_chains != 2 or fit.draws_per_chain != 1:
        raise RuntimeError("The finalizer did not produce the requested two-chain fit.")
    if not config.get("array_workflow"):
        raise RuntimeError("run_config.json did not record the array workflow.")

    return {
        "bucex_version": bx.__version__,
        "run_directory": str(run_dir),
        "task_processes": 2,
        "task_outputs_complete": all(
            "Completed Laplace-MH array task" in output for output in task_outputs
        ),
        "finalizer_complete": "Finished combining array tasks" in final.stdout,
        "combined_chains": fit.n_chains,
        "draws_per_chain": fit.draws_per_chain,
        "array_workflow_recorded": bool(config["array_workflow"]),
        "required_artifacts_present": True,
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("validation/hpc_fanout_smoke_artifacts_1.2.0"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/hpc_fanout_smoke_1.2.0.json"),
    )
    args = parser.parse_args()
    result = run(args.work_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
