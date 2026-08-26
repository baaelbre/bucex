#!/usr/bin/env python3
"""Run one configured example and parallelize its independent MCMC chains.

Scientific and computational settings remain in the selected JSON file. For
examples that save one-chain fits, this runner makes one temporary JSON per
chain, runs those chains concurrently, and then asks the same example to
combine the fits and create the final tables and figures.

When ``output.run_id`` is null, the effective run ID includes the selected
configuration name and the PBS job ID (or local process ID). Concurrent jobs
therefore cannot share outputs merely because they started in the same second.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


CHAIN_EXAMPLES = {"03", "04", "05", "06", "07", "08", "09", "10", "11", "12"}


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must contain a JSON object: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run(command: list[str], *, log_path: Path | None = None) -> subprocess.Popen:
    if log_path is None:
        return subprocess.Popen(command)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
    process._bucex_log_stream = stream  # type: ignore[attr-defined]
    return process


def _wait(process: subprocess.Popen) -> int:
    return_code = process.wait()
    stream = getattr(process, "_bucex_log_stream", None)
    if stream is not None:
        stream.close()
    return return_code


def _result_directory(
    project_root: Path,
    config: dict,
    script_stem: str,
    run_id: str,
) -> Path:
    results_root = Path(config["output"]["results_root"])
    if not results_root.is_absolute():
        results_root = project_root / results_root
    parent = results_root / script_stem
    prefix = f"{run_id}__"
    matches = (
        sorted(
            path
            for path in parent.iterdir()
            if path.is_dir() and path.name.startswith(prefix)
        )
        if parent.is_dir()
        else []
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one result directory for {run_id!r}; found {len(matches)} "
            f"under {parent}."
        )
    return matches[0].resolve()


def _automatic_run_id(config_path: Path) -> str:
    """Return a readable ID unique across simultaneous submissions."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = "".join(
        character if character.isalnum() else "-"
        for character in config_path.stem
    ).strip("-").lower()
    execution = os.environ.get("PBS_JOBID") or f"pid{os.getpid()}"
    execution = execution.split(".", 1)[0]
    execution = "".join(
        character if character.isalnum() else "-"
        for character in execution
    ).strip("-").lower()
    return "_".join(
        value for value in (timestamp, config_name, execution) if value
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-workers", type=int, default=None)
    arguments = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    script = arguments.script
    if not script.is_absolute():
        script = project_root / script
    config_path = arguments.config
    if not config_path.is_absolute():
        config_path = project_root / config_path
    script = script.resolve()
    config_path = config_path.resolve()
    if not script.is_file():
        raise FileNotFoundError(f"Example script not found: {script}")
    config = _load(config_path)

    command = [sys.executable, "-u", str(script), "--config"]
    script_number = script.name[:2]
    mcmc = config.get("mcmc", {})
    chains = int(mcmc.get("chains", 1))
    max_workers = arguments.max_workers
    if max_workers is None:
        max_workers = int(os.environ.get("PBS_NP", os.cpu_count() or 1))
    max_workers = max(1, int(max_workers))
    supports_parallel_chains = script_number in CHAIN_EXAMPLES
    if supports_parallel_chains and chains > max_workers:
        raise ValueError(
            f"The JSON requests {chains} chains but only {max_workers} workers "
            "were allocated. Increase PBS ppn or reduce mcmc.chains."
        )

    output = config.setdefault("output", {})
    base_run_id = str(output.get("run_id") or _automatic_run_id(config_path))
    output["run_id"] = base_run_id
    config["_runner"] = {
        "source_config": str(config_path),
        "base_run_id": base_run_id,
        "pbs_job_id": os.environ.get("PBS_JOBID"),
    }
    log_root = project_root / "logs"
    base_seed = int(mcmc.get("seed", 1))

    with tempfile.TemporaryDirectory(
        prefix=f"bucex_{script.stem}_"
    ) as temporary:
        temporary_root = Path(temporary)

        if chains <= 1 or not supports_parallel_chains:
            effective_path = temporary_root / "effective_config.json"
            _write(effective_path, config)
            print(f"Running {script.name} with {config_path}", flush=True)
            print(f"Run ID: {base_run_id}", flush=True)
            raise SystemExit(_wait(_run(command + [str(effective_path)])))

        processes: list[tuple[int, str, Path, subprocess.Popen]] = []
        print(
            f"Launching {chains} independent chains for {script.name} "
            f"({max_workers} allocated workers)",
            flush=True,
        )
        print(f"Source config: {config_path}", flush=True)
        print(f"Run ID: {base_run_id}", flush=True)
        for chain_index in range(chains):
            chain_number = chain_index + 1
            chain_run_id = f"{base_run_id}_chain{chain_number:02d}"
            chain_config = deepcopy(config)
            chain_config["mcmc"]["chains"] = 1
            chain_config["mcmc"]["seed"] = base_seed + chain_index
            chain_config.setdefault("runtime", {})["chain_only"] = True
            chain_config["runtime"]["combine_runs"] = []
            chain_config["output"]["run_id"] = chain_run_id
            chain_config["_runner"]["chain_number"] = chain_number
            chain_path = temporary_root / f"chain_{chain_number:02d}.json"
            _write(chain_path, chain_config)
            log_path = (
                log_root
                / f"{script.stem}_{base_run_id}_chain{chain_number:02d}.log"
            )
            process = _run(command + [str(chain_path)], log_path=log_path)
            processes.append((chain_number, chain_run_id, log_path, process))
            print(
                f"  chain {chain_number}: seed={base_seed + chain_index}, "
                f"log={log_path}",
                flush=True,
            )

        failures: list[str] = []
        chain_directories: list[Path] = []
        for chain_number, chain_run_id, log_path, process in processes:
            return_code = _wait(process)
            if return_code:
                failures.append(
                    f"chain {chain_number} (exit {return_code}; {log_path})"
                )
                continue
            chain_directories.append(
                _result_directory(project_root, config, script.stem, chain_run_id)
            )
            print(f"  chain {chain_number} finished", flush=True)
        if failures:
            raise RuntimeError("One or more chains failed: " + "; ".join(failures))

        combined_config = deepcopy(config)
        combined_config["mcmc"]["chains"] = chains
        combined_config["mcmc"]["seed"] = base_seed
        combined_config.setdefault("runtime", {})["chain_only"] = False
        combined_config["runtime"]["combine_runs"] = [
            str(path) for path in chain_directories
        ]
        combined_config["output"]["run_id"] = f"{base_run_id}_combined"
        combined_config["_runner"]["chain_number"] = None
        combined_path = temporary_root / "combine.json"
        _write(combined_path, combined_config)
        print("Combining chains and creating final tables and figures", flush=True)
        raise SystemExit(_wait(_run(command + [str(combined_path)])))


if __name__ == "__main__":
    main()
