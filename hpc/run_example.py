#!/usr/bin/env python3
"""Run one configured example, optionally fanning MCMC chains across cores.

Scientific settings remain in the selected JSON file.  For examples that
support saved one-chain fits, this runner makes one temporary JSON per chain,
runs those chains concurrently, and then invokes the example once more to
combine the fits and create the final tables and figures.
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


CHAIN_EXAMPLES = {"03", "04", "05", "06", "07", "08", "09"}


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must contain a JSON object: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _result_directory(project_root: Path, config: dict, script_stem: str, run_id: str) -> Path:
    results_root = Path(config["output"]["results_root"])
    if not results_root.is_absolute():
        results_root = project_root / results_root
    matches = sorted((results_root / script_stem).glob(f"{run_id}__*"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one result directory for {run_id!r}; found {len(matches)} "
            f"under {results_root / script_stem}."
        )
    return matches[0].resolve()


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
    config = _load(config_path)

    python = sys.executable
    command = [python, "-u", str(script), "--config"]
    script_number = script.name[:2]
    mcmc = config.get("mcmc", {})
    chains = int(mcmc.get("chains", 1))
    max_workers = arguments.max_workers
    if max_workers is None:
        max_workers = int(os.environ.get("PBS_NP", os.cpu_count() or 1))
    max_workers = max(1, int(max_workers))

    if chains <= 1 or script_number not in CHAIN_EXAMPLES:
        print(f"Running {script.name} with {config_path}", flush=True)
        raise SystemExit(_wait(_run(command + [str(config_path)])))
    if chains > max_workers:
        raise ValueError(
            f"The JSON requests {chains} chains but only {max_workers} workers "
            "were allocated. Increase PBS ppn or reduce mcmc.chains."
        )

    output = config.setdefault("output", {})
    base_run_id = output.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = project_root / "logs"
    temporary_root = Path(tempfile.mkdtemp(prefix=f"bucex_{script.stem}_"))
    processes: list[tuple[int, str, Path, subprocess.Popen]] = []
    base_seed = int(mcmc.get("seed", 1))

    print(
        f"Launching {chains} independent chains for {script.name} "
        f"({max_workers} allocated workers)",
        flush=True,
    )
    for chain_index in range(chains):
        chain_number = chain_index + 1
        chain_run_id = f"{base_run_id}_chain{chain_number:02d}"
        chain_config = deepcopy(config)
        chain_config["mcmc"]["chains"] = 1
        chain_config["mcmc"]["seed"] = base_seed + chain_index
        chain_config.setdefault("runtime", {})["chain_only"] = True
        chain_config["runtime"]["combine_runs"] = []
        chain_config["output"]["run_id"] = chain_run_id
        chain_path = temporary_root / f"chain_{chain_number:02d}.json"
        _write(chain_path, chain_config)
        log_path = log_root / f"{script.stem}_{base_run_id}_chain{chain_number:02d}.log"
        process = _run(command + [str(chain_path)], log_path=log_path)
        processes.append((chain_number, chain_run_id, log_path, process))
        print(f"  chain {chain_number}: seed={base_seed + chain_index}, log={log_path}", flush=True)

    failures: list[str] = []
    chain_directories: list[Path] = []
    for chain_number, chain_run_id, log_path, process in processes:
        return_code = _wait(process)
        if return_code:
            failures.append(f"chain {chain_number} (exit {return_code}; {log_path})")
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
    combined_path = temporary_root / "combine.json"
    _write(combined_path, combined_config)
    print("Combining chains and creating final tables and figures", flush=True)
    raise SystemExit(_wait(_run(command + [str(combined_path)])))


if __name__ == "__main__":
    main()
