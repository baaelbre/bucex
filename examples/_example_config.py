"""One intentionally small configuration entry point for all examples."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import bucex as bx


EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = EXAMPLE_DIR / "config"


def load_example_config(default_filename: str) -> tuple[dict[str, Any], Path]:
    """Load the default JSON or the file selected with ``--config``."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_DIR / default_filename,
        help="JSON file containing every setting for this run.",
    )
    # ``parse_known_args`` keeps the examples importable from notebooks and
    # test runners that own additional command-line flags. ``--config`` is the
    # only option interpreted by an example.
    arguments, _ = parser.parse_known_args()
    path = arguments.config.expanduser().resolve()
    return bx.load_config(path), path


def load_simulation_config() -> tuple[dict[str, Any], Path]:
    return load_example_config("simulation.json")


def load_uccle_config() -> tuple[dict[str, Any], Path]:
    return load_example_config("uccle.json")


def load_centered_ig_config() -> tuple[dict[str, Any], Path]:
    return load_example_config("centered_ig.json")


__all__ = [
    "load_centered_ig_config",
    "load_example_config",
    "load_simulation_config",
    "load_uccle_config",
]
