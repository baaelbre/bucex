"""Small, explicit JSON configuration helpers.

The examples deliberately keep every scientific and runtime choice in JSON.
This module provides the same behaviour to users without introducing a second
configuration language or hidden environment-variable overrides.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a JSON object and return an independent mutable dictionary."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must contain one JSON object: {resolved}")
    return deepcopy(value)


def save_config(config: Mapping[str, Any], path: str | Path) -> Path:
    """Write a JSON configuration deterministically and return its path."""

    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(dict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return resolved


def config_title(config: Mapping[str, Any], key: str | None = None) -> str | None:
    """Return an optional title from ``config['figures']``.

    ``figures.title`` controls a script-wide title. A named value in
    ``figures.titles`` takes precedence when ``key`` is supplied. JSON ``null``
    and empty strings both mean that no title is drawn.
    """

    figures = config.get("figures", {})
    if not isinstance(figures, Mapping):
        return None
    value: Any = figures.get("title")
    titles = figures.get("titles", {})
    if key is not None and isinstance(titles, Mapping) and key in titles:
        value = titles[key]
    if value is None or not str(value).strip():
        return None
    return str(value)


__all__ = ["config_title", "load_config", "save_config"]
