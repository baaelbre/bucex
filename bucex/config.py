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
    """Read JSON, resolving an optional relative ``extends`` base recursively.

    Dictionaries merge recursively; scalar values, lists and null replace the
    base value. Cycles are errors. Saving the returned dictionary records all
    effective settings and does not depend on the original base file.
    """

    return _load_config(Path(path).expanduser().resolve(), ())


def _load_config(resolved, parents):
    if resolved in parents:
        raise ValueError(f"Configuration inheritance cycle: {resolved}")

    if not resolved.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must contain one JSON object: {resolved}")
    base = value.pop("extends", None)
    if base is None:
        return deepcopy(value)
    if not isinstance(base,str):
        raise ValueError("extends must be one relative or absolute file path.")
    inherited = _load_config((resolved.parent/base).resolve(), (*parents,resolved))
    return _merge(inherited,value)


def _merge(base, overrides):
    result = deepcopy(base)
    for key,value in overrides.items():
        result[key] = (_merge(result[key],value) if isinstance(result.get(key),dict) and isinstance(value,dict)
                       else deepcopy(value))
    return result


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
