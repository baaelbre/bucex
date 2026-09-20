"""Read report exports without loading full latent-state posterior archives."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..diagnostics.calendar import pit_normal_scores


class ReportCollection:
    """An explicit mapping from series names to report directories.

    Discovery rejects duplicate reports for a response, rather than silently
    choosing a fit from another configuration. Construct directly with a dict
    to combine deliberately chosen reports from separate jobs.
    """

    def __init__(self, directories):
        if not directories:
            raise ValueError("Provide at least one report directory.")
        self.directories = {str(name): Path(path).expanduser().resolve()
                            for name, path in directories.items()}
        self._sources = {}
        self._cache = {}

    @classmethod
    def from_directories(cls, paths, *, series=None):
        """Discover response-level exports below one or several directories."""
        if isinstance(paths, (str, Path)):
            paths = [paths]
        selected = None if series is None else tuple(series)
        found = {}
        for root in paths:
            root = Path(root).expanduser().resolve()
            if not root.is_dir():
                raise FileNotFoundError(f"Report directory not found: {root}")
            for file in sorted(root.rglob("*_level.csv")):
                name = file.name.removesuffix("_level.csv")
                if selected is not None and name not in selected:
                    continue
                if name in found and found[name] != file.parent:
                    raise ValueError(f"Multiple reports for {name}; supply specific case directories.")
                found[name] = file.parent
        if selected is not None:
            missing = set(selected)-set(found)
            if missing:
                raise FileNotFoundError(f"Missing report series: {sorted(missing)}")
            found = {name: found[name] for name in selected}
        return cls(found)

    @property
    def series(self):
        return tuple(self.directories)

    def _record(self, path):
        self._sources[str(path)] = dict(path=str(path), bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def read(self, series, table):
        """Read ``SERIES_table.csv[.gz]`` and remember its byte checksum."""
        key = series, table
        if key not in self._cache:
            path = self.directories[series]/f"{series}_{table}.csv"
            if not path.exists():
                path = path.with_suffix(".csv.gz")
            if not path.exists():
                raise FileNotFoundError(f"Missing {series}_{table}.csv[.gz]. Re-export the saved fit with research.serra.report.")
            value = pd.read_csv(path)
            if "time" in value:
                value["time"] = pd.to_datetime(value.time)
                if value.time.isna().any() or value.time.duplicated().any():
                    raise ValueError(f"Invalid or duplicate dates in {path.name}.")
                value = value.sort_values("time").reset_index(drop=True)
            self._record(path)
            self._cache[key] = value
        return self._cache[key].copy()

    def metadata(self, series):
        result = {}
        for name in ("config.json", "run.json", f"{series}_prediction_notes.json"):
            path = self.directories[series]/name
            if path.exists():
                result[name] = json.loads(path.read_text(encoding="utf-8"))
                self._record(path)
        return result

    def interval_level(self):
        """Return the declared common level; never relabel old 90% bands as 95%."""
        levels = set()
        for name in self.series:
            metadata = self.metadata(name)
            config = metadata.get("config.json", {})
            notes = metadata.get(f"{name}_prediction_notes.json", {})
            level = notes.get("interval_level", config.get("credible_interval"))
            if level is None:
                raise ValueError(f"No interval level recorded for {name}; supply its report config.json.")
            level = float(level)
            if not 0 < level < 1:
                raise ValueError("Report interval levels must lie in (0, 1).")
            levels.add(level)
        if len(levels) != 1:
            raise ValueError("Selected reports use different interval levels; re-export them consistently.")
        return levels.pop()

    def check_event(self, series, threshold):
        """Check a risk panel's threshold against the saved report specification."""
        metadata = self.metadata(series)
        notes = metadata.get(f"{series}_prediction_notes.json", {})
        config = metadata.get("config.json", {})
        saved = notes.get("threshold", config.get("risks", {}).get(series))
        if saved is None or not np.isclose(float(saved), float(threshold), rtol=0, atol=1e-12):
            raise ValueError(f"Risk threshold for {series} does not match its saved report: {saved} vs {threshold}.")

    def normal_scores(self, *, series=None):
        """Aligned original-orientation PIT scores; missing months are not dropped."""
        columns = {}
        common_dates = None
        for name in self.series if series is None else series:
            data = self.read(name, "smoothed_pit")
            index = pd.DatetimeIndex(data.time)
            if common_dates is not None and not index.equals(common_dates):
                raise ValueError("Residual dependence requires reports with exactly aligned dates.")
            common_dates = index
            columns[name] = pit_normal_scores(data.pit)
        return pd.DataFrame(columns, index=common_dates)

    @property
    def sources(self):
        return list(self._sources.values())


__all__ = ["ReportCollection"]
