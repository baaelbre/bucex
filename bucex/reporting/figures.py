"""Build a declared set of figures and record inputs, intervals and omissions."""
from __future__ import annotations

import json
from pathlib import Path
import re
import textwrap

from ..__about__ import __version__
from ..plotting.style import publication_style, save_figure
from .panels import PANEL_BUILDERS, UnavailablePanelData


def _placeholder(name, reason):
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots(figsize=(10.6, 3.4), layout="constrained")
    ax.set_axis_off()
    message = "RESULT TO INSERT\n\n" + name.replace("_", " ") + "\n\n" + textwrap.fill(reason, 85)
    ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes,
             fontsize=11, bbox=dict(facecolor="#f5f8fb", edgecolor="#24658a", pad=20))
    return figure


def save_publication_figures(reports, directory, *, recipes, colors=None,
                             formats=("png",), dpi=180, style="manuscript", strict=False):
    """Regenerate manuscript panels using saved report CSVs only.

    Recipes declare data tables, response order, months and scientific labels.
    Missing exports create an explicit placeholder and manifest entry; use
    ``strict=True`` to fail instead. Invalid schemas, mismatched thresholds,
    duplicate fits and inconsistent interval levels always raise an error.
    Saved posterior intervals are used verbatim. No numerical result is fitted,
    smoothed, or invented by this function.
    """
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    colors = dict(colors or {})
    level = reports.interval_level()
    manifest = dict(bucex_version=__version__, interval_level=level, style=style,
        series=list(reports.series), figures=[], recipes=recipes, colors=colors,
        inference="Report-only: no posterior sampling or re-estimation.",
        residuals="Smoothed PIT diagnostics are descriptive in-sample checks; score correlations are not a fitted copula posterior.")
    names_seen = set()
    for spec in recipes:
        name = spec["name"]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name) or name in names_seen:
            raise ValueError("Figure names must be unique safe file stems.")
        names_seen.add(name)
        if spec["kind"] not in PANEL_BUILDERS:
            raise ValueError(f"Unknown figure kind {spec['kind']!r}.")
        names = tuple(spec.get("series", reports.series))
        if not names or set(names)-set(reports.series):
            raise ValueError("Figure series must be present in the selected report collection.")
        entry = dict(name=name, kind=spec["kind"], status="rendered", caption=spec.get("caption", ""))
        with publication_style(style=style, dpi=dpi, primary=colors.get(names[0])):
            existing = set(plt.get_fignums())
            try:
                figure, tables = PANEL_BUILDERS[spec["kind"]](reports, spec, names, colors)
            except (FileNotFoundError, UnavailablePanelData) as exc:
                for number in set(plt.get_fignums())-existing:
                    plt.close(number)
                if strict:
                    raise
                entry.update(status="placeholder", reason=str(exc))
                figure, tables = _placeholder(name, str(exc)), {}
            except Exception:
                for number in set(plt.get_fignums())-existing:
                    plt.close(number)
                raise
            try:
                entry["files"] = [path.name for path in save_figure(figure, directory/name, formats=formats, dpi=dpi)]
                entry["tables"] = []
                for key, data in tables.items():
                    path = directory/(key+".csv")
                    data.to_csv(path, index=False)
                    entry["tables"].append(path.name)
            finally:
                plt.close(figure)
        manifest["figures"].append(entry)
    manifest["sources"] = reports.sources
    manifest["reports"] = {name: reports.metadata(name) for name in reports.series}
    (directory/"figure_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest


__all__ = ["save_publication_figures"]
