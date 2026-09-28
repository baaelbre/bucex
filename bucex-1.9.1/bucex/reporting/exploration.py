"""Save descriptive exploration with its source observations and definitions."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from ..__about__ import __version__
from ..config import save_config
from ..plotting.style import save_figure


def save_exploration_report(report, directory, *, figures=True, formats=("png", "pdf"),
                            dpi=200, style="manuscript", columns=3, figsize=None, units="",
                            period_colors=None, period_linestyles=None, era_colors=None,
                            era_linestyles=None, titles=None):
    """Save manuscript-compatible figure names, CSV tables and provenance.

    The directory contains ``figures/``, ``data/`` and
    ``exploration_metadata.json``. Figure names are
    ``exploratory_seasonal_cycles`` and ``exploratory_monthly_spread``. CSVs
    contain the plotted values, observation counts and missing-value counts.
    ``data/monthly_observations.csv`` is a reproducible input snapshot, not a
    detrended dataset for fitting. Use ``figures=False`` for a tables-only
    export without importing Matplotlib. Existing files at this location are
    replaced; the SERRA entry point creates a fresh timestamped directory.
    """
    directory = Path(directory)
    (directory/"data").mkdir(parents=True, exist_ok=True)
    report.observations.to_csv(directory/"data"/"monthly_observations.csv",
                               date_format="%Y-%m-%d", float_format="%.17g")
    tables = {"exploratory_seasonal_cycles": report.seasonal_cycles,
              "exploratory_monthly_spread": report.monthly_spread}
    for name, table in tables.items():
        table.to_csv(directory/"data"/(name+".csv"), index=False)
    metadata = deepcopy(report.metadata)
    metadata.update(bucex_version=__version__, figures=[],
                    figure_settings=dict(formats=list(formats), dpi=dpi, style=style,
                        columns=columns, figsize=figsize, units=units, period_colors=period_colors,
                        period_linestyles=period_linestyles, era_colors=era_colors,
                        era_linestyles=era_linestyles, titles=titles))
    if figures:
        import matplotlib.pyplot as plt
        builders = [("exploratory_seasonal_cycles", report.plot_cycles, period_colors, period_linestyles),
                    ("exploratory_monthly_spread", report.plot_spread, era_colors, era_linestyles)]
        for name, plot, colors, linestyles in builders:
            figure = plot(colors=colors, linestyles=linestyles, columns=columns, figsize=figsize,
                          units=units, style=style, dpi=dpi, title=(titles or {}).get(name))
            try:
                paths = save_figure(figure, directory/"figures"/name, formats=formats, dpi=dpi)
                metadata["figures"].append(dict(name=name,
                    files=[str(path.relative_to(directory)) for path in paths],
                    table="data/"+name+".csv"))
            finally:
                plt.close(figure)
    save_config(metadata, directory/"exploration_metadata.json")
    (directory/"README.md").write_text(
        "# Monthly data exploration\n\n"
        "Figure 1: observed monthly means and empirical interquartile bands.\n"
        "Figure 2: within-month residual IQR after separate linear detrending in each era.\n\n"
        "These are descriptive summaries, not posterior results or hypothesis tests.\n"
        "All plotted values and sample counts are in data/. The source snapshot retains\n"
        "the original observations. Periods, methods and its SHA-256 are recorded in\n"
        "exploration_metadata.json. No BUCEX model was fitted.\n",
        encoding="utf-8")
    return metadata
