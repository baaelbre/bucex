"""Publication panels for empirical cycles and within-month residual spread."""
from __future__ import annotations

from .style import PUBLICATION_COLORS, publication_style


def _styles(labels, colors, linestyles):
    if colors is None:
        palette = ("#707881", *PUBLICATION_COLORS)
        colors = [palette[i % len(palette)] for i in range(len(labels))]
    if linestyles is None:
        patterns = ("--", "-", "-.", ":")
        linestyles = [patterns[i % len(patterns)] for i in range(len(labels))]
    if len(colors) != len(labels) or len(linestyles) != len(labels):
        raise ValueError("Supply one colour and linestyle for each declared window.")
    return dict(zip(labels, zip(colors, linestyles)))


def _panel(report, kind, *, colors, linestyles, columns, figsize, units,
           style, dpi, title):
    import matplotlib.pyplot as plt

    if isinstance(columns, bool) or not isinstance(columns, int) or columns < 1:
        raise ValueError("columns must be a positive integer.")
    names = list(report.observations.columns)
    is_cycle = kind == "cycle"
    table = report.seasonal_cycles if is_cycle else report.monthly_spread
    window = "period" if is_cycle else "era"
    labels = list(report.metadata["periods" if is_cycle else "eras"])
    styles = _styles(labels, colors, linestyles)
    columns = min(columns, len(names))
    rows = (len(names)+columns-1)//columns
    unit_label = f" / {units}" if units else ""
    with publication_style(style=style, dpi=dpi):
        figure, axes = plt.subplots(rows, columns, squeeze=False, sharex=True,
            figsize=figsize or (10.6, 3.2*rows), layout="constrained")
        try:
            for i, (ax, name) in enumerate(zip(axes.flat, names)):
                for label in labels:
                    values = table.loc[(table.series == name) & (table[window] == label)].sort_values("month")
                    color, linestyle = styles[label]
                    if is_cycle:
                        ax.fill_between(values.month, values.q25, values.q75,
                                        color=color, alpha=.10, lw=0)
                    ax.plot(values.month, values["mean" if is_cycle else "residual_iqr"],
                            color=color, ls=linestyle, lw=1.8 if is_cycle else 1.7, label=label)
                ax.set_title(name, loc="left", weight="bold")
                ax.set_xticks([1, 3, 5, 7, 9, 11], ["Jan", "Mar", "May", "Jul", "Sep", "Nov"])
                if i % columns == 0:
                    ax.set_ylabel(("Monthly summary" if is_cycle else "Detrended interquartile range")+unit_label)
            for ax in list(axes.flat)[len(names):]:
                ax.set_visible(False)
            axes[0, 0].legend(loc="upper left" if is_cycle else "best", frameon=False,
                              fontsize=8.5 if is_cycle else 9)
            if title:
                figure.suptitle(title)
        except Exception:
            plt.close(figure)
            raise
    return figure


def plot_exploratory_cycles(report, *, colors=None, linestyles=None, columns=3,
                            figsize=None, units="", style="manuscript", dpi=200, title=None):
    """Plot observed means and empirical interquartile bands by calendar month.

    ``report`` is returned by ``explore_monthly``. Colours and linestyles follow
    its declared period order. Returns a Figure without changing global style.
    Set ``units='°C'`` for temperature. No figure title is added by default.
    """
    return _panel(report, "cycle", colors=colors, linestyles=linestyles, columns=columns,
                  figsize=figsize, units=units, style=style, dpi=dpi, title=title)


def plot_exploratory_spread(report, *, colors=None, linestyles=None, columns=3,
                            figsize=None, units="", style="manuscript", dpi=200, title=None):
    """Plot observed within-month residual IQRs after separate linear detrending.

    Colours and linestyles follow the declared era order. This panel is
    descriptive and does not represent posterior observation-scale estimates.
    """
    return _panel(report, "spread", colors=colors, linestyles=linestyles, columns=columns,
                  figsize=figsize, units=units, style=style, dpi=dpi, title=title)


__all__ = ["plot_exploratory_cycles", "plot_exploratory_spread"]
