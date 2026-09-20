"""A local, reusable figure style; importing BUCEX never changes rcParams."""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from io import BytesIO
import os
from pathlib import Path
import tempfile


PUBLICATION_COLORS = ("#24658a", "#a44839", "#ba861f", "#53846c", "#78618c", "#555555")


@contextmanager
def publication_style(*, style="manuscript", primary=None, dpi=180, overrides=None):
    """Use the manuscript typography and palette inside a ``with`` block.

    ``style='default'`` respects the caller's Matplotlib settings. ``primary``
    changes the first colour, e.g. for a particular response. Settings are
    restored on exit, including on exceptions. Matplotlib remains optional.
    """
    import matplotlib as mpl
    from cycler import cycler

    if style not in {"manuscript", "default"}:
        raise ValueError("style must be 'manuscript' or 'default'.")
    if dpi <= 0:
        raise ValueError("dpi must be positive.")
    settings = {}
    if style == "manuscript":
        colors = list(PUBLICATION_COLORS)
        if primary is not None:
            colors = [primary] + [c for c in colors if c != primary]
        settings = {
            "font.family": "DejaVu Sans", "font.size": 12,
            "axes.labelsize": 12, "axes.titlesize": 13,
            "xtick.labelsize": 11, "ytick.labelsize": 11,
            "legend.fontsize": 10, "legend.frameon": False,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.prop_cycle": cycler(color=colors),
            "axes.axisbelow": True, "axes.grid": True, "axes.grid.axis": "y",
            "grid.alpha": .16, "grid.linewidth": .7,
            "lines.linewidth": 1.7, "lines.markersize": 4,
            "figure.facecolor": "white", "axes.facecolor": "white",
            "savefig.facecolor": "white", "savefig.dpi": dpi,
            "pdf.fonttype": 42, "ps.fonttype": 42,
        }
    settings.update(overrides or {})
    with mpl.rc_context(settings):
        yield


def styled_report(function):
    """Apply a research config's figure settings without global side effects."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        config = kwargs.get("config", {})
        fit = args[0] if args else None
        primary = config.get('figure_colors',{}).get(getattr(fit,'series_name',None))
        with publication_style(style=config.get("figure_style", "manuscript"),
                               dpi=config.get("figure_dpi", 180), primary=primary):
            return function(*args, **kwargs)
    return wrapped


def save_figure(figure, path, *, formats=None, dpi=180, close=False):
    """Save one figure consistently, optionally in several formats.

    Example: ``save_figure(fig, 'figures/levels', formats=('png', 'pdf'))``.
    No smoothing, resampling, cropping of data, or scientific transformation
    is applied. The returned paths identify the files actually written.
    """
    path = Path(path)
    formats = (path.suffix.lstrip(".") or "png",) if formats is None else tuple(formats)
    if not formats or any(fmt not in {"png", "pdf", "svg"} for fmt in formats):
        raise ValueError("formats must contain png, pdf, and/or svg.")
    if dpi <= 0:
        raise ValueError("dpi must be positive.")
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in dict.fromkeys(formats):
        target = path.with_suffix("." + fmt)
        # Finish the encoder before replacing an existing scientific figure.
        # A failed/interrupting render must not leave a half-written PDF/PNG.
        buffer = BytesIO()
        figure.savefig(buffer, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".bucex-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(buffer.getvalue())
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        written.append(target)
    if close:
        import matplotlib.pyplot as plt
        plt.close(figure)
    return written


__all__ = ["PUBLICATION_COLORS", "publication_style", "save_figure"]
