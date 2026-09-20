"""Reproducible publication figures from compact posterior report tables."""
from .data import ReportCollection
from .figures import save_publication_figures

__all__ = ["ReportCollection", "save_publication_figures"]
