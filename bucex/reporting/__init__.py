"""Reproducible publication figures from compact posterior report tables."""
from .data import ReportCollection
from .figures import save_publication_figures
from .sensitivity import SensitivityReport
from .shrinkage import save_shared_shrinkage_report

__all__ = ["ReportCollection", "save_publication_figures", "SensitivityReport", "save_shared_shrinkage_report"]
