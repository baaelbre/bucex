"""Plots for the unified fit and forecast objects."""
from .style import PUBLICATION_COLORS, publication_style, save_figure
from .exploration import plot_exploratory_cycles, plot_exploratory_spread
from .calendar import plot_pit_calendar, plot_monthly_score_sd
from .traces import parameter_trace_draws, trace_frame, traces_from_frame
from .dependence import plot_copula, plot_ordering
from .report import save_prediction_report
from .predictive import (plot_predictive_diagnostics, plot_pit_diagnostics, plot_chain_traces,
    plot_forecast_months, plot_scale_calendar, plot_calendar_risk_curves)
from .core import (
    loess_smooth,
    plot,
    plot_bulk_tail,
    plot_collection,
    plot_endpoint,
    plot_fit,
    plot_level,
    plot_level_slope,
    plot_parameter_densities,
    plot_parameter_acfs,
    plot_predictor,
    plot_process_sds,
    plot_process_sd_traces,
    plot_risk,
    plot_season,
    plot_seasonal_patterns,
    plot_slope,
    plot_state,
)

__all__ = [
    "PUBLICATION_COLORS", "publication_style", "save_figure",
    "plot_exploratory_cycles", "plot_exploratory_spread",
    "plot_pit_calendar", "plot_monthly_score_sd",
    "parameter_trace_draws", "trace_frame", "traces_from_frame", "plot_pit_diagnostics",
    "save_prediction_report",
    "plot_predictive_diagnostics", "plot_chain_traces", "plot_forecast_months",
    "plot_scale_calendar", "plot_calendar_risk_curves",
    "plot_copula", "plot_ordering",
    "loess_smooth",
    "plot",
    "plot_fit",
    "plot_parameter_densities",
    "plot_parameter_acfs",
    "plot_predictor",
    "plot_process_sd_traces",
    "plot_state",
    "plot_level",
    "plot_slope",
    "plot_level_slope",
    "plot_process_sds",
    "plot_endpoint",
    "plot_risk",
    "plot_season",
    "plot_seasonal_patterns",
    "plot_bulk_tail",
    "plot_collection",
]

from .parameters import plot_parameter_path
__all__.append("plot_parameter_path")
