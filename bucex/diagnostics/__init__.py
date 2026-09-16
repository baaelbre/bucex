"""Posterior, predictive and calibration diagnostics."""
from .comparison import paired_block_comparison
from .calibration import PITResult, empirical_coverage, pit_diagnostics
from .contrasts import summarize_draws
from .cv import LFOResult, leave_future_out, rolling_origin_splits
from .experiments import (
    annual_aggregation_check,
    compare_innovation_priors,
    draw_structural_prior,
    forecast_uncertainty,
    innovation_prior_variant,
    prior_predictive_targets,
    recovery_metrics,
    scientific_summary,
)
from .posterior import ess_bulk, ess_tail, fit_diagnostics, posterior_pit, rhat
from .residuals import one_step_ahead_residuals
from .dependence import residual_dependence_check, residual_serial_check
from .ordering import OrderingResult, ordering_diagnostics, compound_event_probability
from .scores import (
    crps_ensemble,
    evaluate_ensemble,
    exceedance_brier_score,
    exceedance_log_score,
    log_predictive_score,
    quantile_score,
    threshold_weighted_crps,
)

__all__ = [
    "paired_block_comparison",
    "residual_dependence_check",
    "residual_serial_check", "ess_tail",
    "OrderingResult", "ordering_diagnostics", "compound_event_probability",
    "summarize_draws",
    "rhat",
    "ess_bulk",
    "posterior_pit",
    "fit_diagnostics",
    "crps_ensemble",
    "threshold_weighted_crps",
    "quantile_score",
    "exceedance_brier_score",
    "exceedance_log_score",
    "evaluate_ensemble",
    "log_predictive_score",
    "empirical_coverage",
    "PITResult",
    "pit_diagnostics",
    "one_step_ahead_residuals",
    "rolling_origin_splits",
    "LFOResult",
    "leave_future_out",
    "annual_aggregation_check",
    "compare_innovation_priors",
    "draw_structural_prior",
    "forecast_uncertainty",
    "innovation_prior_variant",
    "prior_predictive_targets",
    "recovery_metrics",
    "scientific_summary",
]

from .periods import period_average, period_contrasts, convergence_assessment
__all__ += ["period_average", "period_contrasts", "convergence_assessment"]
