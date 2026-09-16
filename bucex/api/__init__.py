"""Public high-level API.

All fitting routes enter through :func:`fit`; the named structural helpers are
small compatibility conveniences, not separate inference implementations.
"""
from .fit import (
    combine_fits,
    combine_fs_fits,
    fit,
    fit_bayes,
    fit_bulk_tail,
    fit_gaussian_structural,
    fit_gev_structural,
    make_gaussian_model,
    make_gev_model,
    plan,
)
from .predict import Forecast, posterior_predict, posterior_predictive
from .aggregate import AggregateForecast, aggregate_forecast

__all__ = [
    "fit",
    "fit_bayes",
    "fit_bulk_tail",
    "fit_gaussian_structural",
    "fit_gev_structural",
    "combine_fits",
    "combine_fs_fits",
    "make_gaussian_model",
    "make_gev_model",
    "plan",
    "Forecast",
    "AggregateForecast",
    "aggregate_forecast",
    "posterior_predict",
    "posterior_predictive",
]
