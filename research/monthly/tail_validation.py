"""Reviewer comment 5: held-out central and directional tail evidence.

Every statistic counts forecast cases. Forecasts from overlapping origins are
correlated, so these counts must not be treated as independent replications.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import bucex as bx


def _stratify(cases, steps_per_year):
    cases = cases.copy()
    dates = pd.to_datetime(cases["time"], errors="raise")
    if dates.isna().any() or (cases["horizon"] < 1).any():
        raise ValueError("Each forecast case needs a date and a positive horizon.")
    season = dates.dt.month.map({12: "DJF", 1: "DJF", 2: "DJF",
                                 3: "MAM", 4: "MAM", 5: "MAM",
                                 6: "JJA", 7: "JJA", 8: "JJA",
                                 9: "SON", 10: "SON", 11: "SON"})
    year = "year_" + (((cases["horizon"].astype(int) - 1) // steps_per_year) + 1).astype(str)
    return pd.concat([cases.assign(stratum="overall", group="all"),
                      cases.assign(stratum="season", group=season.to_numpy()),
                      cases.assign(stratum="forecast_year", group=year.to_numpy())],
                     ignore_index=True)


def summarize_tail_validation(coverage_cases, threshold_cases, *, steps_per_year):
    """Return three tables with explicit denominators and expected event counts.

    Central 99% uses the 0.005/0.995 endpoints; the directional 99th
    quantile is a separate check with 1% nominal upper exceedance.
    """
    if steps_per_year < 1:
        raise ValueError("steps_per_year must be positive.")
    coverage = _stratify(coverage_cases, steps_per_year)
    intervals = coverage.loc[coverage.kind.eq("central_interval") &
                             coverage.nominal.isin((.90, .95, .99))].copy()
    if intervals.empty:
        raise ValueError("No central 90%, 95% or 99% forecast intervals found.")
    intervals["below"] = intervals.observed < intervals.lower
    intervals["above"] = intervals.observed > intervals.upper
    intervals["inside"] = ~intervals.below & ~intervals.above
    keys = ["channel", "stratum", "group", "nominal"]
    central = intervals.groupby(keys, sort=True).agg(
        n_cases=("observed", "size"), n_dates=("time", "nunique"),
        n_origins=("origin", "nunique"), below=("below", "sum"),
        inside=("inside", "sum"), above=("above", "sum"),
        mean_width=("width", "mean")).reset_index()
    central["empirical_coverage"] = central.inside / central.n_cases
    central["expected_misses"] = central.n_cases * (1 - central.nominal)
    central["expected_each_side"] = central.expected_misses / 2

    quantiles = coverage.loc[coverage.kind.eq("cdf_quantile") &
                             coverage.nominal.isin((.01, .05, .95, .99))].copy()
    if quantiles.empty:
        raise ValueError("No directional 1%, 5%, 95% or 99% forecast quantiles found.")
    quantiles["side"] = np.where(quantiles.nominal < .5, "lower", "upper")
    quantiles["event"] = np.where(quantiles.side.eq("lower"),
                                   quantiles.observed < quantiles["quantile"],
                                   quantiles.observed > quantiles["quantile"])
    quantiles["event_probability"] = np.where(quantiles.side.eq("lower"),
                                               quantiles.nominal, 1 - quantiles.nominal)
    tails = quantiles.groupby(keys + ["side"], sort=True).agg(
        n_cases=("event", "size"), n_dates=("time", "nunique"),
        n_origins=("origin", "nunique"), observed_events=("event", "sum"),
        nominal_event_rate=("event_probability", "first")).reset_index()
    tails["expected_events"] = tails.n_cases * tails.nominal_event_rate
    tails["empirical_event_rate"] = tails.observed_events / tails.n_cases

    risks = _stratify(threshold_cases, steps_per_year)
    if (risks.event_probability.isna().any() or
        (~risks.event_probability.between(0, 1)).any()):
        raise ValueError("Threshold probabilities must be finite and in [0, 1].")
    events = risks.groupby(["channel", "stratum", "group", "threshold", "direction"],
                           sort=True).agg(
        n_cases=("observed_event", "size"), n_dates=("time", "nunique"),
        n_origins=("origin", "nunique"), observed_events=("observed_event", "sum"),
        expected_events=("event_probability", "sum"),
        mean_brier=("brier", "mean"), mean_log_score=("log_score", "mean")
    ).reset_index()
    events["empirical_event_rate"] = events.observed_events / events.n_cases
    events["predicted_event_rate"] = events.expected_events / events.n_cases
    return {"central_coverage.csv": central,
            "directional_tails.csv": tails,
            "threshold_events.csv": events}


def write_tail_tables(directory, coverage_cases=None, threshold_cases=None, *, steps_per_year=None):
    """Write compact tables during a fit, or rebuild them from saved cases."""
    directory = Path(directory)
    if coverage_cases is None:
        coverage_cases = pd.read_csv(directory / "coverage_by_case.csv")
    if threshold_cases is None:
        threshold_cases = pd.read_csv(directory / "threshold_cases.csv")
    if steps_per_year is None:
        config = bx.load_config(directory.parent / "config.json")
        steps_per_year = int(config["model"]["steps_per_year"])
    tables = summarize_tail_validation(coverage_cases, threshold_cases,
                                       steps_per_year=steps_per_year)
    for name, frame in tables.items():
        frame.to_csv(directory / name, index=False)
    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="Saved validation channel directory, e.g. RUN/joint.")
    args = parser.parse_args()
    print("\n".join(str(args.run / name) for name in write_tail_tables(args.run)))
