"""Calendar-period estimands with intact posterior pairing and chain labels."""
import numpy as np
import pandas as pd


def period_average(draws, dates, period, *, month=None, require_complete=True):
    """Average monthly posterior paths within an inclusive calendar period.

    ``draws[..., time]`` may retain chains. This is an equally weighted average
    of monthly latent states, not a day-weighted annual observed temperature.
    Missing requested months raise an error by default; comparisons must not
    silently change their estimand when a fit ends early.
    """
    values = np.asarray(draws)
    index = pd.DatetimeIndex(dates).to_period("M")
    if values.shape[-1] != len(index) or index.has_duplicates:
        raise ValueError("Draws and unique monthly dates must align.")
    if len(period) != 2:
        raise ValueError("period is an inclusive (start, end) pair.")
    start, end = (pd.Period(v, freq="M") for v in period)
    if start > end:
        raise ValueError("Period start must precede its end.")
    expected = pd.period_range(start, end, freq="M")
    # Seasonal blocks are timestamped at their first month. Period boundaries
    # refer to complete blocks; do not average a season extending past 'end'.
    delta = np.diff(index.asi8)
    if len(delta) and np.all(delta == 3) and np.all(np.isin(index.month,[3,6,9,12])):
        expected = expected[np.isin(expected.month,[3,6,9,12]) & (expected+2 <= end)]
        if require_complete and ((start.month not in {3,6,9,12}) or end.month not in {2,5,8,11}):
            raise ValueError("Seasonal period contrasts require complete meteorological block boundaries.")
    if month is not None:
        if int(month) != month or not 1 <= month <= 12:
            raise ValueError("month must be an integer in 1..12.")
        expected = expected[expected.month == month]
    if not len(expected):
        raise ValueError("No requested months fall within this period.")
    if require_complete and not expected.isin(index).all():
        raise ValueError(f"Fit does not contain the complete requested period {period}.")
    mask = index.isin(expected)
    if not np.any(mask):
        raise ValueError("No fitted observations in the requested period.")
    return values[...,mask].mean(axis=-1)


def period_contrasts(fit, reference, comparison, *, pairs=(), months=(), rate_multiplier=120.):
    """Changes in level, location and rate; paired cross-series change contrasts.

    The same posterior draw supplies both periods and both series. Cross-series
    contrasts require a joint fit; unrelated marginal draws are never paired.
    Rates are averages of the latent slope, not differences of noisy levels.
    """
    output = {}
    names = fit.channel_names if fit.is_multiseries_model else (None,)
    changes = {}
    for name in names:
        label = name or fit.series_name or "series"
        paths = {"level": fit.component_draws("level",channel=name,combine_chains=False),
                 "location": fit.parameter_path("mu",channel=name,combine_chains=False),
                 "slope": fit.component_draws("slope",channel=name,combine_chains=False)*rate_multiplier}
        for quantity, values in paths.items():
            before = period_average(values, fit.time, reference)
            after = period_average(values, fit.time, comparison)
            output[f"{label}.{quantity}.reference"] = before
            output[f"{label}.{quantity}.comparison"] = after
            output[f"{label}.{quantity}.change"] = after-before
            changes[label,quantity] = after-before
        for month in months:
            values = paths["location"]
            output[f"{label}.location.month_{month:02d}.change"] = (
                period_average(values,fit.time,comparison,month=month)
                -period_average(values,fit.time,reference,month=month))
    if pairs and not fit.is_multiseries_model:
        raise ValueError("Paired cross-series contrasts require a joint fit.")
    for left,right in pairs:
        if left not in names or right not in names:
            raise ValueError(f"Unknown contrast pair {(left,right)}.")
        for quantity in ("level", "location", "slope"):
            output[f"{left}_minus_{right}.{quantity}.change"] = changes[left,quantity]-changes[right,quantity]
    return output


def convergence_assessment(tables, *, max_rhat=1.01, min_ess=400, min_chains=4):
    """Flag numerical diagnostics; passing does not establish model adequacy.

    Pass named tables from fit.diagnostics()['parameters'] or summarize_draws.
    Constants/undefined diagnostics remain visible and need explicit review.
    """
    issues = []
    checked = 0
    for label,table in tables.items():
        for name,row in table.iterrows():
            checked += 1
            reasons = []
            if "chains" in row and row["chains"] < min_chains:
                reasons.append("too few independent chains")
            for metric in ("rhat", "ess_bulk", "ess_tail"):
                value = row.get(metric, np.nan)
                if not np.isfinite(value):
                    reasons.append(metric+" undefined")
                elif (metric == "rhat" and value > max_rhat) or (metric != "rhat" and value < min_ess):
                    reasons.append(metric+" outside threshold")
            if reasons:
                issues.append({"table": label, "quantity": str(name), "reasons": reasons})
    return {"status": "passed_numerical_checks" if checked and not issues else "needs_review",
            "checked": checked, "max_rhat": max_rhat, "min_ess": min_ess,
            "min_chains": min_chains, "issues": issues,
            "interpretation": "Numerical screening only; assess traces, Monte Carlo precision, predictive fit and sensitivity."}


__all__ = ["period_average", "period_contrasts", "convergence_assessment"]
