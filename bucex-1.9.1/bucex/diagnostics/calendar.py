"""Descriptive calendar diagnostics, with denominators and boundary counts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtri


def pit_normal_scores(values, *, clip=1e-10):
    """Validate PITs, then clip only for finite plotting normal scores."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("PIT values must be a non-empty, finite, one-dimensional array.")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("PIT values must lie in [0, 1].")
    if not 0 < clip < .5:
        raise ValueError("clip must lie in (0, .5).")
    return ndtri(np.clip(values, clip, 1-clip))


def pit_by_month(values, dates, *, clip=1e-10):
    """Summarize PIT and normal-score dispersion for each calendar month.

    ``n`` counts forecast cases; ``n_dates`` records distinct dates when
    rolling forecast windows overlap. No independence-based test or coverage
    claim is made. Exact zero/one PIT counts are retained before clipping.
    """
    values = np.asarray(values, dtype=float)
    scores = pit_normal_scores(values, clip=clip)
    dates = pd.DatetimeIndex(dates)
    if len(dates) != len(values) or dates.hasnans:
        raise ValueError("dates must contain one valid date per PIT.")
    data = pd.DataFrame({"time": dates, "month": dates.month,
                         "pit": values, "normal_score": scores})
    rows = []
    for month, group in data.groupby("month", sort=True):
        p, z = group.pit.to_numpy(), group.normal_score.to_numpy()
        rows.append(dict(month=int(month), n=len(group), n_dates=group.time.nunique(),
                         pit_mean=float(p.mean()), normal_score_mean=float(z.mean()),
                         normal_score_sd=float(z.std(ddof=1)) if len(z)>1 else np.nan,
                         pit_zero=int(np.sum(p == 0)), pit_one=int(np.sum(p == 1)),
                         below_005=int(np.sum(p < .005)), below_025=int(np.sum(p < .025)),
                         above_975=int(np.sum(p > .975)), above_995=int(np.sum(p > .995)),
                         normal_score_clip=clip))
    return pd.DataFrame(rows)


def coverage_by_month(cases):
    """Summarize held-out central coverage and directional quantile counts.

    Accept the case table returned by research validation. ``covered`` means
    interval inclusion for central intervals and ``y <= q`` for CDF quantiles.
    It is never relabelled as a tail-event count without its ``kind``/``nominal``.
    """
    required = {"time", "kind", "nominal", "covered"}
    if not required <= set(cases):
        raise ValueError(f"Coverage cases require {sorted(required)}.")
    data = cases.copy()
    dates = pd.to_datetime(data.time)
    if dates.isna().any() or not data.covered.isin([True, False, 0, 1]).all():
        raise ValueError("Coverage cases need valid dates and Boolean covered values.")
    data["month"] = dates.dt.month
    data["time"] = dates
    keys = (["channel"] if "channel" in data else []) + ["month", "kind", "nominal"]
    return data.groupby(keys, dropna=False).agg(
        empirical=("covered", "mean"), n=("covered", "size"),
        n_covered=("covered", "sum"), n_dates=("time", "nunique")
    ).reset_index()


__all__ = ["pit_normal_scores", "pit_by_month", "coverage_by_month"]
