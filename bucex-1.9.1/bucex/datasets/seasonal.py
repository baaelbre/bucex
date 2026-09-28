"""Complete meteorological seasons from daily records, with explicit provenance."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from .uccle import UCCLE_SERIES, _daily_frame, load_uccle_daily

SEASON_NAMES = {12: "DJF", 3: "MAM", 6: "JJA", 9: "SON"}


def complete_seasons(source=None, *, start=None, end=None, exclude_months=()):
    """Return daily data and a season audit; fail on missing interior data.

    Dates label block STARTS: DJF starts in December and is named for the
    following year. Exclusions are declared months, not fabricated missing
    observations. Only complete three-month blocks enter the analysis.
    """
    frame = _daily_frame(source) if isinstance(source, pd.DataFrame) else load_uccle_daily(source)
    provenance = dict(frame.attrs)
    daily = frame.set_index("DAY")
    if start is not None:
        daily = daily.loc[pd.Period(start, "M").start_time:]
    if end is not None:
        daily = daily.loc[:pd.Period(end, "M").end_time]
    if daily.empty:
        raise ValueError("No daily data remain in the requested range.")
    excluded = pd.PeriodIndex(list(exclude_months or ()), freq="M")
    first, last = daily.index[0], daily.index[-1]
    records, retained = [], []
    blocks = pd.period_range(first.to_period("Q-FEB"), last.to_period("Q-FEB"), freq="Q-FEB")
    for block in blocks:
        lo, hi = block.start_time, block.end_time.normalize()
        expected = pd.date_range(lo, hi, freq="D")
        months = pd.period_range(lo, hi, freq="M")
        explicit = months.intersection(excluded)
        boundary = lo < first or hi > last
        values = daily.reindex(expected)[["TX", "TN"]]
        complete = not boundary and not len(explicit) and values.notna().all().all()
        reason = ("requested_month_exclusion" if len(explicit) else
                  "incomplete_boundary" if boundary else "complete" if complete else "missing_daily_data")
        if reason == "missing_daily_data":
            raise ValueError(f"Missing daily dates or temperatures inside season starting {lo:%Y-%m}.")
        records.append(dict(start=lo, end=hi, season=SEASON_NAMES[lo.month],
            season_year=lo.year + (lo.month == 12), expected_days=len(expected),
            observed_days=int(values.notna().all(axis=1).sum()), retained=bool(complete),
            reason=reason, excluded_months=",".join(map(str, explicit))))
        if complete:
            retained.extend(expected)
    if not retained:
        raise ValueError("No complete meteorological season remains.")
    selected = daily.loc[pd.DatetimeIndex(retained)].copy()
    selected.attrs = provenance
    audit = pd.DataFrame(records)
    kept = audit.loc[audit.retained, "start"]
    if len(kept) > 1 and not np.all(np.diff(pd.DatetimeIndex(kept).to_period("M").asi8) == 3):
        raise ValueError("Season exclusions create an internal gap; use a consecutive analysis window.")
    return selected, audit


def derive_uccle_seasonal(source=None, *, start=None, end=None, exclude_months=(), output_dir=None):
    """Six daily-weighted means/extrema per complete DJF/MAM/JJA/SON block.

    Means average DAILY TX/TN, not three equally weighted monthly means.
    All six summaries use the same complete days. Raw data are never edited.
    """
    daily, audit = complete_seasons(source, start=start, end=end, exclude_months=exclude_months)
    group = daily.groupby(daily.index.to_period("Q-FEB"))
    result = pd.DataFrame(dict(TXm=group.TX.mean(), TNm=group.TN.mean(),
        TXx=group.TX.max(), TXn=group.TX.min(), TNx=group.TN.max(), TNn=group.TN.min()))
    result.index = result.index.start_time
    result.index.name = "date"
    last = audit.loc[audit.retained].iloc[-1]
    result.attrs = {**daily.attrs, "frequency": "seasonal", "seasonal_period": 4,
        "n_blocks": len(result), "date_convention": "season start; DJF named by ending year",
        "first_block": str(result.index[0].date()), "last_block": str(result.index[-1].date()),
        "last_included_day": str(last.end.date()),
        "requested_month_exclusions": list(map(str, exclude_months or ())),
        "block_audit": json.loads(audit.to_json(orient="records", date_format="iso")),
        "daily_TN_above_TX": int((daily.TN > daily.TX).sum())}
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for name in UCCLE_SERIES:
            result[[name]].to_csv(output/f"{name}.csv", date_format="%Y-%m-%d")
        audit.to_csv(output/"block_audit.csv", index=False)
        (output/"quality_report.json").write_text(json.dumps(result.attrs, indent=2)+"\n")
    return result


__all__ = ["SEASON_NAMES", "complete_seasons", "derive_uccle_seasonal"]
