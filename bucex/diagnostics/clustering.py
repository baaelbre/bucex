"""Daily extreme ranks and runs diagnostics; no independence is assumed by extraction."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _daily_series(values):
    if not isinstance(values, pd.Series):
        raise TypeError('Provide a daily pandas Series with dates.')
    index = pd.DatetimeIndex(values.index)
    if (not len(index) or index.hasnans or index.has_duplicates
            or not index.equals(index.normalize()) or not index.is_monotonic_increasing
            or not np.all(np.diff(index.asi8) == pd.Timedelta(days=1).value)
            or not np.all(np.isfinite(values))):
        raise ValueError('Daily observations must be finite, consecutive, unique and sorted.')
    return pd.Series(values.to_numpy(float), index=index, name=values.name)


def ranked_extremes(values, *, r=3, tail='upper'):
    """Raw r largest/smallest DAILY observations in each complete season.

    Ties break by earlier date and are flagged, not jittered. This is an
    extraction/diagnostic API; it does not fit r independent GEV observations.
    """
    values = _daily_series(values)
    if int(r)!=r or r<1 or tail not in {'upper','lower'}:
        raise ValueError('Use integer r>=1 and tail upper or lower.')
    rows = []
    for block, x in values.groupby(values.index.to_period('Q-FEB')):
        if len(x) != len(pd.date_range(block.start_time, block.end_time.normalize(),freq='D')):
            continue
        if len(x) < r+1:
            raise ValueError('r must leave at least one lower-ranked observation for checking ties.')
        ordered = x.sort_values(ascending=tail=='lower', kind='stable')
        boundary_tie = ordered.iloc[r-1] == ordered.iloc[r]
        for rank, (date,value) in enumerate(ordered.iloc[:r].items(),1):
            rows.append(dict(block_start=block.start_time, block_end=block.end_time.normalize(),
                rank=rank, date=date, value=value, tail=tail, boundary_tie=bool(boundary_tie)))
    return pd.DataFrame(rows, columns=['block_start','block_end','rank','date','value','tail','boundary_tie'])


def rank_clustering_diagnostics(ranks):
    """Proximity summaries, not a test that nearby ranks are independent events."""
    rows = []
    for block, group in ranks.groupby('block_start',sort=True):
        dates = pd.DatetimeIndex(group.date).sort_values()
        gaps = np.diff(dates.asi8)/pd.Timedelta(days=1).value
        rows.append(dict(block_start=block, r=len(group),
            minimum_gap_days=float(gaps.min()) if len(gaps) else np.nan,
            any_consecutive_days=bool(np.any(gaps==1)),
            any_gap_at_most_three_days=bool(np.any(gaps<=3)),
            span_days=int((dates[-1]-dates[0]).days),
            same_calendar_month=len(dates.to_period('M').unique())==1,
            boundary_tie=bool(group.boundary_tie.any())))
    return pd.DataFrame(rows)


def extreme_clusters(values, threshold, *, run_length=3, tail='upper'):
    """Runs clusters separated by at least run_length non-exceeding days.

    Threshold may be a scalar or an exactly aligned daily Series. Clustering
    occurs BEFORE assigning peaks to seasons, so a boundary-crossing episode
    is counted once, in its peak's season. This does not estimate a GEV or
    assert that a chosen separation removes all dependence.
    """
    values = _daily_series(values)
    if int(run_length)!=run_length or run_length<1 or tail not in {'upper','lower'}:
        raise ValueError('Use integer run_length>=1 and tail upper or lower.')
    if isinstance(threshold,pd.Series) and not threshold.index.equals(values.index):
        raise ValueError('Daily thresholds must align exactly with observations.')
    cut = np.broadcast_to(np.asarray(threshold,float),values.shape)
    if not np.all(np.isfinite(cut)):
        raise ValueError('Thresholds must be finite.')
    mask = values.to_numpy()>cut if tail=='upper' else values.to_numpy()<cut
    hits = np.flatnonzero(mask)
    groups = np.split(hits, np.flatnonzero(np.diff(hits)>run_length)+1) if len(hits) else []
    rows = []
    for idx in groups:
        part = values.iloc[idx]
        peak = part.idxmax() if tail=='upper' else part.idxmin()
        rows.append(dict(start=part.index[0],end=part.index[-1],peak_date=peak,
            peak=float(values.loc[peak]),n_exceedances=len(idx),
            duration_days=(part.index[-1]-part.index[0]).days+1,
            block_start=peak.to_period('Q-FEB').start_time,
            crosses_season=part.index[0].to_period('Q-FEB') != part.index[-1].to_period('Q-FEB'),
            run_length=int(run_length),tail=tail))
    return pd.DataFrame(rows,columns=['start','end','peak_date','peak','n_exceedances',
        'duration_days','block_start','crosses_season','run_length','tail'])


__all__ = ['ranked_extremes','rank_clustering_diagnostics','extreme_clusters']
