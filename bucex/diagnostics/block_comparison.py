"""Forecast assessment on common physical blocks, regardless of fitting frequency."""
import numpy as np
import pandas as pd


def score_seasonal_forecast(forecast, observed, *, channel=None, threshold=None, level=.95):
    """Score monthly forecasts aggregated to seasons, or direct seasonal forecasts.

    observed is a Series of complete daily-derived seasonal summaries indexed
    by season START. All forecast seasons must have an observed counterpart.
    Predictive CDF/density integrate observation noise before averaging over
    parameter and future-state draws. The returned scores are marginal, not
    a joint density for the vector of six seasonal summaries.
    """
    if not isinstance(observed,pd.Series) or observed.index.has_duplicates:
        raise ValueError('observed must be a uniquely dated seasonal Series.')
    aggregate=forecast.aggregate(frequency='season',channel=channel)
    if not aggregate.n_periods:
        raise ValueError('Forecast contains no complete season.')
    dates=pd.DatetimeIndex(aggregate.periods.start)
    if not dates.isin(observed.index).all():
        raise ValueError('Observed summaries do not cover every forecast season.')
    y=observed.loc[dates].to_numpy(float)
    scores=aggregate.score(y,thresholds=() if threshold is None else (threshold,))
    periods=aggregate.periods.reset_index(drop=True).copy()
    # Legacy monthly aggregation labels its last monthly block by its start.
    # Common-target tables always report the actual inclusive season end.
    periods['end']=pd.DatetimeIndex(periods.start)+pd.offsets.MonthEnd(3)
    indices=scores.time_index.to_numpy(int)
    scores=scores.assign(time=dates[indices],season=periods.window.to_numpy()[indices],
        horizon_seasons=indices+1,observed=y[indices])
    q=np.quantile(aggregate.observations,[(1-level)/2,.5,(1+level)/2],axis=0)
    calibration=periods.assign(observed=y,pit=aggregate.pit(y),lower=q[0],median=q[1],upper=q[2],
        nominal=level,covered=(y>=q[0])&(y<=q[2]),width=q[2]-q[0],horizon_seasons=np.arange(len(y))+1)
    if threshold is not None:
        direction='<' if aggregate.tail=='lower' else '>'
        probability=aggregate.probability_draws(threshold,direction=direction).mean(axis=0)
        event=y<threshold if direction=='<' else y>threshold
        # Analytic probability avoids zero-event artifacts from a small
        # predictive ensemble. No clipping except the event log-score boundary.
        for name,values in [('exceedance_brier',(probability-event)**2),
                            ('exceedance_log',-np.where(event,np.log(np.clip(probability,1e-12,1)),
                                                          np.log(np.clip(1-probability,1e-12,1))))]:
            scores.loc[scores.score.eq(name),'value']=values
        calibration=calibration.assign(threshold=threshold,direction=direction,
            event_probability=probability,event=event)
    return scores,calibration


__all__=['score_seasonal_forecast']
