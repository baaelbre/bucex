"""Posterior predictive discrepancies for marginal tails and residual memory."""
import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import skew


def posterior_predictive_checks(fit, *, prediction=None, draws=200, seed=None, level=.95):
    """Compare conditional normal-score discrepancies with replicated scores.

    Each smoothed posterior draw transforms the observed data, then replicates
    that margin's iid N(0,1) observation scores under the stated model. These
    are descriptive posterior predictive checks, not uniform frequentist
    p-values. They complement held-out PITs and the cross-margin copula check.
    """
    from ..core.calendar import block_months, meteorological_phases
    if not 0<level<1:
        raise ValueError('level must be between zero and one.')
    prediction = fit.posterior_predictive(draws=draws, seed=seed) if prediction is None else prediction
    if prediction.horizon != fit.n_time or not np.array_equal(prediction.dates,fit.time):
        raise ValueError('Prediction must be a replication of this fitted observation window.')
    rng = np.random.default_rng(seed)
    dates = pd.DatetimeIndex(fit.time)
    phases = meteorological_phases(dates)
    phase_names = ('DJF','MAM','JJA','SON')
    rows = []
    names = fit.channel_names if fit.is_multiseries_model else (None,)
    lags = (1, 4 if block_months(dates)==3 else 12)
    for k,name in enumerate(names):
        observed = fit.observed[:,k] if fit.is_multiseries_model else fit.observed
        cdf = prediction.conditional_cdf(observed, channel=name)
        z = ndtri(np.clip(cdf,1e-12,1-1e-12))
        replicate = rng.normal(size=z.shape)
        groups = {'all':np.ones(fit.n_time,dtype=bool),
                  **{label:phases==i+1 for i,label in enumerate(phase_names)}}
        for group,mask in groups.items():
            if mask.sum()<4:
                continue
            statistics = {
                'mean':lambda x:np.mean(x,axis=1),
                'sd':lambda x:np.std(x,axis=1,ddof=1),
                'skewness':lambda x:skew(x,axis=1,bias=False),
                'lower_0.005':lambda x:np.mean(x<ndtri(.005),axis=1),
                'lower_0.025':lambda x:np.mean(x<ndtri(.025),axis=1),
                'upper_0.975':lambda x:np.mean(x>ndtri(.975),axis=1),
                'upper_0.995':lambda x:np.mean(x>ndtri(.995),axis=1)}
            if group=='all':
                for lag in lags:
                    if lag+3 < fit.n_time:
                        def acf(x, lag=lag):
                            a,b=x[:,:-lag],x[:,lag:]
                            a=a-a.mean(axis=1,keepdims=True);b=b-b.mean(axis=1,keepdims=True)
                            return (a*b).sum(axis=1)/np.sqrt((a*a).sum(axis=1)*(b*b).sum(axis=1))
                        statistics[f'lag_{lag}']=acf
            for statistic, function in statistics.items():
                actual, simulated = function(z[:,mask]),function(replicate[:,mask])
                lower,upper = np.quantile(simulated,[(1-level)/2,(1+level)/2])
                rows.append(dict(channel=name or fit.series_name or 'series',group=group,
                    statistic=statistic,n=int(mask.sum()),posterior_draws=len(z),
                    observed_mean=float(np.mean(actual)),replicated_lower=float(lower),
                    replicated_upper=float(upper),predictive_p_greater=float(np.mean(simulated>=actual)),
                    interpretation='Conditional score replication; descriptive in-sample PPC'))
    return pd.DataFrame(rows)


__all__=['posterior_predictive_checks']
