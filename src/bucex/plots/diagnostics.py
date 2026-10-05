import numpy as np
from scipy.stats import norm
from ..results import ChannelResult


def diagnostic(c, ax, *, type, color, max_lag=20, **kwargs):
    from ..diagnostics import pit, normal_scores
    if not isinstance(c, ChannelResult):
        raise TypeError('Residual plots require fitted observations.')
    if type == 'pit':
        u = pit(c).mean(axis=(0, 1))
        ax.hist(u, bins=np.linspace(0, 1, 11), density=True, color=color, alpha=.6)
        ax.axhline(1, color='.3', linestyle='--', lw=1)
        ax.set(xlabel='Posterior mean conditional PIT', ylabel='Density')
        return
    z = normal_scores(c)
    if type == 'normal_qq':
        theoretical = norm.ppf((np.arange(len(z))+.5)/len(z))
        ax.plot(theoretical, np.sort(z), '.', color=color, ms=3, label=kwargs.get('label'))
        ax.plot([theoretical[0], theoretical[-1]], [theoretical[0], theoretical[-1]], '--', color='.4', lw=1)
        ax.set(xlabel='Standard normal quantile', ylabel='Conditional normal-score quantile')
    else:
        z = z-z.mean()
        if z@z == 0:
            raise ValueError('ACF is undefined for constant residuals.')
        lags = np.arange(min(max_lag, len(z)-1)+1)
        values = [1.]+[np.dot(z[:-lag], z[lag:])/np.dot(z, z) for lag in lags[1:]]
        ax.vlines(lags, 0, values, color=color)
        ax.axhline(0, color='.5', lw=.7)
        ax.set(xlabel='Lag (blocks)', ylabel='Score residual correlation')
