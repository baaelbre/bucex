"""Native object-aware plots. No titles by default; all plots accept an axis.

Matplotlib is imported lazily. Posterior central curves are means, with
pointwise central credible/predictive intervals, not simultaneous bands.
"""
from pathlib import Path
import numpy as np
from scipy.stats import norm, gaussian_kde
from .results import FitResult, ChannelResult, summarize
from .prediction import Predictive

COLORS = {'TX': '#24658a', 'TN': '#a44839'}


def _axis(ax):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.4), constrained_layout=True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(direction='out')
    return ax


def _band(ax, index, values, *, color, interval, label=None, axis=(0, 1)):
    s = summarize(values, interval, axis=axis)
    ax.fill_between(index, s['lower'], s['upper'], color=color, alpha=.18, linewidth=0)
    ax.plot(index, s['mean'], color=color, lw=1.5, label=label)


def _density(ax, values, *, color, label, linestyle='-'):
    v = np.asarray(values).ravel()
    v = v[np.isfinite(v)]
    if v.size < 2 or np.ptp(v) == 0:
        ax.axvline(v[0], color=color, label=label, linestyle=linestyle)
        return
    lo, hi = np.quantile(v, [.001, .999])
    x = np.linspace(lo, hi, 250)
    ax.plot(x, gaussian_kde(v)(x), color=color, label=label, linestyle=linestyle)


def plot(value, type='level', *, channel=None, ax=None, interval=.95, color=None,
         label=None, parameter=None, threshold=None, tail='upper', years=20,
         phase=None, phase_labels=None, rate_scale=None, history=None, observed=None,
         max_lag=20, prior=None, path=None, dpi=180):
    """Plot a FitResult, ChannelResult or Predictive object; return the Axes.

    Types: level, slope, location, seasonal, cycle, normal_qq, pit, acf,
    trace, posterior, prior_posterior, risk, return_level, forecast.
    slope uses units per decade by default (10*steps_per_year).
    phase is a zero-based phase relative to the first fitted observation.
    For forecast phase selection, use calendar filtering in the research layer.
    """
    ax = _axis(ax)
    fit = value if isinstance(value, FitResult) else None
    c = fit.channel(channel) if fit else value
    if isinstance(c, Predictive):
        name = c.channel_name(channel)
    elif isinstance(c, ChannelResult):
        name = c.name
    else:
        raise TypeError('plot accepts a FitResult, ChannelResult or Predictive object.')
    color = color or COLORS.get(name[:2], '#24658a')
    if isinstance(c, Predictive):
        if type not in {'forecast', 'level', 'slope', 'location', 'seasonal', 'risk', 'return_level'}:
            raise ValueError('Unsupported plot type for a predictive object.')
        if type == 'forecast':
            if history is not None:
                h = history.channel(name) if isinstance(history, FitResult) else history
                ax.plot(h.index, h.y, '.', color='.65', ms=3, label='Observed')
            _band(ax, c.index, c.y[name], color=color, interval=interval, axis=0, label='Predictive mean')
            ax.plot(c.index, c.location[name].mean(axis=0), '--', color='.2', lw=1, label='Location')
            if observed is not None:
                ax.plot(c.index, observed, '.', color='black', ms=4, label='Held out')
            ax.set_ylabel('Response')
        else:
            values = (c.risk(threshold, channel=name, tail=tail) if type == 'risk' else
                      c.return_level(years, channel=name, tail=tail) if type == 'return_level' else getattr(c, type)[name])
            if type == 'slope':
                values = values*(10*c.steps_per_year if rate_scale is None else rate_scale)
            _band(ax, c.index, values, color=color, interval=interval, axis=0, label=label)
            ax.set_ylabel('Probability' if type == 'risk' else type.replace('_', ' ').capitalize())
    elif type in {'level', 'slope', 'location', 'seasonal', 'risk', 'return_level'}:
        if type == 'risk' and threshold is None:
            raise ValueError('A risk plot requires threshold=...')
        values = (c.risk(threshold, tail=tail) if type == 'risk' else
                  c.return_level(years, tail=tail) if type == 'return_level' else c.path(type))
        if type == 'risk' and threshold is None:
            raise ValueError('A risk plot requires threshold=...')
        if type == 'slope':
            values = values*(10*c.steps_per_year if rate_scale is None else rate_scale)
        ids = np.arange(len(c.y))
        if phase is not None:
            if isinstance(phase, bool) or not isinstance(phase, int) or not 0 <= phase < c.model.period:
                raise ValueError('phase must be an integer in [0, period).')
            ids = ids[ids % c.model.period == phase]
        _band(ax, np.asarray(c.index)[ids], values[:, :, ids], color=color, interval=interval, label=label)
        ax.set_ylabel('Probability' if type == 'risk' else 'Rate per decade' if type == 'slope' and rate_scale is None else type.replace('_', ' ').capitalize())
        if type == 'risk':
            ax.set_ylim(0, 1)
    elif type == 'cycle':
        if not c.model.season:
            raise ValueError('A cycle plot requires a seasonal component.')
        from .inference._state import static_seasonal_design
        from .inference._model import layout_for
        layout = layout_for(c.model)
        p = c.model.period
        initial = np.stack([c.parameters[f'initial_seasonal[{i}]'] for i in range(p)], axis=-1)
        # Last observed state is expressed relative to the last phase. Rotate
        # its deterministic continuation back to the first-record phase.
        last = c.states[:, :, -1, layout.season_slice]
        future = np.einsum('ij,cdj->cdi', static_seasonal_design(p, p-1), last)
        final = np.roll(future, (len(c.y)-1) % p, axis=-1)
        _band(ax, np.arange(p), initial, color='.5', interval=interval, label='Initial cycle')
        _band(ax, np.arange(p), final, color=color, interval=interval, label='Final cycle')
        ax.set_xticks(np.arange(p), phase_labels or [str(i+1) for i in range(p)])
        ax.set_ylabel('Seasonal component')
    elif type in {'normal_qq', 'pit', 'acf'}:
        from .diagnostics import pit, normal_scores
        if type == 'pit':
            u = pit(c).mean(axis=(0, 1))
            ax.hist(u, bins=np.linspace(0, 1, 11), density=True, color=color, alpha=.6)
            ax.axhline(1, color='.3', linestyle='--', lw=1)
            ax.set(xlabel='Posterior mean conditional PIT', ylabel='Density')
        else:
            z = normal_scores(c)
            if type == 'normal_qq':
                theoretical = norm.ppf((np.arange(len(z))+.5)/len(z))
                ax.plot(theoretical, np.sort(z), '.', color=color, ms=3)
                ax.plot([theoretical[0], theoretical[-1]], [theoretical[0], theoretical[-1]], '--', color='.4', lw=1)
                ax.set(xlabel='Standard normal quantile', ylabel='Conditional normal-score quantile')
            else:
                z = z-z.mean()
                if z@z == 0:
                    raise ValueError('ACF is undefined for constant residuals.')
                lags = np.arange(min(max_lag, len(z)-1)+1)
                values = [1.] + [np.dot(z[:-lag], z[lag:])/np.dot(z, z) for lag in lags[1:]]
                ax.vlines(lags, 0, values, color=color)
                ax.axhline(0, color='.5', lw=.7)
                ax.set(xlabel='Lag (blocks)', ylabel='Score residual correlation')
    elif type in {'trace', 'posterior', 'prior_posterior'}:
        if parameter is None:
            raise ValueError('Specify parameter, e.g. variance.level, sigma, xi or tau.level.')
        if parameter.startswith('tau.'):
            if fit is None:
                raise ValueError('Shared-scale plots require the full FitResult.')
            values = fit.shared_scales[parameter[4:]]
        else:
            values = c.parameters[parameter]
        if type == 'trace':
            for i, chain in enumerate(values):
                ax.plot(chain, lw=.6, alpha=.75, label=f'Chain {i+1}')
            ax.set(xlabel='Retained draw', ylabel=parameter)
        else:
            _density(ax, values, color=color, label='Posterior')
            if type == 'prior_posterior':
                if prior is None:
                    from .simulation import prior_samples
                    if fit is None:
                        from .models import MultiSeriesModel, Channel
                        model = MultiSeriesModel((Channel(c.name, c.model),))
                    else:
                        model = fit.model
                    prior = prior_samples(model, draws=20000)[parameter if parameter.startswith('tau.') else name+'.'+parameter]
                _density(ax, prior, color='.45', label='Prior', linestyle='--')
            ax.set(xlabel=parameter, ylabel='Density')
    else:
        raise ValueError(f'Unknown plot type {type!r}.')
    if path is not None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(path, dpi=dpi, bbox_inches='tight')
    return ax
