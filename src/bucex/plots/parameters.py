from .common import _density
from ..results import ChannelResult


def parameter_plot(c, ax, *, fit, name, type, color, parameter=None, prior=None, **kwargs):
    if not isinstance(c, ChannelResult):
        raise TypeError('Parameter plots require a fitted model.')
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
        return
    _density(ax, values, color=color, label='Posterior')
    if type == 'prior_posterior':
        if prior is None:
            from ..simulation import prior_samples
            from ..models import MultiSeriesModel, Channel
            model = fit.model if fit else MultiSeriesModel((Channel(c.name, c.model),))
            prior = prior_samples(model, draws=20000)[parameter if parameter.startswith('tau.') else name+'.'+parameter]
        _density(ax, prior, color='.45', label='Prior', linestyle='--')
    ax.set(xlabel=parameter, ylabel='Density')
