"""Posterior objects consumed by prediction, diagnostics, plotting and storage."""
from dataclasses import dataclass
import numpy as np
from .inference._model import layout_for


def summarize(draws, interval=.95, axis=(0, 1)):
    if not 0 < interval < 1:
        raise ValueError('interval must lie strictly between zero and one.')
    q = (1-interval)/2
    return dict(mean=np.mean(draws, axis=axis), lower=np.quantile(draws, q, axis=axis),
                upper=np.quantile(draws, 1-q, axis=axis))


@dataclass
class ChannelResult:
    name: str
    model: object
    y: np.ndarray
    index: object
    states: np.ndarray  # (chain, draw, time including time zero, state)
    parameters: dict
    metrics: dict
    steps_per_year: float

    def path(self, component='location'):
        """Draws on the original response scale, excluding the initial state."""
        layout = layout_for(self.model)
        level = self.states[:, :, 1:, 0]
        if component == 'level':
            return level
        if component == 'slope':
            return self.states[:, :, 1:, 1] if layout.has_beta else np.zeros_like(level)
        seasonal = self.states[:, :, 1:, layout.season_slice.start] if layout.season_dim else np.zeros_like(level)
        if component == 'seasonal':
            return seasonal
        if component == 'location':
            return level+seasonal
        raise ValueError('component must be location, level, slope or seasonal.')

    def sigma(self):
        if self.model.observation.scale:
            values = np.stack([self.parameters[f'sigma[{i}]'] for i in range(self.model.period)], axis=-1)
            return values[:, :, np.arange(len(self.y)) % self.model.period]
        return np.broadcast_to(self.parameters['sigma'][:, :, None], self.path().shape)

    def risk(self, threshold, *, tail='upper'):
        if tail not in {'upper', 'lower'}:
            raise ValueError('tail must be upper or lower.')
        params = {'sigma': self.sigma(), 'xi': self.parameters.get('xi', np.zeros(self.states.shape[:2]))[:, :, None]}
        fn = self.model.observation.sf if tail == 'upper' else self.model.observation.cdf
        return fn(threshold, self.path(), params)

    def return_level(self, years, *, tail='upper'):
        """Season-specific stationary-equivalent quantile, one opportunity/year.

        Select a phase from the returned path; this is not an annual maximum
        quantile over all seasons. years must exceed one.
        """
        if not np.isfinite(years) or years <= 1 or tail not in {'upper', 'lower'}:
            raise ValueError('years must exceed one and tail must be upper or lower.')
        p = 1-1/years if tail == 'upper' else 1/years
        params = {'sigma': self.sigma(), 'xi': self.parameters.get('xi', np.zeros(self.states.shape[:2]))[:, :, None]}
        return self.model.observation.ppf(p, self.path(), params)

    def plot(self, type='level', **kwargs):
        from .plotting import plot
        return plot(self, type=type, **kwargs)


@dataclass
class FitResult:
    model: object
    channels: dict[str, ChannelResult]
    shared_scales: dict
    metadata: dict

    def __getitem__(self, name):
        return self.channels[name]

    def channel(self, name=None):
        if name is None:
            if len(self.channels) != 1:
                raise ValueError('Select a channel by name for a multiseries result.')
            name = next(iter(self.channels))
        return self.channels[name]

    @property
    def chains(self):
        return next(iter(self.channels.values())).states.shape[0]

    @property
    def draws(self):
        return next(iter(self.channels.values())).states.shape[1]

    def summary(self, interval=.95, *, include_paths=False):
        from .diagnostics import summary
        return summary(self, interval=interval, include_paths=include_paths)

    def predict(self, horizon, **kwargs):
        from .prediction import predict
        return predict(self, horizon, **kwargs)

    def replicate(self, **kwargs):
        from .prediction import replicate
        return replicate(self, **kwargs)

    def plot(self, type='level', **kwargs):
        from .plotting import plot
        return plot(self, type=type, **kwargs)

    def save(self, path):
        from .serialization import save
        return save(self, path)

    @classmethod
    def load(cls, path):
        from .serialization import load
        return load(path)
