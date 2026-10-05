"""Declarative single-series models and named independent observation channels."""
from dataclasses import dataclass, field, replace
from ..components import Component, LocalLevel, LocalLinearTrend, DummySeasonal
from ..priors import Priors, Pooling, Normal, Fixed, InverseGamma
from ..parameters import Constant, Latent


@dataclass(frozen=True)
class Model:
    observation: object
    components: tuple = field(default_factory=lambda: (LocalLinearTrend(),))
    priors: Priors = field(default_factory=Priors)
    parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        components = tuple(self.components)
        parameters = dict(self.parameters)
        if not isinstance(self.priors, Priors):
            raise TypeError('priors must be Priors(...).')
        for method in ('logpdf', 'cdf', 'sf', 'ppf', 'grad_eta', 'hess_eta'):
            if not callable(getattr(self.observation, method, None)):
                raise TypeError(f'Observation family must implement {method}.')
        if not hasattr(self.observation, 'name') or not hasattr(self.observation, 'scale'):
            raise TypeError('Observation family must declare name and scale.')
        if set(parameters)-{'location', 'scale', 'shape'}:
            raise ValueError('Parameter names are location, scale and shape.')
        for value in parameters.values():
            if not isinstance(value, (Constant, Fixed, Latent)):
                raise TypeError('Use Constant, Fixed or Latent parameter declarations.')
        location = parameters.get('location')
        if isinstance(location, Latent):
            components = location.components
        elif isinstance(location, (Constant, Fixed)):
            prior = location if isinstance(location, Fixed) else location.prior or self.priors.initial_level
            components = (LocalLevel('static', initial_prior=prior),)
        priors = self.priors
        scale = parameters.get('scale')
        if isinstance(scale, (Constant, Fixed)) and self.observation.scale is not None:
            raise ValueError('Use either SeasonalScale or a named constant scale declaration.')
        if 'shape' in parameters and getattr(self.observation, 'gaussian_location', False):
            raise ValueError('Gaussian observations have no shape parameter.')
        if isinstance(scale, Fixed):
            if scale.value <= 0:
                raise ValueError('Fixed scale must be positive.')
            priors = replace(priors, variance=Fixed(scale.value**2))
        elif isinstance(scale, Constant) and scale.prior is not None:
            if not isinstance(scale.prior, InverseGamma):
                raise TypeError('A Constant scale uses an InverseGamma prior on its square.')
            priors = replace(priors, variance=scale.prior)
        shape = parameters.get('shape')
        if isinstance(shape, Fixed):
            priors = replace(priors, shape=shape)
        elif isinstance(shape, Constant) and shape.prior is not None:
            priors = replace(priors, shape=shape.prior)
        if not components or any(not isinstance(c, Component) for c in components):
            raise TypeError('Components must implement name and build(steps, priors, exog).')
        if len({c.name for c in components}) != len(components):
            raise ValueError('Component names must be unique.')
        object.__setattr__(self, 'components', components)
        object.__setattr__(self, 'priors', priors)
        object.__setattr__(self, 'parameters', parameters)

    @property
    def season(self):
        return next((c for c in self.components if isinstance(c, DummySeasonal) and c.mode != 'off'), None)

    @property
    def period(self):
        return self.observation.scale.period if self.observation.scale else self.season.period if self.season else 1

    @property
    def active(self):
        # For inspection; fitting obtains innovations from the compiled blocks.
        from .compiler import compile_model, placeholder_exog
        return tuple(g.name for g in compile_model(self, 1, placeholder_exog(self, 1)).groups)


@dataclass(frozen=True)
class Channel:
    name: str
    model: Model

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or '/' in self.name or '\\' in self.name:
            raise ValueError('Channel name must be nonempty and contain no path separators.')
        if not isinstance(self.model, Model):
            raise TypeError('Channel.model must be a Model.')


@dataclass(frozen=True)
class MultiSeriesModel:
    channels: tuple
    pooling: Pooling | None = None

    def __post_init__(self):
        object.__setattr__(self, 'channels', tuple(self.channels))
        if not self.channels or any(not isinstance(c, Channel) for c in self.channels):
            raise ValueError('Provide one or more Channel objects.')
        if len({c.name for c in self.channels}) != len(self.channels):
            raise ValueError('Channel names must be unique.')
        if self.pooling is not None and not isinstance(self.pooling, Pooling):
            raise TypeError('pooling must be Pooling or None.')
        if self.pooling:
            from .compiler import compile_model, placeholder_exog
            blocks = [compile_model(c.model, 1, placeholder_exog(c.model, 1)) for c in self.channels]
            for name in self.pooling.scales:
                members = [g for b in blocks for g in b.groups if g.name == name]
                if not members or any(not isinstance(g.prior, Normal) or g.prior.mean != 0 for g in members):
                    raise ValueError(f'Pooled {name} needs active zero-centred Normal amplitudes.')
