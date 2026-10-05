"""Named distribution-parameter declarations.

Constant means estimated but time invariant; Fixed means known.
Latent attaches a component predictor to a distribution parameter.
The 1.0 backend supports latent location; latent scale/shape are rejected
explicitly at compilation, rather than silently treated as constants.
"""
from dataclasses import dataclass
from ..priors import Fixed


@dataclass(frozen=True)
class Constant:
    prior: object = None


@dataclass(frozen=True)
class Latent:
    components: tuple
    link: str = 'identity'

    def __post_init__(self):
        object.__setattr__(self, 'components', tuple(self.components))
        if self.link not in {'identity', 'log'}:
            raise ValueError('Supported links are identity and log.')

__all__ = ['Fixed', 'Constant', 'Latent']
