"""Separate model expressiveness from the capabilities of the current sampler."""
from dataclasses import dataclass
from ..parameters import Latent


@dataclass(frozen=True)
class InferencePlan:
    path_update: str
    coefficient_update: str
    parameter_blocks: tuple


def plan(model):
    for name, parameter in model.parameters.items():
        if isinstance(parameter, Latent) and (name != 'location' or parameter.link != 'identity'):
            raise NotImplementedError(f'BUCEX 1.0 inference supports identity-linked latent location; '
                f'{name} with link={parameter.link} requires another parameter-update backend.')
    if model.observation.name not in {'gaussian', 'gev'}:
        raise NotImplementedError('The 1.0 observation-parameter backend supports Gaussian and GEV families. '
                                  'A new family also needs matching parameter updates.')
    gaussian = getattr(model.observation, 'gaussian_location', False)
    return InferencePlan('ffbs' if gaussian else 'laplace_mh',
                         'normal' if gaussian else 'elliptical_slice',
                         ('variance','scale_contrasts') + (() if gaussian else ('shape',)))
