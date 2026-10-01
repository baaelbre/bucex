"""Translate the public components to the non-centred structural system."""
import numpy as np
from ._state import NCPLayout


def layout_for(model):
    slope = model.trend.trend_mode != 'off'
    k = model.season.period - 1 if model.season else 0
    names = ['level'] + (['slope'] if slope else []) + [f'seasonal[{i+1}]' for i in range(k)]
    znames = ['tilde_alpha'] + (['tilde_beta', 'A'] if slope else []) + [f'tilde_g{i+1}' for i in range(k)]
    a, z = 1 + int(slope), 1 + 2*int(slope)
    active = tuple({'slope': 'trend', 'seasonal': 'season'}.get(c, c) for c in model.active)
    return NCPLayout(tuple(names), len(names), True, slope, k, 0,
        1 if slope else None, slice(a, a+k), 0, 1 if slope else None,
        2 if slope else None, slice(z, z+k), tuple(znames), len(znames), active)


def physical_system(model):
    layout = layout_for(model)
    d = layout.centered_state_dim
    transition = np.eye(d)
    if layout.has_beta:
        transition[0, 1] = 1
    if layout.season_dim:
        from ._state import seasonal_rotation_matrix
        transition[layout.season_slice, layout.season_slice] = seasonal_rotation_matrix(layout.season_dim)
    return transition, layout
