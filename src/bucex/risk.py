"""Conditional aggregation of event probabilities across forecast blocks."""
import numpy as np


def window_risk(prediction, threshold, *, channel=None, tail='upper', indices=None):
    """Probability of at least one event, conditional on each simulated path.

    Averaging these values integrates parameter and future-state uncertainty.
    Products are taken *within* draws, never across posterior-mean block CDFs.
    The observation model assumes conditional independence across blocks.
    """
    p = prediction.risk(threshold, channel=channel, tail=tail)
    if indices is not None:
        p = p[:, indices]
    if p.ndim != 2 or p.shape[1] == 0:
        raise ValueError('Select at least one block.')
    with np.errstate(divide='ignore'):
        return -np.expm1(np.sum(np.log1p(-p), axis=1))


def block_extremes(prediction, *, channel=None, blocks, tail='upper'):
    """Aggregate consecutive predictive blocks into complete non-overlapping groups.

    The caller chooses calendar-aligned groups. Four meteorological seasons
    beginning in December do not form a January--December calendar year.
    Maxima of block maxima are exactly maxima over the union of those blocks;
    approximation enters through the fitted block distributions, not this identity.
    """
    name = prediction.channel_name(channel)
    y = prediction.y[name]
    if type(blocks) is not int or blocks < 1 or y.shape[1] % blocks:
        raise ValueError('blocks must be a positive divisor of the forecast length; incomplete groups are not discarded.')
    if tail not in {'upper', 'lower'}:
        raise ValueError('tail must be upper or lower.')
    groups = y.reshape(y.shape[0], -1, blocks)
    return np.max(groups, axis=2) if tail == 'upper' else np.min(groups, axis=2)
