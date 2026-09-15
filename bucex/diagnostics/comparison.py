"""Paired uncertainty summaries for held-out predictive scores."""
import numpy as np


def paired_block_comparison(baseline, candidate, blocks, *, draws=2000, level=.95, seed=None):
    """Bootstrap paired score differences by whole user-specified blocks.

    Positive differences mean the candidate improves a lower-is-better score.
    All cases must be aligned and finite. Resampling preserves dependence
    within each block; the interval assumes independent blocks. Choose longer
    blocks if serial dependence extends across the chosen boundaries. With one
    block only the mean is reported: an uncertainty interval is unidentified.
    This is sampling uncertainty across forecast cases, not a posterior interval.
    """
    baseline, candidate, blocks = np.asarray(baseline, float), np.asarray(candidate, float), np.asarray(blocks)
    if baseline.ndim != 1 or candidate.shape != baseline.shape or blocks.shape != baseline.shape or not baseline.size:
        raise ValueError('Supply nonempty aligned one-dimensional scores and block labels.')
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(candidate)):
        raise ValueError('Scores must be finite; investigate zero predictive densities before comparison.')
    if int(draws) < 2 or not 0 < level < 1:
        raise ValueError('draws must be at least two and level must lie in (0,1).')
    _, inverse = np.unique(blocks, return_inverse=True)
    difference = baseline-candidate
    sizes = np.bincount(inverse)
    sums = np.bincount(inverse, weights=difference)
    lower = upper = np.nan
    if len(sizes) > 1:
        rng = np.random.default_rng(seed)
        sampled = rng.integers(0, len(sizes), size=(int(draws), len(sizes)))
        estimates = sums[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)
        lower, upper = np.quantile(estimates, [(1-level)/2, (1+level)/2])
    return {'improvement': float(difference.mean()), 'lower': float(lower), 'upper': float(upper),
            'n_cases': len(difference), 'n_blocks': len(sizes), 'level': float(level),
            'status': 'one block: interval unavailable' if len(sizes) < 2 else 'paired block bootstrap'}
