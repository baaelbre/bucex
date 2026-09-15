import numpy as np
import pytest
import bucex as bx


def test_paired_blocks_preserve_within_block_repetition():
    baseline = np.array([2., 2., 4., 4.])
    candidate = np.array([1., 1., 2., 2.])
    result = bx.paired_block_comparison(baseline, candidate, [1,1,2,2], seed=3)
    repeated = bx.paired_block_comparison(np.repeat(baseline, 3), np.repeat(candidate, 3),
                                         np.repeat([1,1,2,2], 3), seed=3)
    assert result['improvement'] == 1.5
    assert (result['lower'], result['upper']) == (repeated['lower'], repeated['upper'])
    assert result['n_blocks'] == 2
    single = bx.paired_block_comparison(baseline, candidate, [1,1,1,1])
    assert np.isnan(single['lower'])
    with pytest.raises(ValueError, match='finite'):
        bx.paired_block_comparison([np.inf,1], [1,2], [1,2])
