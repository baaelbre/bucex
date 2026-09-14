"""Physical support diagnostics must not change the modeled joint distribution."""
import numpy as np
import pytest
import bucex as bx


def test_ordering_reports_union_and_original_samples_unchanged():
    samples = np.array([[[0., 1., 2.], [2., 1., 0.]],
                        [[1., 0., 2.], [0., 1., 2.]],
                        [[2., 1., 0.], [0., 1., 2.]],
                        [[0., 1., 2.], [1., 0., 2.]]])
    original = samples.copy()
    actual = np.array([[0., 1., 2.], [2., 1., 0.]])
    result = bx.ordering_diagnostics(samples, channel_names=('a', 'b', 'c'),
                                    constraints=(('a', 'b'), ('b', 'c')), observed=actual)
    np.testing.assert_array_equal(samples, original)
    table = result.summary.set_index('constraint')
    assert table.loc['a <= b', 'predictive_probability'] == .5
    assert table.loc['b <= c', 'predictive_probability'] == .25
    assert table.loc['any', 'predictive_probability'] == .5  # overlaps must not be summed
    assert table.loc['any', 'observed_violations'] == 1
    assert result.predictive_violations.shape == (4, 2, 2)
    np.testing.assert_array_equal(result.observed_violations, [[False, False], [True, True]])


def test_observed_only_ordering_and_tolerance_are_explicit():
    values = np.array([[1.001, 1.], [0., 1.]])
    result = bx.ordering_diagnostics(channel_names=('a', 'b'), constraints=(('a', 'b'),),
                                    observed=values, tolerance=.01)
    assert result.summary.observed_violations.sum() == 0
    assert 'predictive_probability' not in result.summary
    with pytest.raises(ValueError, match='Invalid ordering'):
        bx.ordering_diagnostics(values[None], channel_names=('a', 'b'), constraints=(('b', 'b'),))
    with pytest.raises(ValueError, match='finite'):
        bx.ordering_diagnostics(np.array([[[np.nan, 1.]]]), channel_names=('a', 'b'), constraints=(('a', 'b'),))


def test_compound_probabilities_retain_joint_dependence():
    values = np.array([[[-1., -1.]], [[-1., -1.]], [[1., 1.]], [[1., 1.]]])
    events = {'a': ('>', 0), 'b': ('>', 0)}
    np.testing.assert_array_equal(bx.compound_event_probability(values, channel_names=('a', 'b'), events=events), [.5])
    opposing = values.copy()
    opposing[:, :, 1] *= -1
    np.testing.assert_array_equal(bx.compound_event_probability(opposing, channel_names=('a', 'b'), events=events), [0.])
    np.testing.assert_array_equal(bx.compound_event_probability(opposing, channel_names=('a', 'b'), events=events, operation='any'), [1.])


def test_bundled_uccle_order_inequalities_are_valid():
    frame = bx.load_uccle_multiseries()
    result = bx.ordering_diagnostics(channel_names=frame.columns, constraints=bx.UCCLE_ORDER_CONSTRAINTS,
                                    observed=frame.to_numpy(), dates=frame.index)
    assert len(bx.UCCLE_ORDER_CONSTRAINTS) == 7
    assert result.summary.observed_violations.sum() == 0


def test_uccle_loader_supports_single_series_fallback():
    frame = bx.load_uccle_multiseries(series=["TXx"], start="2000-01-01", end="2000-12-01")
    assert frame.shape == (12, 1)
    assert tuple(frame.columns) == ("TXx",)
    with pytest.raises(ValueError, match="at least one"):
        bx.load_uccle_multiseries(series=[])
