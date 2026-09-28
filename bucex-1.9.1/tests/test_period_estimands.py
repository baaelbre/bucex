import numpy as np
import pandas as pd
import pytest
import bucex as bx


def test_monthly_period_average_does_not_silently_use_partial_periods():
    dates = pd.date_range('2000-01-01',periods=36,freq='MS')
    values = np.broadcast_to(np.arange(36.),(4,8,36))
    np.testing.assert_array_equal(bx.period_average(values,dates,('2000-01','2001-12'),month=7),12.)
    with pytest.raises(ValueError,match='complete'):
        bx.period_average(values,dates,('1999-01','2001-12'))
    with pytest.raises(ValueError,match='unique'):
        bx.period_average(values,np.repeat(dates[0],36),('2000-01','2000-12'))


def test_period_contrasts_preserve_joint_cancellation():
    class Joint:
        is_multiseries_model = True
        channel_names = ('a','b')
        time = pd.date_range('2000-01-01',periods=24,freq='MS')
        def component_draws(self,component,channel=None,combine_chains=False):
            base = np.arange(24.)[None,None,:]*np.arange(1,9.).reshape(2,4,1)
            return base + (np.arange(24.)[None,None,:] if channel == 'a' else 0)
        def parameter_path(self,parameter,channel=None,combine_chains=False):
            return self.component_draws('level',channel)
    result = bx.period_contrasts(Joint(),('2000-01','2000-12'),('2001-01','2001-12'),
                                pairs=[('a','b')],months=[1,7])
    assert result['a.level.change'].shape == (2,4)
    np.testing.assert_array_equal(result['a_minus_b.level.change'],12.)
    assert np.std(result['a.level.change']) > 0


def test_constant_or_short_chains_are_not_reported_as_converged():
    table = bx.summarize_draws({'constant':np.ones((1,4))})
    result = bx.convergence_assessment({'test':table})
    assert result['status'] == 'needs_review'
    assert result['issues']
