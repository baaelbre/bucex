"""The released research protocol must match the stated primary analysis."""
from pathlib import Path
import numpy as np
import bucex as bx
from research.serra.models import channel
from research.serra.preflight import inspect
from research.serra.experiment import configured_variant

CONFIG = Path(__file__).resolve().parents[1]/'research/serra/config'


def test_primary_record_and_periods_reach_august_2026():
    config = bx.load_config(CONFIG/'copula_full.json')
    report = inspect(config)
    assert report['end'] == '2026-08-01'
    assert report['n_months'] == 1616
    assert report['contrasts']['comparison'] == ['1996-09','2026-08']
    assert report['asis'] is False
    assert report['copula'] == {'structure':'constant','eta':1.}
    data = bx.load_uccle_multiseries(**config['data'])
    for name in data:
        scale = channel(name,data,config).observation.scale
        assert scale.mode == 'constant' and scale.period == 1


def test_historical_comparison_keeps_complete_climate_windows():
    report = inspect(bx.load_config(CONFIG/'copula_1892_2022.json'))
    assert report['end'] == '2022-12-01' and report['n_months'] == 1572
    assert report['contrasts']['comparison'] == ['1993-01','2022-12']


def test_five_core_variants_and_joint_sensitivity_use_same_models():
    base = bx.load_config(CONFIG/'sensitivity/paper.json')
    joint = bx.load_config(CONFIG/'sensitivity/joint.json')
    assert joint['analysis'] == 'copula'
    assert len(base['variants']) == 5 and base['variants'] == joint['variants']
    for label,factor in [('innovation_half',.5),('innovation_double',2.)]:
        variant = next(v for v in base['variants'] if v['name']==label)
        result = configured_variant(base,variant)
        for key,value in base['priors']['innovation_median'].items():
            assert result['priors']['innovation_median'][key] == factor*value
    result = configured_variant(joint,{'name':'lkj_4','copula':{'eta':4.}})
    assert result['copula']['eta'] == 4. and joint['copula']['eta'] == 1.


def test_reviewer_shape_grid_and_constant_scale_recovery():
    config = bx.load_config(CONFIG/'simulation/paper.json')
    np.testing.assert_allclose(config['simulation']['xi_grid'],np.linspace(-.5,.5,11))
    assert config['priors']['xi_bounds'] == [-.75,.75]
    joint = bx.load_config(CONFIG/'joint_recovery_full.json')
    assert joint['model']['seasonal_scale'] is False
    assert joint['simulation']['log_scale_amplitude'] == 0.
