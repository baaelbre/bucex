"""The paper configurations resolve to the intended windows and contrasts."""
from pathlib import Path

import bucex as bx
from research.monthly.preflight import inspect
from research.monthly.experiment import configured_variant

ROOT = Path(__file__).resolve().parents[1]


def test_reference_windows_and_private_model_contract():
    monthly = inspect(bx.load_config(ROOT / 'research/monthly/config/main.json'))
    seasonal = inspect(bx.load_config(ROOT / 'research/seasonal/config/main.json'))
    assert (monthly['start'], monthly['end'], monthly['n_blocks']) == ('1892-03-01', '2026-08-01', 1614)
    assert seasonal['n_blocks'] == 538 and seasonal['data_audit']['last_included_day'] == '2026-08-31'
    assert monthly['version'] == seasonal['version'] == bx.__version__
    assert monthly['copula']['structure'] == seasonal['copula']['structure'] == 'seasons'
    assert all(item['model']['observation']['phi'] == 'stationary'
               for item in monthly['channels'] if item['family'] == 'gev')


def test_both_adequacy_studies_include_constant_dispersion():
    for frequency in ('monthly', 'seasonal'):
        config = bx.load_config(ROOT / f'research/{frequency}/config/adequacy.json')
        variants = {item['name']: item for item in config['variants']}
        assert config['model']['seasonal_scale'] is True
        assert variants['constant_dispersion']['seasonal_scale'] is False
        reduced = configured_variant(config, variants['constant_dispersion'])
        assert reduced['model']['seasonal_scale'] is False
        assert reduced['priors'] == config['priors']
