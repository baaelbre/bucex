"""A sensitivity grid must use converged fits from the same data and model."""

from hashlib import sha256
import json

import pytest
import bucex as bx
from research.seasonal.grid import check_reference_mixing


def test_grid_requires_matching_converged_reference_at_both_origins(tmp_path):
    source = tmp_path / 'daily.csv'
    source.write_bytes(b'input for provenance\n')
    config = {'data': {'daily_source': str(source), 'frequency': 'seasonal'},
              'model': {'period': 4}, 'priors': {'xi_sd': .3},
              'copula': {'structure': 'seasons'},
              'validation': {'training_ends': ['2015-11', '2020-11']},
              'mixing_gate': str(tmp_path / 'checks')}
    with pytest.raises(RuntimeError, match='Before the prior grid'):
        check_reference_mixing(config)
    for origin in config['validation']['training_ends']:
        target = tmp_path / 'checks' / f'{origin[:4]}_copula'
        target.mkdir(parents=True)
        record = {'version': bx.__version__,
                  'source_sha256': sha256(source.read_bytes()).hexdigest(),
                  'origin': origin, 'mode': 'copula',
                  'config': {key: config[key] for key in ('data', 'model', 'priors', 'copula')}}
        (target / 'provenance.json').write_text(json.dumps(record))
        (target / 'convergence.json').write_text(json.dumps({'status': 'passed'}))
    check_reference_mixing(config)
    stale = tmp_path / 'checks' / '2020_copula' / 'provenance.json'
    content = json.loads(stale.read_text())
    content['version'] = 'older sampler'
    stale.write_text(json.dumps(content))
    with pytest.raises(RuntimeError, match='different model/data/version'):
        check_reference_mixing(config)
