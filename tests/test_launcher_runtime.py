"""HPC startup must preserve a venv while removing login-only directory aliases."""
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
RUNTIME = PROJECT / 'bash_scripts/runtime.sh'


def test_runtime_removes_directory_alias_and_preserves_venv(tmp_path):
    physical = tmp_path / 'physical project'
    physical.mkdir()
    alias = tmp_path / 'login alias'
    alias.symlink_to(physical, target_is_directory=True)
    env_dir = physical / 'env'
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(env_dir)], check=True)
    env = dict(os.environ, BUCEX_PYTHON=str(alias / 'env/bin/python'),
               BUCEX_RESULTS_ROOT=str(alias / 'results'))
    result = subprocess.run([
        'bash', '-c',
        'set -euo pipefail\nsource "$1"\n'
        '"$BUCEX_PYTHON" -c \'import json,os,sys; print(json.dumps(dict('
        'executable=sys.executable,prefix=sys.prefix,base=sys.base_prefix,'
        'selected=os.environ["BUCEX_PYTHON"],root=os.environ["BUCEX_RESULTS_ROOT"])))\'',
        'runtime-test', str(RUNTIME),
    ], cwd=physical, env=env, text=True, capture_output=True, check=True)
    actual = json.loads(result.stdout)
    assert actual['selected'] == actual['executable'] == str(env_dir / 'bin/python')
    assert actual['prefix'] == str(env_dir)
    assert actual['prefix'] != actual['base']
    assert actual['root'] == str(physical / 'results')


def test_missing_python_reports_requested_path_and_host(tmp_path):
    missing = str(tmp_path / 'missing environment/bin/python')
    result = subprocess.run(['bash', '-c', 'source "$1"', 'runtime-test', str(RUNTIME)],
                            cwd=tmp_path,
                            env=dict(os.environ, BUCEX_PYTHON=missing, HOSTNAME='compute-test'),
                            text=True, capture_output=True)
    assert result.returncode == 2
    assert missing in result.stderr and 'compute-test' in result.stderr
    assert not (tmp_path / 'results').exists()
