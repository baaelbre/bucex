"""Release regressions for 1.9.8 pooling, seasonal checks and scheduler dispatch."""
import copy
from pathlib import Path
import os
import subprocess
import sys
import pandas as pd
from research.seasonal.jobs import tasks,task_config
from research.seasonal.job_plan import plan
from research.seasonal.sweetspot_plan import cells,private_cells
from research.seasonal.sweetspot_report import aggregate_posterior,rectangles,CHANNELS
from tests.test_sweetspot_1961 import entry


def test_calibration_dimensions_and_matched_private_priors():
    pooled=cells();private=private_cells()
    core=[c for c in pooled if c['calibration_kind']=='grid']
    assert {(c['A_level'],c['A_slope'],c['A_season']) for c in core}=={
        (a,b,.1) for a in (.05,.1,.2) for b in (.001,.002,.004)}
    assert {(c['A_level'],c['A_slope'],c['A_season']) for c in pooled if c not in core}=={(.1,.002,.05),(.1,.002,.2)}
    assert len(private)==len(pooled)==11
    ts=tasks('sweetspot_posterior',tier='screen')
    for t in ts:
        c=task_config(t,'screen')
        assert c['mcmc']['draws']==2000 and c['mcmc']['warmup']==1000
        if t.scope=='independent':
            shared=next(s for s in ts if s.scope=='shared' and s.variant==c['variant']['setting'])
            p=copy.deepcopy(c['priors']);q=copy.deepcopy(task_config(shared,'screen')['priors'])
            assert p.pop('independent_shrinkage')==q.pop('shared_shrinkage')
            assert p==q
    seeds=[task_config(t,'screen')['mcmc']['seed'] for t in ts]
    assert len(seeds)==len(set(seeds))


def test_private_collection_retains_every_response_and_blocks_partial_bundle():
    parts=[]
    for channel in CHANNELS:
        e=entry();e.update(scope='independent',directory=Path(channel))
        for key in ('paths','forecasts','allocation','risk'):e[key]=e[key][e[key].channel==channel].copy()
        e['local']=pd.DataFrame()
        parts.append((channel,e))
    a=aggregate_posterior({'independent_reference':parts})['independent_reference']
    assert a['complete'] and a['passed']
    assert set(a['paths'].channel)==set(CHANNELS)
    assert len(a['directories'])==6
    b=aggregate_posterior({'independent_reference':parts[:-1]})['independent_reference']
    assert not b['complete'] and not b['passed']


def test_private_forecast_array_calls_bundle_runner(tmp_path):
    groups=plan('screen','sweetspot_hpc','separate')
    target='sweetspot_forecast_independent_reference_1990-11'
    index=1+[g['id'] for g in groups].index(target)
    wrapper=tmp_path/'python_wrapper'
    wrapper.write_text('#!'+sys.executable+'\nimport os,sys\n'
        'if sys.argv[1:3]==["-u","-m"]:\n'
        '    assert sys.argv[3]=="research.seasonal.bundles",sys.argv\n'
        '    print("PRIVATE_BUNDLE_RUNNER_OK",flush=True)\n'
        'else: os.execv(sys.executable,[sys.executable,*sys.argv[1:]])\n')
    wrapper.chmod(0o755)
    env=dict(os.environ,BUCEX_PROJECT_ROOT=str(Path.cwd()),BUCEX_PYTHON=str(wrapper),
        BUCEX_ENV_SETUP='',BUCEX_RESULTS_ROOT=str(tmp_path/'results'),BUCEX_TIER='screen',
        BUCEX_BATCH='sweetspot_hpc',BUCEX_RESOURCE='separate',SLURM_ARRAY_TASK_ID=str(index))
    r=subprocess.run(['bash','job_scripts/run_task.sh'],env=env,capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    assert 'PRIVATE_BUNDLE_RUNNER_OK' in r.stdout


def test_hpc_resources_and_private_array_cap(tmp_path):
    env=dict(os.environ,BUCEX_PYTHON=sys.executable,BUCEX_RESULTS_ROOT=str(tmp_path),VSC_ARRAY_LIMIT='16',VSC_SEPARATE_ARRAY_LIMIT='4')
    r=subprocess.run(['bash','RUN_SWEETSPOT_HPC.sh','--dry-run'],env=env,capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert '--array=1-33%16' in r.stdout and '--array=1-13%4' in r.stdout
    assert '--cpus-per-task=12 --mem=48G' in r.stdout
    assert len(plan('screen','sweetspot_hpc'))==46
    assert len(plan('screen','sweetspot_validation'))==36
