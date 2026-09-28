"""Fixed-SD calibration and six-analysis HPC experiment contracts."""
from copy import deepcopy
from pathlib import Path
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
import bucex as bx
from scipy.stats import norm
from research.monthly.models import independent_model,fit_options
from research.monthly.experiment import configured_variant
from research.monthly.report import write_report
from research.seasonal import jobs,bundles
from research.seasonal.job_plan import plan

PROJECT=Path(__file__).resolve().parents[1]


def config(name,variant='reference'):
    task=next(t for t in jobs.tasks('posterior') if t.channel==name and t.variant=="fixed_"+variant)
    c=jobs.task_config(task,'screen');c['data']['start']='2016-03'
    c['mcmc'].update(chains=2,chain_workers=2,warmup=3,draws=10,progress=False)
    c.update(figures=False,forecast_draws=24,predictive_check_draws=12,prior_draws=80,contrasts=None)
    return c


def test_fixed_normal_calibration_has_no_scale_mixture():
    c=config('TXm');data=bx.load_uccle_multiseries(**c['data']);_,prior=independent_model(data,c)
    assert prior.shrinkage is None
    p=prior.channels['TXm']
    for field,m in [('s_level',.01),('s_trend',.0001),('s_season',.01),('beta0',.01)]:
        normal=getattr(p,field)
        assert normal.mean==0 and normal.sd==pytest.approx(m)
        assert norm.cdf(m,scale=normal.sd)-norm.cdf(-m,scale=normal.sd)==pytest.approx(norm.cdf(1)-norm.cdf(-1))
    draws=bx.draw_marginal_prior(prior,60000,seed=193)
    assert draws['shared']=={} and draws['independent']=={}
    assert np.mean(draws['channels']['TXm']['sd.level']**2)==pytest.approx((.01)**2,rel=.02)
    with pytest.raises(ValueError,match='no innovation_median'):
        bx.fs_priors('gaussian',innovation_sd=c['priors']['innovation_sd'],innovation_median={'level':1,'trend':1,'season':1})
    c['priors']['independent_shrinkage']={'log_sd':1}
    with pytest.raises(ValueError,match='hyperprior'):independent_model(data,c)


def test_every_bundle_matches_tasks_and_cpu_budget():
    for tier in ('screen','paper'):
        groups=plan(tier,'all');assert len(groups)==101
        assert len(plan(tier,'posterior'))==49
        all_members=[]
        for group in groups:
            members=bundles.members(group,tier);all_members.extend(members)
            assert group['required_workers']==len(members)*2
            assert group['cpus']==(2 if group['scope']=='shared' else 12)
            assert len(members)==(1 if group['scope']=='shared' else 6)
            for t in members:
                c=jobs.task_config(t,tier)
                assert c['copula'] is None
                assert not (c['priors'].get('shared_shrinkage') or {}).get('pool_initial_slope',False)
        assert len(all_members)==271 and len({t.id for t in all_members})==271
    assert len(plan('paper','experiments','shared_long'))==1
    assert len(plan('paper','experiments','separate_long'))==2


def test_sd_sensitivity_changes_normal_sd_not_variance():
    ref=config('TXm')
    trial=configured_variant(ref,{'sd_multipliers':{'trend':2.,'season':.25,'initial_slope':.5}})
    p=trial['priors'];r=ref['priors']
    assert p['innovation_sd']['trend']==r['innovation_sd']['trend']*2
    assert p['innovation_sd']['season']==r['innovation_sd']['season']*.25
    assert p['initial_slope_sd']==r['initial_slope_sd']*.5
    assert p['innovation_sd']['level']==r['innovation_sd']['level']


@pytest.mark.parametrize('name', ['TXm','TNm','TXx','TXn','TNx','TNn'])
def test_fixed_prior_fits_two_parallel_chains_and_reports(tmp_path,name):
    c=config(name);data=bx.load_uccle_multiseries(**c['data']);model,prior=independent_model(data,c)
    fit=bx.fit(data,model,priors=prior,**fit_options(c,family=model.family))
    assert fit.n_chains==2 and fit.priors.shrinkage is None and fit.model.copula is None
    assert not any(k.startswith('shrinkage.') for k in fit.parameter_draws)
    write_report(fit,tmp_path,config=c,horizon=120,risks=c['risks'])
    restored=bx.load_fit(tmp_path/'fit.bucex')
    assert restored.priors.shrinkage is None
    table=pd.read_csv(tmp_path/'fixed_prior_settings.csv')
    assert not table.sampled_hyperparameter.any()
    assert table.set_index('component').loc['initial_slope','prior_sd']==pytest.approx(.01)
    assert len(pd.read_csv(tmp_path/'initial_slope_prior_posterior.csv'))==2
    assert len(pd.read_csv(tmp_path/(name+'_forecast_uncertainty.csv')))==360
    assert not list(tmp_path.glob('*shrinkage*.csv'))
    report=bx.SensitivityReport(posterior_runs={'reference':{name:tmp_path}},baseline='reference').tables()
    assert 'fixed_prior_settings' in report and 'initial_slope_prior_posterior' in report
    assert 'shared_shrinkage' not in report and 'independent_shrinkage' not in report


def test_login_submitter_never_executes_compute_python(tmp_path):
    marker=tmp_path/'executed'
    wrong=tmp_path/'wrong_cpu_python';wrong.write_text('#!/bin/sh\ntouch "'+str(marker)+'"\nexit 132\n');wrong.chmod(0o755)
    for tier,cores in [('screen',12),('paper',12)]:
        result=subprocess.run([sys.executable,'-S',str(PROJECT/'job_scripts/submit.py'),tier,'experiments','--dry-run'],
            cwd=PROJECT,env=dict(os.environ,BUCEX_PYTHON=str(wrong),VSC_CLUSTER='gallade',VSC_ARRAY_LIMIT='8'),capture_output=True,text=True,check=True)
        assert '--clusters=gallade' in result.stdout
        assert '--cpus-per-task='+str(cores) in result.stdout
        assert '--dependency=afterok:PROBE_JOB_ID' in result.stdout
        assert '%8' in result.stdout and not marker.exists()


def test_actual_submission_protocol_with_stub_scheduler(tmp_path):
    commands=tmp_path/'commands.jsonl';counter=tmp_path/'counter'
    script=tmp_path/'sbatch'
    script.write_text('#!'+sys.executable+'\n'+'''import os,sys,json
from pathlib import Path
counter=Path(os.environ['TEST_COUNTER']);n=int(counter.read_text())+1 if counter.exists() else 100
counter.write_text(str(n))
with open(os.environ['TEST_COMMANDS'],'a') as f:f.write(json.dumps({'args':sys.argv[1:],'resource':os.environ.get('BUCEX_RESOURCE'),'python':os.environ['BUCEX_PYTHON']})+'\\n')
print(str(n)+';gallade')
''');script.chmod(0o755)
    result=subprocess.run([sys.executable,'-S',str(PROJECT/'job_scripts/submit.py'),'paper','experiments'],cwd=PROJECT,
        env=dict(os.environ,PATH=str(tmp_path)+os.pathsep+os.environ['PATH'],BUCEX_PYTHON=sys.executable,
            BUCEX_RESULTS_ROOT=str(tmp_path/'results'),TEST_COUNTER=str(counter),TEST_COMMANDS=str(commands)),capture_output=True,text=True,check=True)
    records=[json.loads(line) for line in commands.read_text().splitlines()]
    assert [r['resource'] for r in records]==['probe','shared','separate','shared_long','separate_long']
    assert all('--dependency=afterok:100' in r['args'] for r in records[1:])
    assert '--cpus-per-task=12' in records[2]['args']


def test_bundle_starts_six_separate_processes_and_propagates_failure(tmp_path,monkeypatch):
    group=next(g for g in plan('screen','comparison') if g['scope']=='fixed')
    monkeypatch.setattr(bundles,'allocation',lambda required:None)
    real_popen=subprocess.Popen
    worker='''import json,os,sys,time
from pathlib import Path
start=time.time();time.sleep(.5)
Path(sys.argv[1]).write_text(json.dumps({'start':start,'end':time.time(),'threads':os.environ['OPENBLAS_NUM_THREADS']}))
sys.exit(int(sys.argv[2]))
'''
    def launch(args,**kwargs):
        id=args[args.index('--task')+1]
        return real_popen([sys.executable,'-c',worker,str(tmp_path/(id+'.json')),'2' if id.endswith('TXn') else '0'],**kwargs)
    monkeypatch.setattr(bundles.subprocess,'Popen',launch)
    assert bundles.run(group,tier='screen',root=tmp_path/'results')==1
    intervals=[json.loads(p.read_text()) for p in tmp_path.glob('posterior_fixed_reference_*.json')]
    assert len(intervals)==6
    points=sorted([(r['start'],1) for r in intervals]+[(r['end'],-1) for r in intervals])
    assert np.cumsum([x[1] for x in points]).max()==6
    assert all(r['threads']=='1' for r in intervals)
    meta=bx.load_config(tmp_path/'results/screen/bundles/posterior_fixed_reference/bundle.json')
    assert meta['status']=='failed' and len(meta['outcomes'])==6
    assert sum(r['exit_code']!=0 for r in meta['outcomes'])==1


def test_too_small_allocation_is_rejected(monkeypatch):
    monkeypatch.setenv('SLURM_CPUS_PER_TASK','2')
    with pytest.raises(RuntimeError,match='needs 12'):bundles.allocation(12)


def test_paper_30_year_prior_effects_and_initial_slope_variance():
    from research.monthly.fixed_priors import calibration
    c=config('TXm');data=bx.load_uccle_multiseries(**c['data']);_,prior=independent_model(data,c)
    table=calibration(prior.channels['TXm'],period=4,horizon=120,rate_multiplier=40).set_index('component')
    assert prior.channels['TXm'].beta0.sd**2==pytest.approx(.0001)
    assert table.loc['level','displacement_sd']==pytest.approx(.01*np.sqrt(120))
    assert table.loc['slope','displacement_sd']==pytest.approx(.0001*np.sqrt(120*119*239/6))
    assert table.loc['seasonal','displacement_sd']==pytest.approx(.01*np.sqrt(60))
    assert table.loc['initial_slope','displacement_sd']==pytest.approx(1.2)
    assert table.loc['initial_slope','initial_rate_sd_C_per_decade']==pytest.approx(.4)
    for variant,target in [('half_initial_slope',.2),('reference',.4),('double_initial_slope',.8)]:
        c=config('TXm',variant)
        assert 40*c['priors']['initial_slope_sd']==pytest.approx(target)
