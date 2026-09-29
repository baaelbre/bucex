"""Tests of study identity, horizon matching and stability decisions."""
from itertools import combinations
import ast
import copy
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import bucex as bx
from research.seasonal.jobs import tasks,task_config,fingerprint
from research.seasonal.sweetspot_plan import cells,calibration_rows
from research.seasonal.sweetspot_diagnostics import local_derivative,variance_parts
from research.seasonal.sweetspot_report import CHANNELS,compare_pair,rectangles,validation_tables
from research.monthly.validate import validation_splits


def test_disjoint_host_work_and_fixed_other_priors():
    h=tasks('sweetspot_hpc',tier='screen');b=tasks('sweetspot_validation',tier='screen')
    assert len(h)==len(b)==75
    assert not {t.id for t in h}&{t.id for t in b}
    assert {t.id for t in h+b}=={t.id for t in tasks('sweetspot',tier='screen')}
    assert len(cells())==25
    priors=[]
    for task in tasks('sweetspot_posterior',tier='screen'):
        c=task_config(task,'screen');p=copy.deepcopy(c['priors'])
        scale=p.pop('innovation_sd')
        assert scale['season']==.01
        assert c['model']['level']==c['model']['trend']=='dynamic'
        assert c['mcmc']['chains']==c['mcmc']['chain_workers']==2
        assert c['copula'] is None and c['sweetspot_diagnostics']
        assert p['shared_shrinkage']['hyperprior']=='half_normal'
        assert not p['shared_shrinkage']['pool_initial_slope']
        priors.append(p)
    assert all(p==priors[0] for p in priors)


def test_new_folds_are_matched_and_truncate_at_available_data():
    c=task_config(tasks('reference')[0],'screen');data=bx.load_uccle_multiseries(**c['data'])
    lengths={'1990-11':120,'2000-11':103,'2010-11':63,'2015-11':43,'2020-11':23}
    windows={}
    for t in tasks('sweetspot',tier='screen'):
        if t.kind!='forecast':continue
        cfg=task_config(t,'screen');train,test=validation_splits(data,cfg['validation'])[0]
        assert len(test)==lengths[t.origin]
        assert cfg['validation']['training_ends']==[t.origin] and cfg['validation']['horizon']==120
        assert train.stop==test.start
        assert data.index[test.start]==pd.Timestamp(t.origin[:4]+'-12-01')
        assert windows.setdefault(t.origin,(train.stop,test.stop))==(train.stop,test.stop)
    assert len(windows)==5


def test_variance_partition_and_prior_rms():
    a,b,f=variance_parts(np.array([.01,.04]),np.array([.0004,.0001]),40)
    np.testing.assert_allclose(a,[.4,1.6])
    np.testing.assert_allclose(b,40*39*79/6*np.array([.0004,.0001]))
    np.testing.assert_allclose(f,b/(a+b))
    row=next(x for x in calibration_rows() if x['variant']=='reference')
    assert np.isclose(row['total_trend_innovation_SD_30y_C']**2,120*.01**2+120*119*239/6*.0002**2)


def test_local_sensitivity_matches_weight_derivative():
    rng=np.random.default_rng(31);tau=abs(rng.normal(size=(2,800)))
    values=np.stack([tau+.1*rng.normal(size=tau.shape),tau**2],axis=-1);anchor=1.3
    # Reweight a finite posterior empirical measure by the exact HN density ratio.
    def mean(log_change):
        weights=np.exp(.5*(tau.ravel()/anchor)**2*(1-np.exp(-2*log_change)))
        return np.average(values.reshape(-1,2),axis=0,weights=weights)
    eps=1e-5;finite=(mean(eps)-mean(-eps))/(2*eps)
    # Diagnostic uses the usual N-1 sample covariance, empirical derivative uses N.
    np.testing.assert_allclose(local_derivative(values,tau,anchor)*(tau.size-1)/tau.size,finite,rtol=1e-7)


def entry(rate_shift=0.):
    paths=[];forecasts=[];allocation=[];risk=[]
    for ch in CHANNELS:
        for target in ('level','rate'):
            for time in ('1892-03-01','2000-03-01','2026-06-01'):
                v=rate_shift if target=='rate' else 0.
                paths.append(dict(channel=ch,target=target,time=time,median=v,lower=v-.1,upper=v+.1))
            for years in (10,30):
                forecasts.append(dict(channel=ch,target='level' if target=='level' else 'observation',
                    horizon_years=years,time=f'{2026+years}-06-01',median=1.,lower=0.,upper=2.))
        for years in (10,30):
            allocation.append(dict(channel=ch,target='slope_variance_fraction',horizon_years=years,median=.5,lower=.1,upper=.9))
            risk.append(dict(channel=ch,time=f'{2026+years}-06-01',threshold=35.,direction='>',mean=.2))
    return dict(paths=pd.DataFrame(paths),forecasts=pd.DataFrame(forecasts),allocation=pd.DataFrame(allocation),risk=pd.DataFrame(risk),passed=True)


def test_intervals_missing_outputs_and_numerical_flags_prevent_pass():
    left=entry();right=entry()
    assert all(r['history_stable'] for r in compare_pair(left,right))
    right['paths']['upper']+=.25
    assert not any(r['history_stable'] for r in compare_pair(left,right))
    right=entry();right['risk']=pd.DataFrame()
    assert not any(r['recent_stable'] for r in compare_pair(left,right))
    right=entry();right['passed']=False
    assert not any(r['recent_stable'] for r in compare_pair(left,right))


def test_rectangle_checks_diagonals_and_all_responses():
    chosen=[c for c in cells() if c['A_level'] in (.005,.01) and c['A_slope'] in (.0001,.0002)]
    entries={c['name']:entry(.04*i+.04*j) for c,(i,j) in zip(chosen,[(0,0),(0,1),(1,0),(1,1)])}
    rows=[]
    for a,b in combinations(entries,2):
        rows += [dict(left=a,right=b,**r) for r in compare_pair(entries[a],entries[b])]
    r=rectangles(pd.DataFrame(rows)).iloc[0]
    assert r.complete and not r.history_candidate
    assert not rectangles(pd.DataFrame(rows[:-1])).iloc[0].complete
    for a in entries:entries[a]=entry()
    rows=[dict(left=a,right=b,**r) for a,b in combinations(entries,2) for r in compare_pair(entries[a],entries[b])]
    assert rectangles(pd.DataFrame(rows)).iloc[0].history_candidate


def test_fingerprint_ignores_output_but_retains_scientific_settings():
    c=task_config(tasks('reference')[0],'screen');other=copy.deepcopy(c)
    other['output']='/different/host/path'
    assert fingerprint(c)==fingerprint(other)
    other['priors']['initial_slope_sd']*=2
    assert fingerprint(c)['config_sha256']!=fingerprint(other)['config_sha256']


def test_python_39_login_planning_and_physical_venv_path(tmp_path):
    for file in ('job_scripts/submit.py','job_scripts/paths.py','research/seasonal/job_plan.py','research/seasonal/sweetspot_plan.py'):
        ast.parse(Path(file).read_text(),feature_version=(3,9))
    physical=tmp_path/'physical';(physical/'env/bin').mkdir(parents=True)
    (physical/'env/bin/python').symlink_to(sys.executable)
    alias=tmp_path/'alias';alias.symlink_to(physical)
    r=subprocess.run([sys.executable,'job_scripts/paths.py',str(alias/'env/bin/python'),'--executable'],capture_output=True,text=True,check=True)
    assert r.stdout.strip()==str(physical/'env/bin/python')


def test_hpc_launcher_and_biobot_plan(tmp_path,monkeypatch):
    monkeypatch.setenv('BUCEX_PYTHON',sys.executable)
    monkeypatch.setenv('BUCEX_RESULTS_ROOT',str(tmp_path))
    monkeypatch.setenv('VSC_ARRAY_LIMIT','16')
    r=subprocess.run(['bash','RUN_SWEETSPOT_HPC.sh','--dry-run'],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert '--clusters=gallade' in r.stdout and '--array=1-75%16' in r.stdout
    assert '--cpus-per-task=2' in r.stdout and '--dependency=afterany:ARRAY_JOB_IDS' in r.stdout
    from research.seasonal.overnight import make_queue,budget
    q=make_queue(('screen',),'sweetspot_validation')
    assert len(q)==75 and all(x.cpus==2 and x.memory_gb<=6 for x in q)


def test_validation_pairs_match_dates_and_origins(tmp_path):
    from types import SimpleNamespace
    entries=[]
    for variant,dates,values in [('reference',['2021-03-01','2021-06-01'],[1.,3.]),('other',['2021-03-01','2021-09-01'],[2.,9.])]:
        path=tmp_path/variant;path.mkdir()
        pd.DataFrame(dict(channel='TXm',time=dates,horizon=[1,2] if variant=='reference' else [1,3],
            score='crps',value=values)).to_csv(path/'scores.csv',index=False)
        entries.append(dict(task=SimpleNamespace(variant=variant,origin='2020-11'),report=path,passed=True))
    r=validation_tables(entries)['paired_crps'].iloc[0]
    assert r.matched_cases==1 and r.difference==1. and r.ratio==2.
