"""Compute-node gate: tiny two-chain fits of all three shrinkage specifications."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import os
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
import bucex as bx
from research.monthly.run import run
from research.seasonal.jobs import verify,tasks,task_config,result_directory
from research.seasonal.bundles import allocation


def worker(task_id,output):
    task=next(t for t in tasks('comparison',tier='screen') if t.id==task_id)
    c=task_config(task,'screen');c['data']['start']='2021-03'
    c['mcmc'].update(chains=2,chain_workers=2,warmup=3,draws=8,progress=False)
    c.update(figures=False,forecast_horizon=8,forecast_draws=24,predictive_check_draws=12,prior_draws=80)
    output.mkdir(parents=True,exist_ok=True)
    run(c,directory=output/'report')
    fit=bx.load_fit(result_directory(output,task)/'fit.bucex')
    if fit.model.copula is not None or tuple(fit.channel_names)!=task.series:
        raise RuntimeError('Probe response/dependence declaration differs from the plan.')
    expected_scope={'fixed':None,'independent':bx.IndependentShrinkage,'shared':bx.SharedShrinkage}[task.scope]
    if expected_scope is None:
        if fit.priors.shrinkage is not None:raise RuntimeError('Fixed probe unexpectedly has a hierarchy.')
    elif type(fit.priors.shrinkage) is not expected_scope:
        raise RuntimeError('Wrong probe hierarchy.')
    else:
        if fit.priors.shrinkage.scale_parameterization!='normal_sd':raise RuntimeError('Wrong SD convention.')
        if 'initial_slope' in fit.priors.shrinkage.anchors:raise RuntimeError('Initial rates are being pooled.')
    if not np.isfinite(fit.state_draws).all():raise RuntimeError('Nonfinite probe states.')
    if not all(np.isfinite(v).all() for v in fit.parameter_draws.values()):raise RuntimeError('Nonfinite parameters.')
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--task');p.add_argument('--max-parallel',type=int,default=6);a=p.parse_args()
    if not 1<=a.max_parallel<=6:p.error('--max-parallel must be between 1 and 6')
    if a.task:return worker(a.task,a.root)
    allocation(2*a.max_parallel)
    target=a.root/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');target.mkdir(parents=True)
    print('Probe:',target,'Python:',sys.executable,platform.python_version(),'BUCEX',bx.__version__,flush=True)
    bx.save_config(verify(os.environ.get('BUCEX_TIER','screen')),target/'configuration_checks.json')
    def launch(task):
        with (target/(task.id+'.log')).open('w') as log:
            result=subprocess.run([sys.executable,'-u','-m','research.seasonal.probe','--task',task.id,
                '--root',str(target/task.id)],stdout=log,stderr=subprocess.STDOUT)
        print(task.id,'passed' if result.returncode==0 else 'FAILED','log:',target/(task.id+'.log'),flush=True)
        return dict(task_id=task.id,scope=task.scope,exit_code=result.returncode)
    results=[]
    for scope in ('shared','independent','fixed'):
        group=[t for t in tasks('comparison',tier='screen') if t.scope==scope]
        with ThreadPoolExecutor(max_workers=a.max_parallel) as pool:results.extend(pool.map(launch,group))
    passed=all(r['exit_code']==0 for r in results)
    bx.save_config(dict(status='passed' if passed else 'failed',version=bx.__version__,python=sys.executable,
        host=platform.node(),max_parallel=a.max_parallel,results=results,note='Startup smoke only; short chains do not establish convergence.'),target/'probe.json')
    print('Probe','PASSED' if passed else 'FAILED','; reports:',target,flush=True)
    return 0 if passed else 1

if __name__=='__main__':sys.exit(main())
