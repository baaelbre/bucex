"""Compute-node startup gate: six tiny independent fits, each with two chains."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
import bucex as bx
from research.monthly.run import run
from research.seasonal.jobs import CONFIG,verify
from research.seasonal.bundles import allocation


def worker(name,output):
    c=bx.load_config(CONFIG/'main.json');c['data'].update(series=[name],start='2021-03')
    c['mcmc'].update(chains=2,chain_workers=2,warmup=3,draws=8,progress=False,seed=193+bx.UCCLE_SERIES.index(name))
    c.update(figures=False,contrasts=None,forecast_horizon=8,forecast_draws=24,predictive_check_draws=12,prior_draws=80)
    run(c,directory=output)
    fit=bx.load_fit(output/name/'fit.bucex')
    if fit.priors.shrinkage is not None or fit.model.copula is not None or len(fit.channel_names)!=1:
        raise RuntimeError('Probe did not use a fixed-prior independent fit.')
    if any(k.startswith('shrinkage.') for k in fit.parameter_draws):raise RuntimeError('Unexpected sampled shrinkage hyperparameter.')
    if not np.isfinite(fit.state_draws).all():raise RuntimeError('Nonfinite probe state draws.')
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--response',choices=bx.UCCLE_SERIES);a=p.parse_args()
    if a.response:return worker(a.response,a.root)
    allocation(12)
    target=a.root/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');target.mkdir(parents=True)
    print('Probe:',target,'Python:',sys.executable,platform.python_version(),flush=True)
    bx.save_config(verify(os.environ.get('BUCEX_TIER','screen')),target/'configuration_checks.json')
    def launch(name):
        with (target/(name+'.log')).open('w') as log:
            result=subprocess.run([sys.executable,'-u','-m','research.seasonal.probe','--response',name,'--root',str(target/name)],stdout=log,stderr=subprocess.STDOUT)
        print(name,'passed' if result.returncode==0 else 'FAILED',flush=True)
        return dict(response=name,exit_code=result.returncode)
    with ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(launch,bx.UCCLE_SERIES))
    passed=all(r['exit_code']==0 for r in results)
    bx.save_config(dict(status='passed' if passed else 'failed',version=bx.__version__,python=sys.executable,
        host=platform.node(),results=results,note='Startup smoke only; not evidence of posterior convergence.'),target/'probe.json')
    print('Probe', 'PASSED' if passed else 'FAILED', '; reports:',target,flush=True)
    return 0 if passed else 1

if __name__=='__main__':sys.exit(main())
