"""Time short full-record chains on the actual host; no scientific inference."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import time
import bucex as bx
from research.seasonal.jobs import tasks,task_config,ROOT
from research.monthly.models import joint_model,fit_options


def run(output,*,warmup=20,draws=40,cpus=32):
    task=next(t for t in tasks('reference') if t.variant=='reference')
    c=task_config(task,'screen');c['mcmc'].update(warmup=warmup,draws=draws,progress=False)
    data=bx.load_uccle_multiseries(**c['data']);model,priors=joint_model(data,c)
    start=time.monotonic()
    fit=bx.fit(data,model,priors=priors,**fit_options(c,family=model.family))
    elapsed=time.monotonic()-start;per_iteration=elapsed/(warmup+draws)
    forecasts={}
    for tier in ('screen','paper'):
        settings=[task_config(t,tier)['mcmc'] for t in tasks('all',tier=tier)]
        work=sum(m['chains']*(m['warmup']+m['draws']) for m in settings)
        forecasts[tier]=dict(optimistic_sampling_hours=work*per_iteration/cpus/3600,
            longest_fit_hours=max(m['warmup']+m['draws'] for m in settings)*per_iteration/3600)
    result=dict(version=bx.__version__,wall_seconds=elapsed,two_parallel_chains=True,
        warmup=warmup,draws=draws,seconds_per_parallel_iteration=per_iteration,
        projected=forecasts,cpu_budget=cpus,
        note='Rough planning estimate only: short startup/warmup, memory limits, other users, heavier-tail variants, monthly fits and reporting can change runtime. These draws are not used in either tier.')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    bx.save_config(result,output/'benchmark.json');print(result,flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'benchmark')
    p.add_argument('--warmup',type=int,default=20);p.add_argument('--draws',type=int,default=40);p.add_argument('--cpus',type=int,default=32)
    a=p.parse_args();run(a.output,warmup=a.warmup,draws=a.draws,cpus=a.cpus)
