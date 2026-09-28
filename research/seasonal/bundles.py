"""Run the six independent response fits inside one allocated HPC experiment."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import bucex as bx
from research.seasonal.jobs import ROOT,PROJECT,tasks,task_config
from research.seasonal.job_plan import plan,BATCHES


def members(group,tier):
    selected=[t for t in tasks('all',tier=tier) if t.id.replace('_'+t.channel,'',1)==group['id']]
    if [t.channel for t in selected]!=group['series']:raise ValueError('Runtime tasks differ from the submission plan: '+group['id'])
    for task in selected:
        c=task_config(task,tier)
        if c['mcmc']['chains']!=group['chains'] or c['mcmc']['chain_workers']!=group['chain_workers']:
            raise ValueError('Chain budget differs from the submission plan.')
    return selected


def allocation(required):
    for key in ('SLURM_CPUS_PER_TASK','PBS_NP','PBS_NUM_PPN','NSLOTS'):
        value=os.environ.get(key)
        if value:
            count=int(value)
            if count<required:raise RuntimeError(f'{key}={count}, but this experiment needs {required} chain workers.')
            break
    if hasattr(os,'sched_getaffinity') and len(os.sched_getaffinity(0))<required:
        raise RuntimeError('CPU affinity allows fewer CPUs than response x chain workers.')


def run(group,*,tier='screen',root=ROOT,retry_failed=False):
    selected=members(group,tier);allocation(group['required_workers'])
    root=Path(root).resolve();directory=root/tier/'bundles'/group['id'];directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This experiment already has an active runner.')
        meta=dict(group=group,tier=tier,started=datetime.now(timezone.utc).isoformat(),status='running',outcomes=[])
        bx.save_config(meta,directory/'bundle.json')
        logs=root/tier/'logs';logs.mkdir(exist_ok=True)
        environment=os.environ.copy()
        environment.update({key:'1' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
        environment['MPLBACKEND']='Agg';environment['PYTHONPATH']=str(PROJECT)+os.pathsep+environment.get('PYTHONPATH','')
        active={};interrupted=False
        def stop(signum,frame):raise KeyboardInterrupt
        previous={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
        print(f'{group["id"]}: {len(selected)} separate responses x {group["chain_workers"]} chains = {group["required_workers"]} workers',flush=True)
        try:
            for task in selected:
                log=(logs/(task.id+'.log')).open('a');log.write('\nSTART '+meta['started']+'\n');log.flush()
                args=[sys.executable,'-u','-m','research.seasonal.jobs','--task',task.id,'--tier',tier,'--root',str(root)]
                if retry_failed:args.append('--retry-failed')
                proc=subprocess.Popen(args,cwd=PROJECT,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[proc.pid]=(proc,task,log)
            while active:
                for pid,(proc,task,log) in list(active.items()):
                    code=proc.poll()
                    if code is None:continue
                    log.close();del active[pid]
                    meta['outcomes'].append(dict(task_id=task.id,channel=task.channel,exit_code=code))
                    print(f'{task.channel}: {"completed" if code==0 else "FAILED"}; {logs/(task.id+".log")}',flush=True)
                    bx.save_config(meta,directory/'bundle.json')
                if active:time.sleep(.25)
        except KeyboardInterrupt:interrupted=True
        except BaseException as error:
            meta['error']=str(error);raise
        finally:
            for proc,task,log in active.values():
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            deadline=time.monotonic()+5
            for proc,task,log in active.values():
                try:proc.wait(timeout=max(.01,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    proc.wait()
                log.close();meta['outcomes'].append(dict(task_id=task.id,channel=task.channel,exit_code=proc.returncode,interrupted=True))
            for s,handler in previous.items():signal.signal(s,handler)
            meta['finished']=datetime.now(timezone.utc).isoformat()
            failed=interrupted or len(meta['outcomes'])!=len(selected) or any(r['exit_code']!=0 for r in meta['outcomes'])
            meta['status']='interrupted' if interrupted else 'failed' if failed else 'completed'
            bx.save_config(meta,directory/'bundle.json')
        return 130 if interrupted else 1 if failed else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tier',choices=('screen','paper'),default='screen');p.add_argument('--batch',choices=BATCHES,default='experiments')
    p.add_argument('--resource',choices=('all','standard','long'),default='all');p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--index',type=int);p.add_argument('--bundle');p.add_argument('--list',action='store_true');p.add_argument('--retry-failed',action='store_true')
    a=p.parse_args();groups=plan(a.tier,a.batch,a.resource)
    if a.list:
        for i,g in enumerate(groups,1):print(i,g['id'],g['parallel_responses'],g['chains'],g['cpus'])
        return 0
    selected=(next((g for g in plan(a.tier,'all') if g['id']==a.bundle),None) if a.bundle else
              groups[a.index-1] if a.index and 1<=a.index<=len(groups) else None)
    if selected is None:p.error('Supply a valid --index or --bundle')
    return run(selected,tier=a.tier,root=a.root,retry_failed=a.retry_failed)

if __name__=='__main__':sys.exit(main())
