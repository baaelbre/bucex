"""Bounded single-response jobs on biobot, with logs, resume and failure accounting."""
from __future__ import annotations
import argparse
from collections import deque
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import bucex as bx
from research.seasonal.jobs import BATCHES, PROJECT, ROOT, tasks, task_config, verify


def run(tier, batch, root, max_jobs, *, dry_run=False, retry_failed=False, series=None):
    if max_jobs < 1:raise ValueError('max_jobs must be positive')
    root=Path(root).resolve();directory=root/tier;directory.mkdir(parents=True,exist_ok=True)
    # One concurrency budget across all batches for this root and tier.
    with (directory/'.local.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError(f'A {tier} runner is already active in {root}; use its existing queue.')
        print(json.dumps(verify(tier,output=directory/'plan'),indent=2),flush=True)
        selected=[t for t in tasks(batch,tier=tier) if not series or t.channel in series]
        chains=max(task_config(t,tier)['mcmc']['chain_workers'] for t in selected)
        print(f'{len(selected)} single-response tasks; at most {max_jobs} fits x {chains} chain workers = {max_jobs*chains} workers.',flush=True)
        if dry_run:
            for t in selected:print(t.id)
            return 0
        logs=directory/'logs';logs.mkdir(exist_ok=True)
        queue=deque(selected);active={};outcomes=[]
        environment=os.environ.copy()
        environment.update({name:'1' for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
        environment['MPLBACKEND']='Agg'
        environment['PYTHONPATH']=str(PROJECT)+os.pathsep+environment.get('PYTHONPATH','')
        def stop(signum,frame):raise KeyboardInterrupt
        old={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
        interrupted=False
        try:
            while queue or active:
                while queue and len(active)<max_jobs:
                    task=queue.popleft();log=(logs/f'{task.id}.log').open('a')
                    log.write('\nSTART '+datetime.now(timezone.utc).isoformat()+'\n');log.flush()
                    args=[sys.executable,'-u','-m','research.seasonal.jobs','--tier',tier,'--task',task.id,'--root',str(root)]
                    if retry_failed:args.append('--retry-failed')
                    proc=subprocess.Popen(args,cwd=PROJECT,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active[proc.pid]=(proc,task,log)
                    print(f'START {task.id} -> {logs/task.id}.log',flush=True)
                for pid,(proc,task,log) in list(active.items()):
                    code=proc.poll()
                    if code is None:continue
                    log.close();del active[pid]
                    outcomes.append(dict(task_id=task.id,exit_code=code))
                    print(f'{"DONE" if code==0 else "FAILED"} {task.id} ({len(outcomes)}/{len(selected)})',flush=True)
                if active:time.sleep(.25)
        except KeyboardInterrupt:
            interrupted=True
            print('Stopping active fits and their chain workers; completed reports are retained.',flush=True)
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
                log.close()
                outcomes.append(dict(task_id=task.id,exit_code=proc.returncode,interrupted=True))
            for s,handler in old.items():signal.signal(s,handler)
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            bx.save_config(dict(tier=tier,batch=batch,max_jobs=max_jobs,interrupted=interrupted,
                outcomes=outcomes,not_started=[t.id for t in queue]),directory/f'local_run_{stamp}.json')
        failed=sum(r['exit_code']!=0 for r in outcomes)
        print(f'Finished {len(outcomes)} tasks; {failed} process failures. Numerical flags remain in each report.',flush=True)
        return 130 if interrupted else 1 if failed else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tier',choices=('screen','paper'),default='screen')
    p.add_argument('--batch',choices=BATCHES,default='experiments')
    p.add_argument('--root',type=Path,default=Path(os.environ.get('BUCEX_RESULTS_ROOT',ROOT)))
    p.add_argument('--max-jobs',type=int,default=int(os.environ.get('BUCEX_MAX_JOBS','6')))
    p.add_argument('--dry-run',action='store_true');p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--series',nargs='+',choices=tuple(bx.UCCLE_INFO))
    a=p.parse_args()
    return run(a.tier,a.batch,a.root,a.max_jobs,dry_run=a.dry_run,retry_failed=a.retry_failed,series=a.series)

if __name__=='__main__':sys.exit(main())
