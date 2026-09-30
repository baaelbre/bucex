"""One resumable BIOBOT queue for both tiers, bounded by CPUs and memory."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import bucex as bx
from research.seasonal.jobs import PROJECT, ROOT, BATCHES, tasks, task_config, fingerprint, verify


@dataclass
class Work:
    tier: str
    task: object
    cpus: int
    memory_gb: float


def budget(task,tier):
    c=task_config(task,tier)
    # Parent merging, two worker results and reporting temporaries can coexist.
    # This reservation is conservative planning, not a hard OS memory limit.
    n=1614 if task.frequency=='monthly' else 538
    dimension=c['model']['period']+1
    raw=8*len(task.series)*dimension*n*c['mcmc']['draws']*c['mcmc']['chains']/1024**3
    return c['mcmc']['chain_workers'],round(max(6.,3.+3.5*raw),1)


def make_queue(tiers,batch):
    result=[]
    for tier in tiers:
        for task in tasks(batch,tier=tier):
            cpus,memory=budget(task,tier)
            result.append(Work(tier,task,cpus,memory))
    def priority(w):
        t=w.task
        if t.frequency=='monthly':return (1,0 if w.tier=='screen' else 1,t.id)
        if t.variant=='reference' and t.kind in ('posterior','pre2019'):
            return (0,0 if w.tier=='paper' else 1,0 if t.kind=='posterior' else 1,t.id)
        if t.kind=='posterior' and t.variant in ('half_slope','double_slope','slope_1e3'):
            return (1,0 if w.tier=='screen' else 1,t.id)
        if w.tier=='screen':return (2,0 if t.kind=='forecast' else 1,t.id)
        return (3,0 if t.kind=='forecast' and t.variant=='reference' else 1 if t.kind=='pre2019' else 2,t.id)
    return sorted(result,key=priority)


def mem_available_gb():
    try:
        line=next(l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
        return int(line.split()[1])/1024**2
    except (OSError,StopIteration,ValueError):return float('inf')


def snapshot(root,tiers,batch):
    rows=[]
    for tier in tiers:
        for task in tasks(batch,tier=tier):
            path=Path(root)/tier/task.id/'task.json'
            record=bx.load_config(path) if path.exists() else {}
            rows.append(dict(tier=tier,task=task.id,status=record.get('status','pending'),
                numerical_status=record.get('numerical_status'),started=record.get('started'),finished=record.get('finished')))
    counts={}
    for row in rows:
        k=row['tier']+'/'+row['status'];counts[k]=counts.get(k,0)+1
    return dict(counts=counts,tasks=rows)


def run(*,tiers=('screen','paper'),batch='all',root=ROOT,cpus=32,memory_gb=150.,max_jobs=32,
        dry_run=False,retry_failed=False,collect=True,prior_simulations=True):
    if cpus<1 or memory_gb<=0 or max_jobs<1:raise ValueError('Resource budgets must be positive.')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'.biobot.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError(f'A BIOBOT queue already owns {root}. Use --status; do not launch a second queue.')
        selected=make_queue(tiers,batch)
        if not selected:raise ValueError('No tasks selected.')
        if any(w.cpus>cpus or w.memory_gb>memory_gb for w in selected):
            raise ValueError('At least one task exceeds the CPU or memory budget. Increase it or use a smaller batch.')
        allowed=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count()
        if cpus>allowed:raise ValueError(f'Requested {cpus} workers but only {allowed} CPUs are available.')
        for tier in tiers:verify(tier,output=root/tier/'plan')
        queue=[];skipped=[]
        for work in selected:
            path=root/work.tier/work.task.id/'task.json'
            if path.exists():
                record=bx.load_config(path)
                identity=dict(bucex_version=bx.__version__,**fingerprint(task_config(work.task,work.tier)))
                if any(record.get(k)!=v for k,v in identity.items()):
                    raise ValueError(f'{path}: provenance changed. Use a new results root.')
                if record['status']=='completed':skipped.append(work);continue
                if record['status']=='running':
                    # Recover only a process on this host proven no longer alive.
                    pid=record.get('pid');host=record.get('hostname')
                    if not retry_failed or host!=socket.gethostname() or not pid:
                        raise RuntimeError(f'{path}: unfinished run; inspect before retrying.')
                    try:os.kill(pid,0)
                    except ProcessLookupError:
                        if not dry_run:
                            record.update(status='failed',error='Worker no longer exists on its recorded host.')
                            bx.save_config(record,path)
                    else:raise RuntimeError(f'{path}: worker PID {pid} is still alive.')
                elif record['status']!='failed' or not retry_failed:
                    raise RuntimeError(f'{path}: recorded failure; inspect logs, then use --retry-failed.')
            queue.append(work)
        print(f'{len(selected)} selected fits; {len(skipped)} already completed; {len(queue)} queued. '
              f'Limits: {cpus} chain workers, {memory_gb:g} GiB reserved, {max_jobs} jobs.',flush=True)
        print('Two chains run in parallel inside each fit. A pooled chain jointly updates all six responses. '
              'Screening and paper draws remain separate. No completion time or convergence is assumed.',flush=True)
        if dry_run:
            if prior_simulations:
                print('Before fitting: joint prior simulations, '+
                    ('11-calibration shared prior suite; 1000 screen / 5000 paper replications.' if batch.startswith('sweetspot')
                     else 'core suite; 2000 screen / 10000 paper replications.'))
            for w in queue:print(w.tier,w.task.id,f'{w.cpus} workers / {w.memory_gb:g} GiB')
            return 0
        if prior_simulations:
            from research.seasonal.prior_simulations import run_suite
            for tier in tiers:
                run_suite(root, tier=tier, suite='sweetspot' if batch.startswith('sweetspot') else 'core',
                    draws=(1000 if tier=='screen' else 5000) if batch.startswith('sweetspot') else None)
        environment=os.environ.copy();environment['MPLBACKEND']='Agg'
        environment.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
        environment['PYTHONPATH']=str(PROJECT)+os.pathsep+environment.get('PYTHONPATH','')
        active={};outcomes=[];interrupted=False;finished_tiers=set()
        def stop(signum,frame):raise KeyboardInterrupt
        old={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
        def record_exit(proc,work,code):
            path=root/work.tier/work.task.id/'task.json'
            if code and path.exists():
                m=bx.load_config(path)
                if m.get('status')=='running':
                    m.update(status='failed',error=f'Queue-owned process exited {code}',finished=datetime.now(timezone.utc).isoformat())
                    bx.save_config(m,path)
            outcomes.append(dict(tier=work.tier,task=work.task.id,exit_code=code))
        try:
            while queue or active:
                used_cpus=sum(w.cpus for _,w,_ in active.values())
                used_mem=sum(w.memory_gb for _,w,_ in active.values())
                while queue and len(active)<max_jobs:
                    candidate=next((i for i,w in enumerate(queue) if w.cpus+used_cpus<=cpus and
                        w.memory_gb+used_mem<=memory_gb and mem_available_gb()>max(4.,w.memory_gb)),None)
                    if candidate is None:break
                    w=queue.pop(candidate);logs=root/w.tier/'logs';logs.mkdir(parents=True,exist_ok=True)
                    log=(logs/(w.task.id+'.log')).open('a');log.write('\nSTART '+datetime.now(timezone.utc).isoformat()+'\n');log.flush()
                    args=[sys.executable,'-u','-m','research.seasonal.jobs','--tier',w.tier,'--task',w.task.id,'--root',str(root)]
                    if retry_failed:args.append('--retry-failed')
                    proc=subprocess.Popen(args,cwd=PROJECT,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active[proc.pid]=(proc,w,log);used_cpus+=w.cpus;used_mem+=w.memory_gb
                    print(f'START {w.tier}/{w.task.id}; active {used_cpus} workers / {used_mem:.1f} GiB',flush=True)
                for pid,(proc,w,log) in list(active.items()):
                    code=proc.poll()
                    if code is None:continue
                    log.close();del active[pid];record_exit(proc,w,code)
                    print(f'{"DONE" if code==0 else "FAILED"} {w.tier}/{w.task.id}; '
                          f'{len(outcomes)+len(skipped)}/{len(selected)} fits finished',flush=True)
                # Summarise a finished tier while the other tier continues sampling.
                for tier in tiers:
                    if collect and tier not in finished_tiers and not any(w.tier==tier for w in queue) and not any(w.tier==tier for _,w,_ in active.values()):
                        # Run after active jobs free a reservation for report building.
                        if sum(w.memory_gb for _,w,_ in active.values())+4>memory_gb:continue
                        path=root/tier/'collection.log'
                        with path.open('a') as output:
                            result=subprocess.run([sys.executable,'-u','-m','research.seasonal.finish','--tier',tier,
                                '--batch',batch,'--root',str(root)],cwd=PROJECT,env=environment,stdout=output,stderr=subprocess.STDOUT)
                        outcomes.append(dict(tier=tier,task='collection',exit_code=result.returncode))
                        finished_tiers.add(tier);print(f'COLLECT {tier}: exit {result.returncode}; {path}',flush=True)
                if not active and queue and mem_available_gb()<min(w.memory_gb for w in queue):
                    print('Waiting for available memory before starting another fit.',flush=True)
                if active or queue:time.sleep(1.)
        except KeyboardInterrupt:interrupted=True
        finally:
            for proc,w,log in active.values():
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            deadline=time.monotonic()+5
            for proc,w,log in active.values():
                try:proc.wait(timeout=max(.01,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    proc.wait()
                log.close();record_exit(proc,w,proc.returncode)
            for s,h in old.items():signal.signal(s,h)
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            bx.save_config(dict(tiers=list(tiers),batch=batch,cpus=cpus,memory_gb=memory_gb,
                interrupted=interrupted,outcomes=outcomes,queued=[w.tier+'/'+w.task.id for w in queue]),root/('queue_'+stamp+'.json'))
        return 130 if interrupted else 1 if any(r['exit_code'] for r in outcomes) else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tier',choices=('screen','paper','both'),default='screen');p.add_argument('--batch',choices=BATCHES,default='all')
    p.add_argument('--root',type=Path,default=Path(os.environ.get('BUCEX_RESULTS_ROOT',ROOT)))
    p.add_argument('--cpus',type=int,default=int(os.environ.get('BUCEX_CPUS','48')))
    p.add_argument('--memory-gb',type=float,default=float(os.environ.get('BUCEX_MEMORY_GB','150')))
    p.add_argument('--max-jobs',type=int,default=int(os.environ.get('BUCEX_MAX_JOBS','32')))
    p.add_argument('--dry-run',action='store_true');p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--status',action='store_true');p.add_argument('--no-collect',action='store_true')
    p.add_argument('--skip-priors',action='store_true',help='Skip automatic prior checks before fitting; standalone command remains available.')
    a=p.parse_args();tiers=('screen','paper') if a.tier=='both' else (a.tier,)
    if a.status:
        result=snapshot(a.root,tiers,a.batch);print(json.dumps(result['counts'],indent=2));return 0
    return run(tiers=tiers,batch=a.batch,root=a.root,cpus=a.cpus,memory_gb=a.memory_gb,max_jobs=a.max_jobs,
        dry_run=a.dry_run,retry_failed=a.retry_failed,collect=not a.no_collect,prior_simulations=not a.skip_priors)

if __name__=='__main__':sys.exit(main())
