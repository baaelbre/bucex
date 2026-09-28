"""Submit experiment bundles using only the login host's Python standard library."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from research.seasonal.job_plan import plan,BATCHES,RESOURCES


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('tier',choices=('screen','paper'));p.add_argument('batch',choices=BATCHES,nargs='?',default='experiments')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--scheduler',choices=('slurm','pbs'),default=os.environ.get('BUCEX_SCHEDULER','slurm'))
    a=p.parse_args(argv)
    cap=int(os.environ.get('VSC_ARRAY_LIMIT','4'))
    if cap<1:p.error('VSC_ARRAY_LIMIT must be positive')
    # Do not execute a Gallade Python binary on the login node.
    request=os.environ.get('BUCEX_PYTHON',str(PROJECT/'bucex_env_gallade_py311_195/bin/python'))
    found=shutil.which(request) if '/' not in request else request
    if not found:p.error('Cannot locate BUCEX_PYTHON: '+request)
    path=Path(found).absolute();compute_python=str(path.parent.resolve()/path.name)
    if not a.dry_run and not os.access(compute_python,os.X_OK):p.error('Compute Python is not executable; finish SETUP_HPC_ENV.sh first: '+compute_python)
    root=Path(os.environ.get('BUCEX_RESULTS_ROOT',str(PROJECT/'results/serra_195'))).resolve()
    logs=PROJECT/'job_scripts/logs';logs.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,BUCEX_PROJECT_ROOT=str(PROJECT),BUCEX_PYTHON=compute_python,
             BUCEX_RESULTS_ROOT=str(root),BUCEX_TIER=a.tier,BUCEX_BATCH=a.batch,
             BUCEX_RETRY_FAILED='1' if a.retry_failed else '0')
    default_setup=PROJECT/'bucex_env_gallade_py311_195/environment.sh'
    if 'BUCEX_ENV_SETUP' not in env and default_setup.is_file():env['BUCEX_ENV_SETUP']=str(default_setup)
    if not a.dry_run and env.get('BUCEX_ENV_SETUP') and not Path(env['BUCEX_ENV_SETUP']).is_file():
        p.error('Cannot read BUCEX_ENV_SETUP: '+env['BUCEX_ENV_SETUP'])
    ids=[];commands=[]
    def submit(command,resource):
        command=list(map(str,command));commands.append(command)
        print(shlex.join(command),flush=True)
        if a.dry_run:return 'PROBE_JOB_ID'
        result=subprocess.run(command,cwd=PROJECT,env=dict(env,BUCEX_RESOURCE=resource),text=True,capture_output=True)
        if result.stderr:print(result.stderr,file=sys.stderr,end='',flush=True)
        result.check_returncode()
        raw=result.stdout.strip();print(raw,flush=True)
        id=raw.split(';')[0].splitlines()[-1].strip()
        if not id or (a.scheduler=='slurm' and not id.isdigit()):raise RuntimeError('Unrecognized scheduler job ID: '+raw)
        ids.append(dict(resource=resource,job_id=id));return id
    groups=plan(a.tier,a.batch)
    print('{}: {} HPC experiments; {} fits; shared experiments use 2 CPUs, separate bundles use 12 CPUs; array cap {}'.format(
        a.tier,len(groups),sum(g['parallel_fits'] for g in groups),cap),flush=True)
    account=os.environ.get('VSC_PROJECT');cluster=os.environ.get('VSC_CLUSTER');partition=os.environ.get('VSC_PARTITION')
    if a.scheduler=='slurm':
        base=['sbatch','--parsable','--chdir='+str(PROJECT),'--nodes=1','--ntasks=1','--export=ALL']
        if account:base+=['--account='+account]
        if cluster:base+=['--clusters='+cluster]
        if partition:base+=['--partition='+partition]
        probe=submit(base+['--job-name=bx195_probe','--cpus-per-task=12','--mem=16G','--time=00:20:00',
            '--output='+str(logs/'bx195_probe_%j.log'),PROJECT/'job_scripts/probe.slurm'],'probe')
        for resource in RESOURCES:
            group=[g for g in groups if g['resource']==resource]
            if not group:continue
            r=group[0]
            if any((g['cpus'],g['memory_gb'],g['hours'])!=(r['cpus'],r['memory_gb'],r['hours']) for g in group):raise ValueError('Mixed resources in one array')
            submit(base+['--job-name=bx195_'+a.tier+'_'+resource,'--array=1-{}%{}'.format(len(group),cap),
                '--cpus-per-task='+str(r['cpus']),'--mem='+str(r['memory_gb'])+'G','--time='+str(r['hours'])+':00:00',
                '--dependency=afterok:'+probe,'--kill-on-invalid-dep=yes',
                '--output='+str(logs/(a.tier+'_'+a.batch+'_'+resource+'_%A_%a.log')),PROJECT/'job_scripts/seasonal_array.slurm'],resource)
    else:
        base=['qsub','-V']+(['-A',account] if account else [])
        probe=submit(base+['-N','bx195_probe','-l','nodes=1:ppn=12,mem=16gb,walltime=00:20:00',
            '-o',str(logs),'-e',str(logs),PROJECT/'job_scripts/probe.pbs'],'probe')
        for resource in RESOURCES:
            group=[g for g in groups if g['resource']==resource]
            if not group:continue
            r=group[0]
            submit(base+['-N','bx195_'+a.tier[0]+'_'+resource[:3],'-t','1-{}%{}'.format(len(group),cap),
                '-W','depend=afterok:'+probe,'-l','nodes=1:ppn={},mem={}gb,walltime={}:00:00'.format(r['cpus'],r['memory_gb'],r['hours']),
                '-o',str(logs),'-e',str(logs),PROJECT/'job_scripts/seasonal_array.pbs'],resource)
    if not a.dry_run:
        target=root/a.tier/'submissions';target.mkdir(parents=True,exist_ok=True)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        (target/(stamp+'.json')).write_text(json.dumps(dict(tier=a.tier,batch=a.batch,job_ids=ids,commands=commands,
            compute_python=compute_python,env_setup=env.get('BUCEX_ENV_SETUP'),experiment_plan=groups),indent=2)+'\n')
    return 0

if __name__=='__main__':sys.exit(main())
