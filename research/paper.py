"""List or execute the complete reproducibility plan from portable Python commands."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shlex
import subprocess
import sys
from .run import ANALYSES
from .sensitivity import plan


def commands(profile, root, workers, groups):
    tasks = []
    def add(name, module, args):
        tasks.append((name,[sys.executable,'-m','research.'+module,*map(str,args)]))
    if 'analyses' in groups:
        for name in ANALYSES:
            add(name,'run',['--analysis',name,'--profile',profile,'--output',root/name/profile,'--workers',workers])
    for key, config in [('sensitivity','sensitivity.json'), ('private_sensitivity','private_sensitivity.json')]:
        if key in groups:
            for variant in plan(Path(__file__).parent/'config'/config):
                add(variant['name'],'sensitivity',['--config',Path(__file__).parent/'config'/config,'--profile',profile,
                    '--output',root/key,'--variant',variant['name'],'--workers',workers])
    if 'validation' in groups:
        add('validation','validation',['--profile',profile,'--output',root/'validation','--workers',workers])
        add('validation_recent','validation',['--recent','--profile',profile,'--output',root/'validation_recent','--workers',workers])
    if 'pre2019' in groups:
        add('pre2019','pre2019',['--profile',profile,'--output',root/'pre2019','--workers',workers])
    if 'priors' in groups:
        add('prior_calibration','prior_calibration',['--profile',profile,'--output',root/'prior_calibration'])
    return tasks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=('smoke','screen','paper'),default='screen')
    p.add_argument('--output',type=Path,default=Path('results/paper_suite'))
    p.add_argument('--groups',nargs='+',choices=('analyses','sensitivity','private_sensitivity','validation','pre2019','priors'),
                   default=['analyses','sensitivity','private_sensitivity','validation','pre2019','priors'])
    p.add_argument('--workers',type=int,default=2,help='Chain processes per fitting job.')
    p.add_argument('--jobs',type=int,default=1,help='Concurrent experiments; budget jobs × workers CPUs.')
    p.add_argument('--list',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.workers < 1 or args.jobs < 1:
        p.error('workers and jobs must be positive.')
    tasks=commands(args.profile,args.output,args.workers,set(args.groups))
    if args.list or args.dry_run:
        for name,command in tasks:
            print(name+': '+shlex.join(command))
        return
    logroot=args.output/'logs'/args.profile
    logroot.mkdir(parents=True,exist_ok=True)
    def execute(task):
        name,command=task
        log=logroot/(name+'.log')
        with log.open('w') as stream:
            stream.write(shlex.join(command)+'\n');stream.flush()
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=False)
        return dict(name=name,command=command,exit_code=result.returncode,log=str(log))
    results=[]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures={pool.submit(execute,task):task[0] for task in tasks}
        for future in as_completed(futures):
            result=future.result();results.append(result)
            print(f'{result["name"]}: exit {result["exit_code"]}; {result["log"]}',flush=True)
            (logroot/'status.json').write_text(json.dumps(results,indent=2)+'\n')
    if any(r['exit_code'] for r in results):
        raise SystemExit('Some experiments failed; see logs and status.json. Completed fits are reusable.')
    print('Fits complete. Regenerate sensitivity/comparison figures with research.sensitivity --compare and research.compare.')


if __name__=='__main__': main()
