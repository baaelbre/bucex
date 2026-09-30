"""Write the exact final-run inventory and transparent runtime planning estimates."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from research.seasonal.jobs import tasks,task_config,verify
from research.seasonal.overnight import make_queue
from research.seasonal.job_plan import plan


def audit(output, *, cpus=48, memory_gb=150., check=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    selected=tasks('final_paper',tier='paper')
    rows=[]
    for t in selected:
        c=task_config(t,'paper');m=c['mcmc'];p=c['priors']
        rows.append(dict(task=t.id,kind=t.kind,variant=t.variant,channel=t.channel,
            frequency=t.frequency,scope=t.scope,sampling_role=t.sampling_role,
            chains=m['chains'],warmup=m['warmup'],retained_per_chain=m['draws'],
            forecast_draws=c['forecast_draws'],validation_draws=c['validation']['draws'],
            origin=t.origin,horizon=(c['validation']['horizon'] if t.kind in ('forecast','block') else c['forecast_horizon']),
            A_or_SD_alpha=p['innovation_sd']['level'],A_or_SD_beta=p['innovation_sd']['trend'],
            A_or_SD_gamma=p['innovation_sd']['season'],initial_slope_SD=p['initial_slope_sd'],
            interval=c['credible_interval'],max_rhat=c['diagnostic_thresholds']['max_rhat'],
            min_ESS=c['diagnostic_thresholds']['min_ess']))
    with (output/'FINAL_RUNS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # These are extrapolations, not newly measured 1.9.8.3 runtimes.
    # Seasonal shared 2x(1000+2000): 66.40 min BIOBOT / 79.81 min Gallade.
    # Private fits approximate 1/6 of a pooled sweep, with 15% overhead.
    # Monthly work scales conservatively with observations AND state dimension.
    queue=make_queue(('paper',),'final_paper')
    estimates={}
    for label,base in [('biobot',66.40),('gallade',79.81)]:
        jobs=[]
        for w in queue:
            t=w.task;c=task_config(t,'paper');m=c['mcmc']
            end=int(t.origin[:4]) if t.origin else 2019 if t.kind=='pre2019' else 2026
            record_fraction=(end-1892+1)/135
            scale=(m['warmup']+m['draws'])/3000
            if t.scope!='shared':scale*=1.15/6
            if t.frequency=='monthly':scale*=3*13/5
            duration=base*scale*record_fraction
            jobs.append([w,duration])
        waiting=list(jobs);active=[];clock=0.
        while waiting or active:
            while waiting:
                used_cpu=sum(w.cpus for _,w in active);used_mem=sum(w.memory_gb for _,w in active)
                k=next((i for i,(w,d) in enumerate(waiting) if used_cpu+w.cpus<=cpus
                        and used_mem+w.memory_gb<=memory_gb and len(active)<24),None)
                if k is None:break
                w,d=waiting.pop(k);active.append((clock+d,w))
            if not active:raise ValueError('A task exceeds the requested planning capacity.')
            clock=min(end for end,_ in active);active=[(end,w) for end,w in active if end>clock]
        estimates[label]=dict(reference_minutes=2*base,
            queued_compute_minutes_at_requested_capacity=round(clock,1),
            CPU_hours=round(sum(w.cpus*d for w,d in jobs)/60,1))
    result=dict(version='1.9.8.3',fits=len(selected),HPC_groups=len(plan('paper','final_paper')),
        by_kind=dict(Counter(t.kind for t in selected)),by_scope=dict(Counter(t.scope for t in selected)),
        budgets=dict(reference='4 chains x (2000 warm-up + 4000 retained)',checks='2 chains x (1000 + 1000)',
            monthly_checks='2 chains x (500 + 500)'),
        assumed_capacity=dict(chain_workers=cpus,memory_gb=memory_gb),runtime_estimates=estimates,
        limitations=['Runtime estimates extrapolate prior runs; they do not include scheduler waiting, contention, setup or final collection.',
            'Monthly state-space cost scaling is deliberately conservative and not a timing measurement.',
            'The 3-4 hour aim depends on resources being available; no chain is killed at that target.',
            'Passing time or draw budgets is not evidence of convergence.'])
    if check:result['configuration_check']=verify('paper',batch='final_paper')['status']
    (output/'FINAL_PLAN.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('release_checks'))
    p.add_argument('--cpus',type=int,default=48);p.add_argument('--memory-gb',type=float,default=150.)
    a=p.parse_args();audit(a.output,cpus=a.cpus,memory_gb=a.memory_gb)
