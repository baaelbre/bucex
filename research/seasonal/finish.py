"""Collect a completed/partial tier, render review figures, and export evidence."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import bucex as bx
from research.seasonal.jobs import ROOT,BATCHES,tasks,task_config
from research.seasonal.collect_jobs import collect
from research.seasonal.prior_effects import run as prior_effects
from research.seasonal.manuscript_figures import build
from research.seasonal.export_results import export


def finish(root,tier,batch='all',require_complete=False):
    root=Path(root)
    focused=batch.startswith(('sweetspot','monthly_'))
    if batch.startswith('monthly_'):
        from research.monthly.study_report import build as build_monthly
        out=build_monthly(root,tier=tier,batch=batch)
        status=bx.load_config(out/'status.json')
        if require_complete and (status['completed']!=status['expected'] or status['numerically_passed']!=status['expected']):
            raise RuntimeError('Incomplete or numerically unchecked monthly results; inspect '+str(out/'status.json'))
    elif focused:
        from research.seasonal.sweetspot_report import build as build_sweetspot
        out=build_sweetspot(root,tier=tier,batch=batch)
        status=bx.load_config(out/'status.json')
        if require_complete and (tier!='paper' or status['numerically_passed']!=status['expected'] or status['completed']!=status['expected']):
            raise RuntimeError('Complete numerically checked paper results are required; inspect '+str(out/'status.json'))
    else:
        out=collect(root,tier,batch=batch,figures=True,require_complete=require_complete)
    for task in ([] if focused else tasks('comparison',tier=tier)):
        prior_effects(task_config(task,tier),out/'prior_calibration'/task.variant,draws=50000,seed=195)
    reference=root/tier/'posterior_reference/report'
    marker=root/tier/'posterior_reference/task.json'
    if not focused and marker.exists() and bx.load_config(marker).get('status')=='completed':
        pre=[]
        for task in tasks('pre2019',tier=tier):
            path=root/tier/task.id
            if (path/'task.json').exists() and bx.load_config(path/'task.json').get('status')=='completed':pre.append(path/'report')
        passed=bx.load_config(reference/'convergence.json').get('status')=='passed_numerical_checks'
        figures=out/('manuscript_figures' if passed and tier=='paper' else 'diagnostic_figures_NOT_FINAL')
        build(reference,figures,allow_unconverged=not passed,pre2019=pre)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    archive,count=export(root,tier,root/'exports'/f'bucex1982_{tier}_{batch}_{stamp}.zip',figures=True)
    print(f'{out}\n{archive}: {count} review files; posterior archives remain in their fit directories.',flush=True)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--tier',choices=('screen','paper'),required=True);p.add_argument('--batch',choices=BATCHES,default='all')
    p.add_argument('--require-complete',action='store_true')
    a=p.parse_args();finish(a.root,a.tier,a.batch,a.require_complete)
