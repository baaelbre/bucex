"""Collect the final suite without promoting screening fits to convergence evidence."""
import argparse
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import os
import bucex as bx
from research.seasonal.jobs import tasks, task_config, ROOT
from research.seasonal.final_plan import REFERENCE_ID


def reference_figures(root, tier='paper'):
    from research.seasonal.manuscript_figures import build
    root=Path(root);directory=root/tier/REFERENCE_ID;run=directory/'report'
    if not (directory/'task.json').exists():return None
    record=bx.load_config(directory/'task.json')
    if record.get('status')!='completed':return None
    passed=record.get('final_check')=='passed'
    out=directory/('manuscript_figures' if passed else 'diagnostic_figures_NOT_FINAL')
    pre=root/tier/'final_pre2019_reference_2019-05/report'
    pre_marker=pre.parent/'task.json'
    pre_reports=[pre] if pre_marker.exists() and bx.load_config(pre_marker).get('status')=='completed' else []
    build(run,out,allow_unconverged=not passed,pre2019=pre_reports)
    return out


def finish(root, tier='paper', batch='final_paper', require_complete=False):
    from research.seasonal.collect_jobs import collect
    from research.seasonal.export_results import export
    from research.seasonal.prior_simulations import run_suite
    from research.seasonal.prior_effects import run as prior_effects
    from research.monthly.study_prior import run as monthly_priors
    import pandas as pd
    root=Path(root).resolve()
    out=collect(root,tier,batch=batch,figures=True,require_complete=False)
    selected=tasks(batch,tier=tier)
    groups=[t for t in selected if t.kind=='posterior' and t.frequency=='seasonal']
    for task in groups:
        prior_effects(task_config(task,tier),out/'prior_calibration'/task.variant/task.channel,
                      draws=10000,seed=1983)
    names=list(dict.fromkeys(t.variant for t in groups if t.scope=='shared'))
    if names:
        run_suite(root,tier=tier,suite=','.join(names),draws=2000,seed=1983)
    if any(t.frequency=='monthly' for t in selected):
        # Monthly analytic calibration is exported; the full optional monthly
        # sensitivity prior suite is not launched by the final manuscript batch.
        from research.monthly.study_plan import calibration_rows
        pd.DataFrame(calibration_rows()).to_csv(out/'monthly_calibration.csv',index=False)
    figures=reference_figures(root,tier)
    status=bx.load_config(out/'status.json')
    marker=root/tier/REFERENCE_ID/'task.json'
    reference=bx.load_config(marker) if marker.exists() else {}
    status.update(reference_status=reference.get('status','missing'),
        reference_publication_check=reference.get('final_check','not_passed'),
        check_budget='All nonreference fits use screening budgets; review their numerical flags before interpreting differences.',
        point_summary='Posterior means with pointwise equal-tailed 95% intervals; predictive observation means are finite-draw averages.',
        interpretation='Reference convergence and predictive adequacy are separate questions; completion is not convergence.')
    status['status']=('reference_checked_screens_complete' if reference.get('final_check')=='passed'
                      and not status['missing_or_failed'] else 'needs_review')
    bx.save_config(status,out/'status.json')
    table=pd.read_csv(out/'tasks.csv')
    table['sampling_role']=[t.sampling_role for t in selected]
    table.to_csv(out/'tasks.csv',index=False)
    def link(path,label):
        return '<a href="'+escape(os.path.relpath(path,out),quote=True)+'">'+escape(label)+'</a>'
    links=[link(out/'tasks.csv','Complete task inventory and numerical flags'),
           link(out/'status.json','Reference readiness and suite status'),
           link(root/tier/REFERENCE_ID/'report','Reference tables, diagnostics, risk and forecasts'),
           link(out/'pre2019_risk_sensitivity.csv','Pre-2019 risk sensitivity'),
           link(out/'calendar_35y','Held-out forecasts, coverage, CRPS and threshold counts'),
           link(out/'posterior','Posterior sensitivity and private/fixed comparisons'),
           link(root/tier/'prior_simulations','Prior simulations'),
           link(out/'block_comparison','Monthly versus seasonal matched-target validation')]
    pictures=[]
    if figures:
        for image in sorted(figures.glob('*.png')):
            pictures.append('<figure><img loading="lazy" style="max-width:1100px;width:100%" src="'+
                escape(os.path.relpath(image,out),quote=True)+'"><figcaption>'+escape(image.stem)+'</figcaption></figure>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>BUCEX 1.9.8.3 final runs</title>'
        '<body style="font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px">'
        '<h1>BUCEX 1.9.8.3 final runs</h1><p>'+escape(status['interpretation'])+'</p><p>'+
        escape(str(status['completed']))+' / '+str(status['expected'])+' fits completed. Reference check: <b>'+
        escape(status['reference_publication_check'])+'</b>. Screening numerical flags: '+str(len(status['numerical_flags']))+
        '.</p><p>'+escape(status['check_budget'])+'</p><p>'+escape(status['point_summary'])+'</p><ul>'+
        ''.join('<li>'+s+'</li>' for s in links)+'</ul>'+''.join(pictures)+table.to_html(index=False,escape=True)+'</body>',encoding='utf-8')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    archive,count=export(root,tier,root/'exports'/f'bucex1983_final_{stamp}.zip',figures=True)
    print(f'{out / "index.html"}\n{archive}: {count} review files; fit archives remain on the compute host.',flush=True)
    if require_complete and (status['missing_or_failed'] or status['numerical_flags']
                             or reference.get('final_check')!='passed'):
        raise RuntimeError('Final results require review; see '+str(out/'status.json'))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--tier',default='paper',choices=('paper','screen'))
    p.add_argument('--reference-only',action='store_true')
    a=p.parse_args()
    print(reference_figures(a.root,a.tier) if a.reference_only else finish(a.root,a.tier))
