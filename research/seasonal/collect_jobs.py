"""Collect a named 1.9.0 batch without requiring deferred validation jobs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import pandas as pd
import bucex as bx
from research.seasonal.jobs import ROOT, BATCHES, tasks, task_config, fingerprint


def merge_folds(paths, destination):
    """Combine disjoint single-origin reports without discarding paired cases."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    folds=pd.concat([pd.read_csv(Path(p)/'joint/folds.csv') for p in paths],ignore_index=True)
    if folds.origin.duplicated().any():raise ValueError(f'{destination}: duplicate validation origin')
    for name in ('scores','held_out_pit','coverage_by_case','predictions','event_counts',
                 'threshold_cases','joint_log_scores','compound_heat_scores'):
        inputs=[Path(p)/'joint'/f'{name}.csv' for p in paths]
        if all(p.exists() for p in inputs):
            pd.concat([pd.read_csv(p) for p in inputs],ignore_index=True).to_csv(destination/f'{name}.csv',index=False)
    folds.to_csv(destination/'folds.csv',index=False)
    for path in paths:
        for pattern in ('convergence_*.json','shared_shrinkage_*.csv'):
            for file in (Path(path)/'joint').glob(pattern):shutil.copy2(file,destination/file.name)


def collect(root=ROOT,tier='screen',*,batch='posterior',figures=False,require_complete=False):
    root=Path(root);expected=tasks(batch,tier=tier);records={};rows=[];mismatch=[]
    for task in expected:
        manifest=root/tier/task.id/'task.json'
        record=bx.load_config(manifest) if manifest.exists() else dict(status='missing')
        if manifest.exists():
            identity=dict(bucex_version=bx.__version__,**fingerprint(task_config(task,tier)))
            if any(record.get(k)!=v for k,v in identity.items()):
                mismatch.append(task.id);record=dict(record,status='provenance_mismatch')
        records[task.id]=record
        rows.append(dict(task_id=task.id,kind=task.kind,variant=task.variant,origin=task.origin,
            status=record['status'],numerical_status=record.get('numerical_status'),result=record.get('result')))
    incomplete=[t.id for t in expected if records[t.id]['status']!='completed']
    flagged=[t.id for t in expected if records[t.id]['status']=='completed' and
             (records[t.id].get('numerical_status')!='passed_numerical_checks' or
              records[t.id].get('final_check')=='failed')]
    out=root/tier/'collected'/batch;out.mkdir(parents=True,exist_ok=True)
    report=dict(version=bx.__version__,tier=tier,batch=batch,expected=len(expected),
        completed=len(expected)-len(incomplete),missing_or_failed=incomplete,
        numerical_flags=flagged,provenance_mismatch=mismatch,
        status='passed' if not incomplete and not flagged and tier=='paper' else 'incomplete_or_screening')
    bx.save_config(report,out/'status.json');pd.DataFrame(rows).to_csv(out/'tasks.csv',index=False)
    if require_complete and (incomplete or flagged or tier!='paper'):
        raise RuntimeError(f'{len(incomplete)} incomplete, {len(flagged)} numerically flagged; see {out}/status.json')
    for kind in ('posterior','pre2019'):
        completed=[t for t in expected if t.kind==kind and records[t.id]['status']=='completed']
        if not any(t.variant=='reference' for t in completed):continue
        posterior={t.variant:{ch:Path(records[t.id]['result']) for ch in bx.UCCLE_SERIES} for t in completed}
        split_figures = figures and kind=='posterior' and len(posterior)>10
        bx.SensitivityReport(posterior_runs=posterior,baseline='reference',block_frequency='seasonal').save(
            out/kind,figures=figures and not split_figures)
        if split_figures:
            for study in dict.fromkeys(t.study for t in completed):
                variants = ['reference']+[t.variant for t in completed if t.study==study and t.variant!='reference']
                if len(variants)<2:continue
                selected = {name:posterior[name] for name in variants}
                bx.SensitivityReport(posterior_runs=selected,baseline='reference',block_frequency='seasonal').save(
                    out/kind/'by_study'/study,figures=True)
    forecasts=[t for t in expected if t.kind=='forecast']
    matched=[]
    for variant in sorted({t.variant for t in forecasts}):
        group=[t for t in forecasts if t.variant==variant]
        if not all(records[t.id]['status']=='completed' for t in group):continue
        merge_folds([Path(records[t.id]['result']) for t in group],out/'validation'/variant)
        if variant=='reference':continue
        origins={t.origin for t in group}
        refs=[t for t in forecasts if t.variant=='reference' and t.origin in origins]
        if len(refs)!=len(origins) or not all(records[t.id]['status']=='completed' for t in refs):continue
        target=out/'paired_validation'/variant
        merge_folds([Path(records[t.id]['result']) for t in refs],target/'reference')
        merge_folds([Path(records[t.id]['result']) for t in group],target/variant)
        predictive={name:{ch:target/name for ch in bx.UCCLE_SERIES} for name in ('reference',variant)}
        bx.SensitivityReport(predictive_runs=predictive,baseline='reference',block_frequency='seasonal').save(
            target/'comparison',figures=figures)
        matched.append(dict(variant=variant,origins=sorted(origins),numerically_passed=all(
            records[t.id].get('numerical_status')=='passed_numerical_checks' for t in refs+group)))
    bx.save_config(dict(comparisons=matched,note='Scores are paired only over identical forecast origins and cases.'),out/'paired_validation.json')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--tier',choices=('screen','paper'),default='screen')
    p.add_argument('--batch',choices=BATCHES,default='posterior');p.add_argument('--figures',action='store_true')
    p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    print(collect(a.root,a.tier,batch=a.batch,figures=a.figures,require_complete=a.require_complete))

if __name__=='__main__':main()
