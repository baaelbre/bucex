"""Collect mixed shared/separate fits and compare identical predictive cases."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import pandas as pd
import bucex as bx
from research.seasonal.jobs import ROOT,BATCHES,tasks,task_config,fingerprint,result_directory
from research.seasonal.job_plan import BASELINES


def merge_folds(paths,destination,channel=None):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    folds=pd.concat([pd.read_csv(Path(p)/'folds.csv') for p in paths],ignore_index=True)
    if folds.origin.duplicated().any():raise ValueError('duplicate validation origins: '+str(destination))
    for name in ('scores','held_out_pit','coverage_by_case','predictions','event_counts','threshold_cases'):
        inputs=[Path(p)/(name+'.csv') for p in paths]
        if all(p.exists() for p in inputs):
            frames=[]
            for path in inputs:
                f=pd.read_csv(path)
                if channel is not None and 'channel' in f:f=f[f.channel==channel]
                frames.append(f)
            pd.concat(frames,ignore_index=True).to_csv(destination/(name+'.csv'),index=False)
    folds.to_csv(destination/'folds.csv',index=False)
    for path in paths:
        for pattern in ('convergence_*.json','independent_shrinkage_*.csv','shared_shrinkage_*.csv'):
            for file in Path(path).glob(pattern):shutil.copy2(file,destination/file.name)


def collect(root=ROOT,tier='screen',*,batch='experiments',figures=False,require_complete=False):
    root=Path(root);expected=tasks(batch,tier=tier);records={};rows=[];mismatch=[]
    for task in expected:
        directory=root/tier/task.id;manifest=directory/'task.json'
        record=bx.load_config(manifest) if manifest.exists() else dict(status='missing')
        if manifest.exists():
            identity=dict(bucex_version=bx.__version__,**fingerprint(task_config(task,tier)))
            if any(record.get(k)!=v for k,v in identity.items()):
                mismatch.append(task.id);record=dict(record,status='provenance_mismatch')
        local=result_directory(directory,task)
        if record['status']=='completed':
            record=dict(record,result=str(local.resolve()))
            if not local.is_dir():record['status']='missing_report'
        records[task.id]=record
        rows.append(dict(task_id=task.id,kind=task.kind,variant=task.variant,scope=task.scope,
            channel=task.channel,series=','.join(task.series),origin=task.origin,status=record['status'],
            numerical_status=record.get('numerical_status'),result=record.get('result')))
    incomplete=[t.id for t in expected if records[t.id]['status']!='completed']
    flagged=[t.id for t in expected if records[t.id]['status']=='completed' and
             (records[t.id].get('numerical_status')!='passed_numerical_checks' or records[t.id].get('final_check')=='failed')]
    out=root/tier/'collected'/batch;out.mkdir(parents=True,exist_ok=True)
    report=dict(version=bx.__version__,tier=tier,batch=batch,expected=len(expected),
        completed=len(expected)-len(incomplete),missing_or_failed=incomplete,numerical_flags=flagged,
        provenance_mismatch=mismatch,status='passed' if not incomplete and not flagged and tier=='paper' else 'incomplete_or_screening')
    bx.save_config(report,out/'status.json');pd.DataFrame(rows).to_csv(out/'tasks.csv',index=False)
    if require_complete and (incomplete or flagged or tier!='paper'):
        raise RuntimeError(f'{len(incomplete)} incomplete, {len(flagged)} flagged; see {out}/status.json')
    completed=[t for t in expected if t.kind=='posterior' and records[t.id]['status']=='completed']
    for ch in dict.fromkeys(ch for t in completed for ch in t.series):
        group=[t for t in completed if ch in t.series]
        posterior={t.variant:{ch:Path(records[t.id]['result'])} for t in group}
        def save(names,baseline,destination,plot):
            available={v:posterior[v] for v in names if v in posterior}
            if baseline not in available or not available:return
            bx.SensitivityReport(posterior_runs=available,baseline=baseline,block_frequency='seasonal').save(destination,figures=plot)
        base=out/'posterior'/ch
        save(list(posterior),'reference',base,False)
        save(BASELINES,'independent_reference',base/'pooling',figures)
        for scope,baseline in [('shared','reference'),('independent','independent_reference'),('fixed','fixed_reference')]:
            names=[t.variant for t in group if t.scope==scope and t.study=='structural']
            save(names,baseline,base/'by_scope'/scope,figures)
        for family in ('width','adequacy','influence'):
            names=['reference']+[t.variant for t in group if t.study==family]
            save(names,'reference',base/family,figures and len(names)<=10)
    forecasts=[t for t in expected if t.kind=='forecast'];matched=[]
    keys=sorted({(t.design,t.variant,ch) for t in forecasts for ch in t.series})
    for design,variant,ch in keys:
        group=[t for t in forecasts if t.variant==variant and t.design==design and ch in t.series]
        if not all(records[t.id]['status']=='completed' for t in group):continue
        base=out/design/ch
        merge_folds([records[t.id]['result'] for t in group],base/'validation'/variant,ch)
        if variant=='reference':continue
        origins={t.origin for t in group}
        refs=[t for t in forecasts if t.variant=='reference' and t.origin in origins and t.design==design and ch in t.series]
        if len(refs)!=len(origins) or not all(records[t.id]['status']=='completed' for t in refs):continue
        target=base/'paired_validation'/variant
        merge_folds([records[t.id]['result'] for t in refs],target/'reference',ch)
        merge_folds([records[t.id]['result'] for t in group],target/variant,ch)
        bx.SensitivityReport(predictive_runs={v:{ch:target/v} for v in ('reference',variant)},
            baseline='reference',block_frequency='seasonal').save(target/'comparison',figures=figures)
        matched.append(dict(variant=variant,channel=ch,design=design,origins=sorted(origins),
            numerically_passed=all(records[t.id].get('numerical_status')=='passed_numerical_checks' for t in refs+group)))
    bx.save_config(dict(comparisons=matched,note='Matching forecast cases only; convergence flags remain visible. Hierarchical residual independence is a working assumption.'),out/'paired_validation.json')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--tier',choices=('screen','paper'),default='screen')
    p.add_argument('--batch',choices=BATCHES,default='experiments');p.add_argument('--figures',action='store_true')
    p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    print(collect(a.root,a.tier,batch=a.batch,figures=a.figures,require_complete=a.require_complete))

if __name__=='__main__':main()
