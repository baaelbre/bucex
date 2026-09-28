"""Exercise the real bounded process loop without production-length inference."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from research.seasonal import local,jobs


def test_local_runner_bounds_concurrency_and_records_failure(tmp_path,monkeypatch):
    selected=jobs.tasks('comparison',tier='screen')[:4]
    monkeypatch.setattr(local,'tasks',lambda *a,**k:selected)
    monkeypatch.setattr(local,'verify',lambda *a,**k:dict(status='test_preflight'))
    real_popen=subprocess.Popen
    code='''import json,os,sys,time
from pathlib import Path
start=time.time()
time.sleep(.3)
Path(sys.argv[1]).write_text(json.dumps({'start':start,'end':time.time(),'threads':os.environ['OPENBLAS_NUM_THREADS']}))
sys.exit(int(sys.argv[2]))
'''
    def launch(args,**kwargs):
        task=args[args.index('--task')+1]
        return real_popen([sys.executable,'-c',code,str(tmp_path/(task+'.json')),
                          '2' if task==selected[1].id else '0'],**kwargs)
    monkeypatch.setattr(local.subprocess,'Popen',launch)
    assert local.run('screen','reference',tmp_path/'results',2)==1
    intervals=[json.loads((tmp_path/(t.id+'.json')).read_text()) for t in selected]
    points=sorted([(r['start'],1) for r in intervals]+[(r['end'],-1) for r in intervals])
    active=np.cumsum([x[1] for x in points])
    assert active.max()==2 and active.min()>=0
    assert all(r['threads']=='1' for r in intervals)
    records=list((tmp_path/'results/screen').glob('local_run_*.json'))
    summary=json.loads(records[0].read_text())
    assert len(summary['outcomes'])==len(selected) and not summary['not_started']
    assert sum(r['exit_code']!=0 for r in summary['outcomes'])==1
    assert len(list((tmp_path/'results/screen/logs').glob('*.log')))==len(selected)
