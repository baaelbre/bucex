"""Staged, matched rolling-origin comparisons of margins and dependence."""
import argparse
from pathlib import Path
import pandas as pd
import bucex as bx
from .experiment import configured_variant
from .report import new_run
from .validate import validate


def run(config, *, stage='margins', candidates=None):
    directory=new_run(config['output'],'model_comparison_'+stage)
    bx.save_config(config,directory/'config.json')
    selected=config['comparison'][stage]
    if candidates:
        unknown=set(candidates)-{v['name'] for v in selected}
        if unknown:
            raise ValueError(f'Unknown candidates: {sorted(unknown)}')
        selected=[v for v in selected if v['name'] in candidates]
    runs=[]
    for variant in selected:
        current=configured_variant(config,variant)
        current['analysis']=variant.get('analysis','independent')
        if 'copula' in variant:
            current['copula']=variant['copula']
        current['output']=str(directory/variant['name'])
        print(variant['name'],flush=True)
        result=validate(current)
        runs.append(dict(candidate=variant['name'],directory=str(result)))
        pd.DataFrame(runs).to_csv(directory/'runs.csv',index=False)
    return directory


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('research/serra/config/model_comparison_smoke.json'))
    parser.add_argument('--stage',choices=['margins','dependence'],default='margins')
    parser.add_argument('--candidates',nargs='+')
    args=parser.parse_args()
    print(run(bx.load_config(args.config),stage=args.stage,candidates=args.candidates))
