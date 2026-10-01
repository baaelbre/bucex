"""Compare completed main/monthly/private/dispersion analyses without refitting."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import bucex as bx
from .configuration import load_config
from .figures import style,panels,savefig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',nargs='+',type=Path)
    p.add_argument('--output',type=Path,default=Path('results/comparisons'))
    args=p.parse_args()
    fits={run.parent.name+'_'+run.name:bx.load(run/'fit.bucex') for run in args.runs}
    if len(fits)!=len(args.runs):raise ValueError('Run labels must be unique.')
    names=list(next(iter(fits.values())).channels)
    if any(list(f.channels)!=names for f in fits.values()):raise ValueError('Compared fits need the same response names.')
    style();args.output.mkdir(parents=True,exist_ok=True)
    config=load_config()
    import matplotlib.pyplot as plt
    rows=[]
    for kind in ['level','slope','normal_qq','cycle']:
        fig,axes=panels(names)
        for name,ax in zip(names,axes):
            for i,(label,f) in enumerate(fits.items()):
                if kind=='cycle' and f[name].model.period!=4:continue
                bx.plot(f,channel=name,type=kind,ax=ax,color=plt.get_cmap('tab10')(i),label=label)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=7)
        savefig(fig,args.output/'figures','comparison_'+kind,config)
    for label,f in fits.items():
        table=f.summary().reset_index();table['analysis']=label;rows.append(table)
    pd.concat(rows,ignore_index=True).to_csv(args.output/'parameter_comparison.csv',index=False)


if __name__=='__main__':main()
