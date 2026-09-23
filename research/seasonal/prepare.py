"""Audit complete seasons and export six seasonal summaries and exploratory plots."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import bucex as bx


def prepare(config,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    data=bx.load_uccle_multiseries(**config['data'])
    data.to_csv(output/'seasonal_summaries.csv')
    bx.save_config(data.attrs,output/'quality_report.json')
    pd.DataFrame(data.attrs['block_audit']).to_csv(output/'block_audit.csv',index=False)
    phase=data.index.month.map(bx.SEASON_NAMES)
    cycle=data.groupby(phase).agg(['mean','std','min','max']).reindex(['DJF','MAM','JJA','SON'])
    cycle.to_csv(output/'seasonal_cycle.csv')
    import matplotlib.pyplot as plt
    with bx.publication_style(style=config.get('figure_style','manuscript')):
        figure,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
        for name,ax in zip(data,axes.flat):
            for season in ('DJF','MAM','JJA','SON'):
                selected=phase==season
                ax.plot(data.index[selected],data.loc[selected,name],lw=.5,alpha=.7,label=season)
            ax.set(ylabel=name+' / °C',xlabel='season start')
        axes.flat[0].legend(fontsize=8,ncol=2)
        bx.save_figure(figure,output/'seasonal_records',formats=('png',),dpi=180,close=True)
        figure,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
        for name,ax in zip(data,axes.flat):
            values=[data.loc[phase==s,name] for s in ('DJF','MAM','JJA','SON')]
            ax.boxplot(values)
            ax.set(xticks=[1,2,3,4],xticklabels=['DJF','MAM','JJA','SON'],ylabel=name+' / °C')
        bx.save_figure(figure,output/'seasonal_distributions',formats=('png',),dpi=180,close=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='research/seasonal/config/main.json')
    parser.add_argument('--output',default='results/serra_184_seasonal_data')
    args=parser.parse_args()
    print(prepare(bx.load_config(args.config),args.output))
