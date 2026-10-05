"""Optional constant linear trend benchmark with its own explicit JSON specification."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import bucex as bx
from .configuration import CONFIG, load_config
from .data import load_summaries
from .run import run
from .figures import style, panels, savefig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=CONFIG/'linear_benchmark.json')
    p.add_argument('--profile',choices=('smoke','screen','paper'),default='screen')
    p.add_argument('--output',type=Path,default=Path('results/linear_benchmark'))
    p.add_argument('--workers',type=int)
    args=p.parse_args();config=load_config(args.config)
    config['profiles'][args.profile]['max_blocks']=None
    fit,root=run(config,profile=args.profile,output=args.output/args.profile,workers=args.workers,figures=False)
    data=load_summaries(config['data']['source'])[list(fit.channels)]
    heldout=data.loc[data.index > next(iter(fit.channels.values())).index[-1]]
    prediction=fit.predict(len(heldout),draws=config['profiles'][args.profile]['predictive_draws'],seed=config['seed']+43)
    bx.coverage(prediction,heldout).to_csv(root/'coverage.csv',index=False)
    summer=prediction.index.month==8
    risk=prediction.risk(35,channel='TXx')[:,summer]
    observed=int(np.sum(heldout.loc[summer,'TXx'] > 35))
    counts=np.sum(prediction.y['TXx'][:,summer] > 35,axis=1)
    pd.DataFrame([dict(summers=int(summer.sum()),threshold=35,observed=observed,
        expected=float(risk.sum(1).mean()),probability_at_least_observed=float(np.mean(counts>=observed)))]).to_csv(root/'summer_threshold_count.csv',index=False)
    style();fig,axes=panels(list(fit.channels))
    for name,ax in zip(fit.channels,axes):
        bx.plot(prediction,channel=name,type='forecast',observed=heldout[name].to_numpy(),ax=ax)
    savefig(fig,root/'figures','linear_forecasts',config)


if __name__=='__main__': main()
