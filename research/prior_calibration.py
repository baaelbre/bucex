"""Joint prior implications for same-season changes at 10 and 30 years.

The public prior sampler draws shared scales once per replication and all
response amplitudes conditional on them. Exact Gaussian horizon covariances
then avoid simulating (and retaining) every intervening state.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import bucex as bx
from .configuration import CONFIG, load_config, build_model
from .figures import style, panels, savefig


def calibrate(config, *, profile='screen', output=Path('results/prior_calibration'), seed=19830):
    root=Path(output)/profile;root.mkdir(parents=True,exist_ok=True)
    draws=config['profiles'][profile]['prior_draws']
    steps=config['model']['steps_per_year']
    horizons=np.array([10,30])*steps
    years=horizons/steps
    parameters=bx.prior_samples(build_model(config),draws=draws,seed=seed)
    rng=np.random.default_rng(np.random.SeedSequence(seed,spawn_key=(1,)))
    level_cov=np.minimum.outer(horizons,horizons).astype(float)
    slope_cov=np.array([[sum((h-r)*(k-r) for r in range(1,min(h,k))) for k in horizons] for h in horizons],float)
    seasonal_cov=2*np.minimum.outer(years,years)
    names=config['data']['series']
    rows=[];plot_values={}
    for name in names:
        changes={}
        for key,covariance in [('level',level_cov),('slope',slope_cov),('seasonal',seasonal_cov)]:
            changes[key]=np.sqrt(parameters[name+'.variance.'+key])[:,None]*rng.multivariate_normal(np.zeros(2),covariance,size=draws)
        changes['initial_rate']=parameters[name+'.initial_slope'][:,None]*horizons
        changes['combined']=sum(changes.values())
        plot_values[name]=changes
        for key,values in changes.items():
            for j,year in enumerate(years):
                low,high=np.quantile(values[:,j],[.025,.975])
                rows.append(dict(channel=name,years=int(year),contribution=key,mean=float(values[:,j].mean()),
                    sd=float(values[:,j].std()),lower_95=float(low),upper_95=float(high),draws=draws))
    table=pd.DataFrame(rows);table.to_csv(root/'prior_horizons.csv',index=False)
    (root/'prior_run.json').write_text(json.dumps(dict(seed=seed,draws=draws,method='joint amplitudes and exact horizon covariances',
                                                   config=config),indent=2)+'\n')
    style()
    # Marginal calibration is exchangeable across responses in this paper.
    # The first channel illustrates the common prior; all six are exported.
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
    keys=['level','slope','seasonal','initial_rate','combined']
    labels=['Level innovations','Integrated slope','Seasonal change','Initial rate','Combined change']
    for j,(year,ax) in enumerate(zip(years,axes)):
        selected=table[(table.channel==names[0])&(table.years==int(year))].set_index('contribution').loc[keys]
        y=np.arange(len(keys))
        ax.hlines(y,selected.lower_95,selected.upper_95,color='#24658a')
        ax.plot(selected['mean'],y,'o',color='#24658a',ms=4)
        ax.axvline(0,color='.6',ls=':',lw=.8)
        ax.set_yticks(y,labels);ax.invert_yaxis()
        ax.set_xlabel(f'Same-season change over {int(year)} years (°C)')
    savefig(fig,root/'figures','prior_calibration',config)
    return root


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=CONFIG/'main.json')
    p.add_argument('--profile',choices=('smoke','screen','paper'),default='screen')
    p.add_argument('--output',type=Path,default=Path('results/prior_calibration'))
    p.add_argument('--seed',type=int,default=19830)
    args=p.parse_args()
    print(calibrate(load_config(args.config),profile=args.profile,output=args.output,seed=args.seed))


if __name__=='__main__': main()
