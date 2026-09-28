"""Calibration and prior/posterior checks for fixed Normal shrinkage priors."""
import numpy as np
import pandas as pd
import bucex as bx
from scipy.special import ndtri


def calibration(prior, *, period, horizon, rate_multiplier):
    gains=bx.innovation_response_gains(horizon,period=period)
    fields={'level':'s_level','slope':'s_trend','seasonal':'s_season','initial_slope':'beta0'}
    rows=[]
    for component,field in fields.items():
        normal=getattr(prior,field)
        if not isinstance(normal,bx.NormalPrior):raise ValueError('Fixed calibration requires Normal coefficients.')
        gain=horizon if component=='initial_slope' else gains[component]
        rows.append(dict(component=component,prior_mean=normal.mean,prior_sd=normal.sd,
            prior_median_absolute=normal.sd*ndtri(.75),horizon_updates=horizon,
            response_gain=gain,displacement_sd=normal.sd*gain,
            initial_rate_sd_C_per_decade=normal.sd*rate_multiplier if component=='initial_slope' else np.nan,
            sampled_hyperparameter=False))
    return pd.DataFrame(rows)


def save(fit,directory,config,*,level=.95):
    """Compare initial slopes with their fixed priors; no hyperparameter posteriors."""
    period=config['model']['period'];rate=10*config['model']['steps_per_year']
    horizon=config.get('prior_calibration',{}).get('horizon',120)
    rows=[]
    for name in fit.channel_names:
        frame=calibration(fit.priors.channels[name],period=period,horizon=horizon,rate_multiplier=rate)
        frame['channel']=name
        frame['active']=[('initial.channel.'+name+'.slope' if c=='initial_slope' else 'sd.channel.'+name+'.'+c)
                         in fit.parameter_draws for c in frame.component]
        rows.append(frame)
    pd.concat(rows,ignore_index=True).to_csv(directory/'fixed_prior_settings.csv',index=False)
    initial=bx.compare_initial_slope_priors(fit,size=config.get('prior_draws',20000),
        seed=config['seed'],rate_multiplier=rate,level=level)
    initial.to_csv(directory/'initial_slope_prior_posterior.csv',index=False)
    if config.get('figures',True):
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(6,3.2),layout='constrained')
        for i,name in enumerate(initial.channel.drop_duplicates()):
            for label,offset,color in [('prior',-.12,'.6'),('posterior',.12,'C0')]:
                row=initial[(initial.channel==name)&(initial.distribution==label)].iloc[0]
                ax.errorbar(row['median'],i+offset,xerr=[[row['median']-row['lower']],[row['upper']-row['median']]],
                    fmt='o',color=color,label=label if i==0 else None)
        names=list(initial.channel.drop_duplicates())
        ax.axvline(0,color='.5',lw=.8);ax.set(yticks=range(len(names)),yticklabels=names,xlabel='initial slope / °C per decade');ax.legend()
        fig.savefig(directory/'initial_slope_prior_posterior.png',dpi=config.get('figure_dpi',180),bbox_inches='tight');plt.close(fig)
