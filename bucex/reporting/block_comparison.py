"""Compact comparison of common seasonal forecast targets."""
from pathlib import Path
import numpy as np
import pandas as pd
from ..config import save_config
from ..plotting.style import publication_style


def save_block_comparison(directory, *, figures=True):
    """Summarize paired scores; preserve origin and forecast horizon.

    No automatic winner or iid standard error is attached to overlapping
    multi-step forecasts. With few origins these are descriptive comparisons.
    """
    directory=Path(directory)
    paths=sorted(directory.glob('*/*/complete.json'))
    if not paths:
        raise ValueError('No completed forecast folds to compare.')
    scores=pd.concat([pd.read_csv(p.parent/'scores.csv') for p in paths],ignore_index=True)
    calibration=pd.concat([pd.read_csv(p.parent/'calibration.csv') for p in paths],ignore_index=True)
    output=directory/'comparison';output.mkdir(exist_ok=True)
    scores.to_csv(output/'scores_by_case.csv',index=False)
    calibration.to_csv(output/'calibration_by_case.csv',index=False)
    scores['lead_group']=np.where(scores.horizon_seasons<=4,'first_year','after_first_year')
    for name,keys in [('score_summary',['model','channel','score','setting']),
                      ('scores_by_season',['model','channel','season','score','setting']),
                      ('scores_by_horizon',['model','channel','lead_group','score','setting']),
                      ('scores_by_origin',['model','channel','origin','score','setting'])]:
        scores.groupby(keys,dropna=False).value.agg(mean='mean',n='size').to_csv(output/(name+'.csv'))
    calibration.groupby(['model','channel','window']).agg(pit_mean=('pit','mean'),
        coverage=('covered','mean'),mean_width=('width','mean'),n=('pit','size')).to_csv(output/'calibration_summary.csv')
    keys=['origin','channel','time','score','setting']
    a=scores[scores.model.eq('monthly')];b=scores[scores.model.eq('seasonal')]
    paired=a.merge(b,on=keys,suffixes=('_monthly','_seasonal'),validate='one_to_one')
    if len(paired):
        if not np.allclose(paired.observed_monthly,paired.observed_seasonal):
            raise ValueError('Cannot compare predictions for different observed targets.')
        paired['seasonal_minus_monthly']=paired.value_seasonal-paired.value_monthly
        paired.to_csv(output/'paired_score_differences.csv',index=False)
        paired.groupby(['channel','score','setting'],dropna=False).seasonal_minus_monthly.agg(
            mean='mean',n='size').to_csv(output/'paired_summary.csv')
    notes=dict(target='Same observed seasonal daily-weighted means, maxima or minima',
        difference='seasonal minus monthly; negative is better for seasonal',
        completed_model_origins=len(paths),n_paired_cases=len(paired),
        caution='Descriptive short-run assessment. Check each fold convergence.json; wider or narrower intervals alone do not choose a model.',
        density='Marginal seasonal predictive density, not raw monthly-vs-seasonal fitting likelihoods or a joint six-dimensional score',
        excluded_claims='No asymptotic guarantee, no formal significance from a few origins, no r-largest or daily-cluster likelihood.')
    save_config(notes,output/'comparison_notes.json')
    if figures:
        import matplotlib.pyplot as plt
        with publication_style(style='manuscript'):
            table=scores[scores.score.eq('crps')].groupby(['channel','model']).value.mean().unstack()
            ax=table.plot.bar(figsize=(9,4),rot=0)
            ax.set(ylabel='seasonal forecast CRPS / °C',xlabel='summary')
            ax.figure.tight_layout();ax.figure.savefig(output/'seasonal_crps.png',dpi=180);plt.close(ax.figure)
            names=list(calibration.channel.drop_duplicates())
            figure,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
            for name,ax in zip(names,axes.flat):
                for model,group in calibration[calibration.channel.eq(name)].groupby('model'):
                    pit=np.sort(group.pit.to_numpy());ax.plot(pit,np.arange(1,len(pit)+1)/len(pit),label=model)
                ax.plot([0,1],[0,1],color='black',ls='--',lw=.8)
                ax.set(xlabel='held-out seasonal PIT',ylabel='empirical CDF',title=name)
            axes.flat[0].legend(fontsize=8)
            for ax in list(axes.flat)[len(names):]:ax.set_axis_off()
            figure.savefig(output/'seasonal_forecast_pit.png',dpi=180);plt.close(figure)
    return output


__all__=['save_block_comparison']
