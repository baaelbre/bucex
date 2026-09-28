"""One independently scheduled fit scored on common held-out seasonal targets."""
from pathlib import Path
import numpy as np
import pandas as pd
import bucex as bx
from research.monthly.models import joint_model, fit_options
from research.monthly.report import scientific_targets, convergence_parameters
from research.monthly.validate import validation_splits


def run(config, task, directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    data=bx.load_uccle_multiseries(**config['data'])
    seasonal=bx.load_uccle_multiseries(**dict(config['data'],frequency='seasonal'))
    train,test=validation_splits(data,config['validation'])[0]
    training=data.iloc[list(train)];held=data.iloc[list(test)]
    model,prior=joint_model(training,config)
    fit=bx.fit(training,model,priors=prior,**fit_options(config,family=model.family))
    parameters=fit.diagnostics()['parameters'];parameters.to_csv(directory/'mcmc.csv')
    targets=fit.contrast_diagnostics(scientific_targets(fit,config),credible_interval=.95)
    targets.to_csv(directory/'scientific_targets.csv')
    check=bx.convergence_assessment({'parameters':convergence_parameters(parameters,fit.n_chains,fit),
        'scientific_targets':targets},**config['diagnostic_thresholds'])
    bx.save_config(check,directory/'convergence.json');bx.save_config(config,directory/'config.json')
    if config.get('save_fits',True):fit.save(directory/'fit.bucex')
    bx.save_shared_shrinkage_report(fit,directory,level=.95,figures=config.get('figures',False),
        horizon=config['prior_calibration']['horizon'],rate_multiplier=config['prior_calibration']['slope_time_unit'])
    forecast=fit.forecast(len(test),dates=held.index,draws=config['validation']['draws'],seed=config['seed'])
    scores=[];calibration=[]
    for channel in data:
        aggregated=forecast.aggregate(frequency='season',channel=channel)
        actual=aggregated.aggregate_values(held[channel].to_numpy())
        expected=seasonal.loc[pd.DatetimeIndex(aggregated.periods.start),channel].to_numpy()
        if len(actual)!=task.horizon or not np.allclose(actual,expected,rtol=0,atol=1e-10):
            raise ValueError('Monthly/seasonal targets differ; no valid block comparison.')
        score,cal=bx.score_seasonal_forecast(forecast,seasonal[channel],channel=channel,
            threshold=config['risks'][channel],level=.95)
        scores.append(score.assign(model=task.frequency,channel=channel,origin=task.origin))
        calibration.append(cal.assign(model=task.frequency,channel=channel,origin=task.origin))
    pd.concat(scores).to_csv(directory/'scores.csv',index=False)
    pd.concat(calibration).to_csv(directory/'calibration.csv',index=False)
    step=1 if task.frequency=='monthly' else 3
    bx.save_config(dict(version=bx.__version__,model=task.frequency,cutoff=task.origin,
        status=check['status'],training_blocks=len(training),
        training_start=str(training.index[0].date()),
        training_end=str((training.index[-1]+pd.offsets.MonthEnd(step)).date()),
        test_start=str(held.index[0].date()),
        test_end=str((held.index[-1]+pd.offsets.MonthEnd(step)).date()),
        daily_sha256=data.attrs.get('daily_sha256')),directory/'complete.json')
    return check['status']
