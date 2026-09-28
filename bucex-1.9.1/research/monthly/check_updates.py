"""Check a saved fit against later bundled months, without refitting.

This is one fixed-origin multistep forecast, not rolling one-step validation.
Only months after the actual saved training cutoff enter the check.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import bucex as bx
from research.monthly.report import new_run
from research.monthly.validate import calibration


def check_updates(fit, directory, *, draws=1000, seed=56102, level=.95):
    import matplotlib.pyplot as plt
    names = list(fit.channel_names) if fit.is_multiseries_model else [fit.series_name]
    if any(name not in bx.UCCLE_INFO for name in names):
        raise ValueError("This example requires a named Uccle fit. General data can use Forecast.score and Forecast.pit directly.")
    data = bx.load_uccle_multiseries(series=names)
    future = data.loc[data.index > pd.Timestamp(fit.time[-1])]
    if future.empty:
        raise ValueError("No bundled observations occur after this fit's training cutoff.")
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    forecast = fit.forecast(len(future), draws=draws, seed=seed, dates=future.index)
    bx.save_config(dict(fitted_end=str(fit.time[-1]),held_out_start=str(future.index[0]),
        held_out_end=str(future.index[-1]),n_held_out=len(future),seed=seed,draws=draws,
        method="Fixed-origin multistep prediction; future observations do not update this posterior.",
        caveat="One path of correlated forecast errors; not an independent uniformity test or rolling-origin validation."),path/'held_out_window.json')
    for name in names:
        channel = name if fit.is_multiseries_model else None
        observed = future[name].to_numpy()
        kwargs = dict(channel=channel)
        scores=forecast.score(observed, aggregate=False, **kwargs)
        scores.assign(time=future.index[scores.time_index.to_numpy(dtype=int)]).to_csv(path/f'{name}_scores.csv',index=False)
        calibration(forecast, observed, **kwargs).to_csv(path/f'{name}_calibration.csv',index=False)
        forecast.summary(level=level, **kwargs).assign(observed=observed,
            pit=forecast.pit(observed, **kwargs)).to_csv(path/f'{name}_held_out.csv',index=False)
        figure,_ = bx.plot_predictive_diagnostics(forecast, observed, **kwargs)
        figure.savefig(path/f'{name}_held_out_pit_qq.png',dpi=150,bbox_inches='tight');plt.close(figure)
        axis = forecast.plot(level=level, observed=observed, **kwargs)
        axis.figure.savefig(path/f'{name}_held_out_forecast.png',dpi=150,bbox_inches='tight');plt.close(axis.figure)
        figure,_ = bx.plot_forecast_months(forecast, observed=observed, level=level, **kwargs)
        figure.savefig(path/f'{name}_held_out_by_month.png',dpi=150,bbox_inches='tight');plt.close(figure)
        for frequency in ('year','season'):
            aggregate=forecast.aggregate(frequency=frequency, **kwargs)
            actual=aggregate.aggregate_values(observed)
            aggregate.summary(level).assign(observed=actual).to_csv(path/f'{name}_held_out_{frequency}.csv',index=False)
            if not aggregate.n_periods:
                continue
            windows=list(dict.fromkeys(aggregate.periods.window))
            figure,axes=plt.subplots(len(windows),1,squeeze=False,figsize=(9,3*len(windows)))
            for window,axis in zip(windows,axes[:,0]):
                aggregate.plot(level=level,window=window,ax=axis)
                selected=aggregate.periods.window.to_numpy()==window
                axis.scatter(aggregate.periods.loc[selected,'year'],actual[selected],color='black',s=18,label='observed')
                axis.set_title(window,fontsize=11);axis.legend(fontsize=8)
            figure.tight_layout();figure.savefig(path/f'{name}_held_out_{frequency}.png',dpi=150);plt.close(figure)
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('results/serra_184_monthly'))
    parser.add_argument('--draws',type=int,default=1000)
    args=parser.parse_args()
    directory=new_run(args.output,'updated_data_check')
    print(check_updates(bx.load_fit(args.fit),directory,draws=args.draws))


if __name__=='__main__':
    main()
