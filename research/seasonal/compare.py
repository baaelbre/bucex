"""Fit monthly/seasonal models at identical cutoffs and score common seasonal targets."""
import argparse
from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import bucex as bx
from research.monthly.models import joint_model,fit_options
from research.monthly.report import new_run,scientific_targets,convergence_parameters


def study(config):
    seasonal=deepcopy(config)
    monthly=bx.load_config(config['comparison']['monthly_config'])
    monthly['mcmc']=deepcopy(config['mcmc'])
    monthly['data']['series']=list(config['data']['series'])
    monthly['copula']=deepcopy(config['copula'])
    monthly['analysis']=config['analysis']
    analyses={}
    for name,local,step in [('monthly',monthly,1),('seasonal',seasonal,3)]:
        data=bx.load_uccle_multiseries(**local['data'])
        horizon=config['comparison']['horizon_seasons']*(3 if step==1 else 1)
        splits=list(bx.calendar_origin_splits(data.index,config['comparison']['training_ends'],
            horizon=horizon,block_frequency=name))
        analyses[name]=(local,data,splits)
    # Both frequencies must use identical DAILY start/end dates in every fold.
    for i,origin in enumerate(config['comparison']['training_ends']):
        boundaries=[]
        for name,(_,data,splits) in analyses.items():
            train,test=splits[i];step=1 if name=='monthly' else 3
            boundaries.append((data.index[0],data.index[train.stop-1]+pd.offsets.MonthEnd(step),
                               data.index[test.start],data.index[test.stop-1]+pd.offsets.MonthEnd(step)))
        if boundaries[0]!=boundaries[1]:
            raise ValueError(f'Monthly and seasonal training/forecast windows differ at {origin}.')
    return analyses


def run(config,*,directory=None):
    analyses=study(config)
    reference=analyses['monthly'][0]
    fingerprint={name:hashlib.sha256(data.to_csv().encode()).hexdigest()
                 for name,(_,data,_) in analyses.items()}
    if directory is None:
        directory=new_run(config['output'],'block_comparison')
        bx.save_config(config,directory/'config.json')
        bx.save_config(reference,directory/'monthly_reference.json')
        bx.save_config(dict(version=bx.__version__,data=fingerprint),directory/'provenance.json')
    else:
        directory=Path(directory)
        if bx.load_config(directory/'config.json')!=config:
            raise ValueError('Resume must retain the exact saved configuration.')
        if (bx.load_config(directory/'monthly_reference.json')!=reference or
            bx.load_config(directory/'provenance.json')!=dict(version=bx.__version__,data=fingerprint)):
            raise ValueError('Monthly reference, data or package version changed; start a fresh comparison.')
    seasonal_data=analyses['seasonal'][1]
    for name,(local,data,splits) in analyses.items():
        for cutoff,(train,test) in zip(config['comparison']['training_ends'],splits):
            target=directory/name/cutoff;target.mkdir(parents=True,exist_ok=True)
            if (target/'complete.json').exists():
                print(f'{name} {cutoff}: completed; keeping results.',flush=True);continue
            print(f'{name} {cutoff}: {train.stop} training blocks, {len(test)} forecast blocks.',flush=True)
            training=data.iloc[list(train)]
            model,prior=joint_model(training,local)
            fit=bx.fit(training,model,priors=prior,**fit_options(local,family=model.family))
            diagnostics=fit.diagnostics()['parameters']
            diagnostics.to_csv(target/'mcmc.csv')
            targets=fit.contrast_diagnostics(scientific_targets(fit,{'model':local['model']}))
            targets.to_csv(target/'targets.csv')
            check=bx.convergence_assessment({'parameters':convergence_parameters(diagnostics,fit.n_chains),
                'scientific_targets':targets},**local.get('diagnostic_thresholds',{}))
            bx.save_config(check,target/'convergence.json')
            bx.save_config(local,target/'config.json')
            if config.get('save_fits',True):fit.save(target/'fit.bucex')
            if getattr(fit.priors,'shrinkage',None) is not None:
                bx.save_shared_shrinkage_report(fit,target,figures=config.get('figures',True),
                    horizon=local['prior_calibration']['horizon'],
                    rate_multiplier=local['prior_calibration']['slope_time_unit'],response_unit='degC')
            forecast=fit.forecast(len(test),dates=data.index[list(test)],
                draws=config['comparison'].get('draws',2000),seed=config['seed'])
            scores=[];calibration=[]
            for channel in data:
                aggregate=forecast.aggregate(frequency='season',channel=channel)
                actual=aggregate.aggregate_values(data[channel].iloc[list(test)].to_numpy())
                expected=seasonal_data.loc[pd.DatetimeIndex(aggregate.periods.start),channel].to_numpy()
                if not np.allclose(actual,expected,rtol=0,atol=1e-10):
                    raise ValueError('Daily seasonal and monthly-derived targets disagree; check data provenance.')
                score,cal=bx.score_seasonal_forecast(forecast,seasonal_data[channel],channel=channel,
                    threshold=config['risks'][channel],level=config['credible_interval'])
                scores.append(score.assign(model=name,channel=channel,origin=cutoff))
                calibration.append(cal.assign(model=name,channel=channel,origin=cutoff))
            constraints=[p for p in bx.UCCLE_ORDER_CONSTRAINTS if set(p)<=set(data)]
            if constraints:
                forecast.ordering_diagnostics(constraints).summary.to_csv(target/'ordering.csv',index=False)
            pd.concat(scores).to_csv(target/'scores.csv',index=False)
            pd.concat(calibration).to_csv(target/'calibration.csv',index=False)
            bx.save_config(dict(version=bx.__version__,model=name,cutoff=cutoff,
                status=check['status'],training_blocks=len(training)),target/'complete.json')
            bx.save_block_comparison(directory,figures=config.get('figures',True))
            del fit,forecast
    return bx.save_block_comparison(directory,figures=config.get('figures',True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='research/seasonal/config/compare.json')
    parser.add_argument('--run',type=Path,help='Resume completed folds using this run\'s saved config.')
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args()
    config=bx.load_config(args.run/'config.json' if args.run else args.config)
    if args.plan:
        for name,(local,data,splits) in study(config).items():
            print(name,[(train.stop,len(test)) for train,test in splits],local['mcmc'])
    else:
        print(run(config,directory=args.run))


if __name__=='__main__':
    main()
