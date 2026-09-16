"""Small runnable API example; these short chains are not a convergence study.

From the release directory: python docs/examples/parameter_evolution.py
Outputs are written to results/parameter_evolution, outside the paper workflow.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import bucex as bx


def main():
    directory = Path('results/parameter_evolution')
    directory.mkdir(parents=True, exist_ok=True)
    model = bx.Model(bx.Gaussian(), parameters={
        'mu': bx.Latent([bx.LocalLinearTrend(), bx.DummySeasonal(12)]),
        'sigma': bx.Latent([bx.LocalLevel(), bx.DummySeasonal(12, mode='static')],
            priors=bx.EvolutionPriors(innovation='lasso')),
    })
    truth = bx.simulate(model, 120, params={
        'sigma': 1., 'sd.level': .03, 'sd.slope': .0001, 'sd.seasonal': .01,
        'scale.sd.level': .02, 'scale.initial.seasonal': [.1]*11,
    }, seed=17)
    y = pd.Series(truth.y,index=pd.date_range('2000-01-01',periods=120,freq='MS'))
    fit = bx.fit(y, model, priors=bx.fs_priors('gaussian'), parameterization='fs',
        mcmc=bx.MCMC(chains=2,warmup=40,draws=40,seed=18,progress=False))
    fit.save(directory/'fit.bucex')
    fit.diagnostics()['parameters'].to_csv(directory/'diagnostics.csv')
    for parameter in ('mu','sigma'):
        axis = fit.plot('parameter_path',parameter=parameter)
        axis.figure.tight_layout()
        axis.figure.savefig(directory/(parameter+'.png'),dpi=150)
        plt.close(axis.figure)
    fit.forecast(24,seed=19).summary().to_csv(directory/'forecast.csv',index=False)
    bx.save_prediction_report(fit,directory,threshold=2.,draws=40,seed=20,image_format='png')
    print(directory.resolve())


if __name__ == '__main__':
    main()
