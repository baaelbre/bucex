"""Compare fixed/estimated/static/dynamic components with the same data."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import bucex as bx


def main():
    y = 1+.03*np.arange(24)+np.random.default_rng(5).normal(0,.2,24)
    priors = bx.Priors(initial_level=bx.Normal(0,2), initial_slope=bx.Normal(0,.1),
                      variance=bx.Fixed(.2**2))
    models = {
        'constant': bx.Model(bx.Gaussian(), [bx.LocalLevel('static')], priors),
        'linear': bx.Model(bx.Gaussian(), [bx.LocalLinearTrend('static','static')], priors),
        'local_level': bx.Model(bx.Gaussian(), [bx.LocalLevel()], priors),
        'local_trend': bx.Model(bx.Gaussian(), [bx.LocalLinearTrend()], priors),
    }
    output = Path('results/examples/components'); output.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        fit = bx.fit(y, model=model, steps_per_year=4,
                     mcmc=bx.MCMC(draws=30,warmup=20,chains=2,progress=False))
        fit.save(output/(name+'.bucex'))
        bx.plot(fit, type='level', path=output/(name+'.png'))
    print(output)


if __name__ == '__main__':
    main()
