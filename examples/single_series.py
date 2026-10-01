"""A self-contained seasonal GEV example. Run: python examples/single_series.py"""
from pathlib import Path
import numpy as np
import pandas as pd
import bucex as bx


def main():
    truth = bx.Model(bx.GEV(), [bx.LocalLinearTrend(), bx.DummySeasonal(4)],
        bx.Priors(initial_level=bx.Fixed(20), initial_slope=bx.Fixed(.02),
                  initial_seasonal=bx.Fixed(1), level=bx.Fixed(.08),
                  slope=bx.Fixed(.003), seasonal=bx.Fixed(.02),
                  variance=bx.Fixed(1.5**2), shape=bx.Fixed(-.15)))
    simulated = bx.simulate(truth, 48, seed=2)
    dates = pd.date_range('2000-05-31', periods=48, freq=pd.offsets.QuarterEnd(startingMonth=2))
    y = pd.Series(simulated.y['y'][0], index=dates, name='TXx')
    model = bx.Model(bx.GEV(scale=bx.SeasonalScale(4)),
                     [bx.LocalLinearTrend(), bx.DummySeasonal(4)])
    # Deliberately short demonstration, not a converged analysis.
    fit = bx.fit(y, model=model, mcmc=bx.MCMC(draws=30, warmup=20, chains=2, workers=1))
    output = Path('results/examples/single_series'); output.mkdir(parents=True, exist_ok=True)
    fit.save(output/'fit.bucex')
    bx.plot(fit, type='level', path=output/'level.png')
    bx.plot(fit, type='normal_qq', path=output/'normal_qq.png')
    bx.plot(fit.predict(40, draws=500), type='forecast', history=fit, path=output/'forecast.png')
    print(fit.summary())


if __name__ == '__main__':
    main()
