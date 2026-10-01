"""Named mixed-family channels, three shared scales, two parallel chains."""
from pathlib import Path
import numpy as np
import pandas as pd
import bucex as bx


def main():
    rng = np.random.default_rng(3)
    t = np.arange(32)
    dates = pd.date_range('2000-02-29', periods=32, freq=pd.offsets.QuarterEnd(startingMonth=2))
    seasonal = 3*np.sin(2*np.pi*t/4)
    data = pd.DataFrame({'mean': 10+.03*t+seasonal+rng.normal(size=32),
                         'maximum': 18+.05*t+seasonal+rng.gumbel(size=32),
                         'minimum': 2+.04*t+seasonal-rng.gumbel(size=32)}, index=dates)
    components = [bx.LocalLinearTrend(), bx.DummySeasonal(4)]
    scale = bx.SeasonalScale(4)
    model = bx.MultiSeriesModel([
        bx.Channel('mean', bx.Model(bx.Gaussian(scale), components)),
        bx.Channel('maximum', bx.Model(bx.GEV('upper', scale), components)),
        bx.Channel('minimum', bx.Model(bx.GEV('lower', scale), components)),
    ], pooling=bx.Pooling(level=bx.HalfNormal(.1), slope=bx.HalfNormal(.002), seasonal=bx.HalfNormal(.1)))
    fit = bx.fit(data, model=model, mcmc=bx.MCMC(draws=20, warmup=10, chains=2, workers=2))
    output = Path('results/examples/multiseries'); output.mkdir(parents=True, exist_ok=True)
    fit.save(output/'fit.bucex')
    bx.plot(fit, channel='maximum', type='slope', path=output/'maximum_rate.png')
    bx.plot(fit, channel='minimum', type='risk', threshold=0, tail='lower', path=output/'cold_risk.png')
    print(fit.summary())


if __name__ == '__main__':
    main()
