"""Regression/TVP fitting, future covariates, serialization and native plots."""
from pathlib import Path
import numpy as np
import pandas as pd
import bucex as bx


def main():
    rng=np.random.default_rng(31)
    x=pd.DataFrame({'wind':rng.normal(size=60)})
    y=5+1.2*x.wind.to_numpy()+rng.normal(0,.4,60)
    model=bx.Model(bx.Gaussian(),[
        bx.LocalLevel('static',initial_prior=bx.Normal(0,10)),
        bx.Regression(('wind',),mode='dynamic',prior=bx.Normal(0,2),innovation_prior=bx.Normal(0,.03)),
    ],bx.Priors(variance=bx.InverseGamma(2,.16)))
    fit=bx.fit(y,model=model,exog=x,mcmc=bx.MCMC(draws=40,warmup=30,chains=2,progress=False))
    output=Path('results/examples/regression');output.mkdir(parents=True,exist_ok=True)
    restored=bx.load(fit.save(output/'fit.bucex'))
    future_x=pd.DataFrame({'wind':np.linspace(-1,1,12)})
    future=restored.predict(12,exog=future_x,draws=300)
    bx.plot(restored,type='component',component='regression.wind',path=output/'coefficient.png')
    bx.plot(future,type='forecast',history=restored,path=output/'forecast.png')
    print(restored.summary())


if __name__=='__main__': main()
