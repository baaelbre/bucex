"""General API example: private FS locations, common shrinkage, joint copula.

Run from the checkout with `python -m docs.examples.shared_shrinkage`.
The four-draw default checks execution only; use the SERRA configs for research.
"""
import bucex as bx


def main():
    y = bx.load_uccle_multiseries(series=['TXm','TXx'], start='2023-01-01', end='2026-08-01')
    components = [bx.LocalLinearTrend(), bx.DummySeasonal(12)]
    channels = [
        bx.Channel('TXm', bx.Gaussian(scale=bx.SeasonalScale(12)),
                   parameters={'mu': bx.Latent(components)}),
        bx.Channel('TXx', bx.GEV(scale=bx.SeasonalScale(12)),
                   parameters={'mu': bx.Latent(components)}),
    ]
    model = bx.MultiSeriesModel(channels, copula=bx.GaussianCopula())
    priors = bx.MarginalPriors(
        {c.name: bx.fs_priors(c.family, period=12, innovation='normal') for c in channels},
        shrinkage=bx.SharedShrinkage(medians={'level': .0025, 'slope': .0000125, 'seasonal': .02}, initial_slope_sd=.0025),
    )
    fit = bx.fit(y, model, priors=priors, parameterization='fs',
                 mcmc=bx.MCMC(chains=4, chain_workers=4, warmup=3, draws=4, seed=180))
    print(bx.compare_shared_shrinkage(fit))
    print(bx.compare_innovation_priors(fit, channel='TXm'))
    print(fit.diagnostics()['parameters'])
    future = fit.forecast(12, draws=100, seed=181)
    print(future.summary(channel='TXx'))
    print('Monthly probabilities of TXx > 35 C:', future.probability_draws(35., channel='TXx').mean(axis=0))
    bx.save_shared_shrinkage_report(fit, 'results/shared_shrinkage_example')


if __name__ == '__main__':
    main()
