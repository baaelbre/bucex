# BUCEX 1.8.7

Bayesian unobserved component models for Gaussian temperature summaries and GEV block extremes. This release supports private structural trajectories for each series, continuous innovation priors, optional shared shrinkage of **prior scales**, and an optional Gaussian residual copula. The research workflows are [monthly](research/monthly/README.md) and [seasonal](research/seasonal/README.md).

```bash
python -m pip install -e ".[plot,test]"
python -c "import bucex; print(bucex.__version__)"
```

The monthly window is March 1892–August 2026 (1,614 months). The seasonal reference has 538 complete DJF/MAM/JJA/SON blocks, through JJA 2026. Version 1.8.7 locks the scientific settings to the successful `uccle_copula_20260923T222238_406159Z.zip` run: shared-scale medians per season are `(0.0144513322, 0.000108839649, 0.00834348054, 0.00463877353)` for level, slope, seasonality and initial slope. The 1.8.6.1 seasonal-state initialization correction is retained. [FINAL_RUN.md](FINAL_RUN.md) gives the exact final-run, checking and manuscript-figure commands.

## Model and fit

```python
import bucex as bx

channels = (
    bx.Channel("TXm", bx.Gaussian(scale=bx.SeasonalScale(12)),
               (bx.LocalLinearTrend(), bx.DummySeasonal(12))),
    bx.Channel("TXx", bx.GEV(scale=bx.SeasonalScale(12)),
               (bx.LocalLinearTrend(), bx.DummySeasonal(12))),
)
model = bx.MultiSeriesModel(channels, copula=bx.SeasonalGaussianCopula())
priors = bx.MarginalPriors({
    c.name: bx.fs_priors(c.family, period=12, innovation="normal")
    for c in channels
})
data = bx.load_uccle_multiseries(series=["TXm", "TXx"], end="2026-08-01")
fit = bx.fit(data, model, priors=priors, parameterization="fs",
             mcmc=bx.MCMC(chains=4, warmup=1000, draws=1000, seed=2026))
fit.save("two_series.bucex")
```

`MarginalPriors(..., shrinkage=bx.SharedShrinkage(...))` can learn common normal-prior widths for level, slope, seasonal innovations and initial slopes. Each series retains its own state path and innovation SD. See [shared shrinkage](docs/SHARED_SHRINKAGE.md) and [physical calibration](docs/PRIOR_CALIBRATION.md). Univariate `bx.fit(y, bx.Model(...), priors=bx.fs_priors(...))` remains available. Continuous normal, lasso, horseshoe, triple-gamma and PC prior profiles remain supported.

The default observation scale is constant through time. `SeasonalScale(12)` adds repeating calendar-month effects, and `SeasonalScale(4, calendar="meteorological")` adds repeating season effects. Whether those effects improve the scientific fit is tested in both workflows with `constant_dispersion` alternatives; the research configs do not assume that the answer is known.

Models use finite, aligned observations for the private multiseries sampler. Gaussian channels use conditional FFBS; mixed or GEV channels use exact-likelihood-corrected Laplace–MH. The copula is fitted jointly with the margins and is used for forecasts and joint scores. A Gaussian copula models residual dependence, not a common latent warming factor. Seasonal aggregation and raw daily rank/clustering checks are available, while the fitted seasonal extrema use one extreme per block.

## Checks

```bash
python -m pytest -q
python -m research.monthly.preflight --config research/monthly/config/main.json
python -m research.seasonal.compare --plan
```

The smoke configurations exercise code paths on short windows; their few draws are not scientific evidence. Long empirical runs, convergence checks and held-out predictions remain prerequisites for manuscript claims.
