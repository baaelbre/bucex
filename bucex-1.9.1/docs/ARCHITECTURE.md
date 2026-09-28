# Paper model architecture

`Channel` declares a Gaussian or GEV observation and a local linear trend with optional dummy seasonality. `MultiSeriesModel` combines named channels with an optional `GaussianCopula` or `SeasonalGaussianCopula`. All latent trajectories belong to individual channels. The copula links contemporaneous observation residuals.

`MarginalPriors` supplies one continuous FS prior per channel. Optional `SharedShrinkage` couples **prior widths** for selected innovation components and the initial slope without equating their trajectories. The scientific declaration and month/season units live in [`research/monthly/models.py`](../research/monthly/models.py); seasonal configurations inherit the monthly baseline then override block frequency, period and physically calibrated prior values.

`bucex.fit` compiles the private state blocks, runs conditional FS updates and writes one `FitResult`. Gaussian channels use conditional FFBS and GEV channels use conditional Laplace proposals corrected by exact Metropolis–Hastings. Copula parameters and marginal states update in one posterior. `FitResult.forecast` propagates states and simulates correlated residuals with the fitted copula. Persistence uses a checksummed `.bucex` archive.

Use [START_HERE](../START_HERE.md) for executable plans and [seasonal analysis](SEASONAL_ANALYSIS.md) for complete-block aggregation and common-target prediction. The paper release does not expose shared latent factors, structural SSVS, or evolving observation-scale models.
