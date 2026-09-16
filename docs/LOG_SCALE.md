# Observation variability and location seasonality

Each channel has separate location states and an observation-scale model.
`DummySeasonal` changes location; `SeasonalScale` changes log observational scale.
They need not have the same amplitude or phase and are never forced to share a
latent path across the six summaries.

```python
bx.Gaussian(scale=bx.LogScale("constant", seasonal=bx.SeasonalScale(12)))
bx.GEV(scale=bx.LogScale("linear", seasonal=bx.SeasonalScale(12),
                        slope_sd=.1, time_unit=120))
bx.GEV(scale=bx.LogScale("rw", seasonal=bx.SeasonalScale(12),
                        innovation_sd=.01))
```

The model is `log(sigma_t) = log(sigma) + d[month(t)] + offset_t`.
Seasonal effects sum to zero. Orthonormal contrast coefficients independently
follow N(0,.3²) by default, so the variance of an individual monthly effect is
.3²(1−1/12). No reference month is privileged. The baseline sigma is therefore
the geometric mean seasonal scale when the secular offset is zero.

| Secular mode | Offset | Declared prior |
|---|---|---|
| constant | 0 | No secular parameter |
| linear | slope × t / time_unit | slope ~ N(0,slope_sd²) |
| rw | omega × z_t; z_0=0, z_t=z_(t−1)+N(0,1) | omega ~ N(0,innovation_sd²) |

Observations start at t=1; `time_unit=120` denotes a decade for monthly data.
Seasonality may be omitted. The initial baseline scale prior remains in
`FSGaussianPriors`/`FSGEVPriors`, separately from these secular priors.

The private sampler updates every scale term under the full conditional
likelihood, including the residual copula. RW path updates use an elliptical
slice; no approximation replaces the likelihood. Location ASIS is not a scale
RW interweaving update, so this optional RW can still mix slowly. Check its
physical `scale_rw_sd`, final offset, sigma paths and risks; do not use arbitrary
iteration counts as evidence that the added flexibility was identified.

`fit.sigma_draws(channel=...)` includes both seasonality and the secular path.
Forecasts advance the linear time index and simulate fresh RW innovations from
the last scale state. Saved fits retain the baseline, seasonal effects, signed
coefficient and standardized path needed for restart. Dated monthly data use
calendar months; undated data begin at phase one and advance from the fitted
record length in forecasts.

The established univariate `GEV(phi="linear"|"rw"|"ssvs")` API remains available,
with its original `PhiPrior` parameterization. It is a different declaration
from `LogScale`; do not specify both. Its linear time basis and variance priors
must not be confused with the new fixed-unit signed-coefficient formulation.

Use the staged `research.serra.model_comparison` study to assess added scale
structure. Compare residual PIT by season, serial residual behavior, held-out
calibration and marginal/compound risk changes. Avoid choosing scale flexibility
solely from smoothed in-sample residuals.
