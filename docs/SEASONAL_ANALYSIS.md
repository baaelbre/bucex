# Seasonal blocks, comparable prediction, and future r-largest models

## Scope of 1.8.4

The release provides complete seasonal r=1 models for all six summaries, joint
hierarchical shrinkage, seasonal-scale effects, four shrunk copula matrices,
independent fallback fits, posterior diagnostics and matched forecast assessment.
The separate research workflow is `research/seasonal`; runnable commands
are in `START_HERE.md`.

It also provides raw r-largest/r-smallest extraction and runs-cluster diagnostics.
These assess whether an r>1 extension is sensible. It does **not** fit an r>1
state-space model or estimate event-cluster occurrence rates. That would need a
joint block likelihood, rank-aware prediction, rounding/selection treatment,
and a dependence construction appropriate to vector observations. A list of
ranked values passed to scalar `GEV` is not such an implementation.

## Data and calendar

`derive_uccle_seasonal` reads daily TX/TN and builds complete DJF/MAM/JJA/SON
blocks. Each channel uses exactly the same complete days. Seasonal means are
daily-weighted; minima retain their original sign until the model's reflection.
Dates label season starts, so DJF 2026 is timestamped December 1, 2025.
Metadata record actual last included days separately from those timestamps.

Only January–February 1892 are explicitly excluded in the research configs.
The incomplete initial DJF is omitted. The final JJA 2026 **includes August**.
There are 538 seasonal blocks, exactly one third of the 1,614 retained monthly
blocks. Missing interior dates or temperatures raise errors; the code does not
silently turn an incomplete block into a complete extreme. Raw observations are
preserved. The existing daily TN>TX provenance remains visible in quality data.

Public API:

```python
import bucex as bx

seasonal = bx.derive_uccle_seasonal(
    'data/Uccle_31_08_26.csv',
    exclude_months=['1892-01', '1892-02'],
)
model = bx.Model(
    bx.GEV(scale=bx.LogScale('constant',
        bx.SeasonalScale(4, prior_sd=.3, calendar='meteorological'))),
    parameters={'mu': bx.Latent([bx.LocalLinearTrend(), bx.DummySeasonal(4)])},
)
```

Explicit meteorological scale/correlation phases are DJF=1, MAM=2, JJA=3,
SON=4. They use dates, including forecasts. A plain `SeasonalScale(4)` retains
its original relative-phase semantics for compatibility. A
`SeasonalGaussianCopula(period=4, structure='seasons')` uses the dated
meteorological convention; the period-12 version groups calendar months.
The shared model builder declares these choices explicitly.

Every series retains its own latent seasonality and its own four scale effects.
The shared seasonal hyperparameter regulates *how much location seasonality can
change*; it is neither a shared seasonal pattern nor a shared observation scale.

## Physical prior equivalence

At H updates, the innovation-to-level gains are sqrt(H) and
sqrt(H(H-1)(2H-1)/6). The dummy-season gain is computed from its actual transition
matrix. At 30 years the monthly H=360 and seasonal H=120 dummy-season gains both
equal sqrt(60). Use the same physical effects with `SharedShrinkage.from_effects`:

```python
shared = bx.SharedShrinkage.from_effects(
    horizon=120, period=4, slope_time_unit=40,
    level_displacement_sd=.02 * 360**.5,
    slope_displacement_sd=.00005 * (360*359*719/6)**.5,
    seasonal_displacement_sd=.02 * 60**.5,
    initial_slope_sd=.30, calibration='marginal',
)
```

These values reproduce the declared COMPSTAT second moments after integrating
lognormal hyperpriors. They are not estimated from Uccle and not universal
climate constants. They match component-wise RMS effects at one horizon;
they do not derive a quarterly model by exact marginalization of a monthly one.
Slope plots use 40 seasonal rather than 120 monthly updates per decade.
Initial-pattern priors remain explicit dummy-coordinate priors, as in the
monthly model; they are not twelve/four exchangeable month/season effects.

## Comparing models on a common outcome

A monthly maximum and a seasonal maximum are different observations. Comparing
their raw likelihoods, BIC values or GEV locations is not a fair performance
comparison. `compare.py` uses identical daily training spans and block-end
forecast cutoffs. It aggregates complete monthly forecast paths to seasons and
scores them against the same daily-derived seasonal summaries as the direct
seasonal forecast. It verifies that the two observed aggregations agree.

For Gaussian means, conditional aggregate variance is the sum of squared
weights times observation variances. For maxima the conditional CDF is the
product of monthly CDFs; for minima it is one minus the product of survival
probabilities. Aggregate densities differentiate these products and handle GEV
support boundaries without subtracting infinities. Parameters and future state
paths are paired within each draw and integrated afterwards. Thus serial
uncertainty induced by latent paths is retained; residual noise is conditionally
independent across blocks under both fitted models.

The public `Forecast.aggregate(...).conditional_log_density`, `.pit` and `.score`
methods support both monthly-to-seasonal and direct seasonal forecasts.
`score_seasonal_forecast` exports the common-target scores and calibration data.
These are marginal scores, not a six-dimensional seasonal joint density.

Short and long forecast horizons must be inspected separately. The comparison
exports CRPS, log scores, interval coverage/width, PITs and threshold scores by
season, origin and lead group. Numerical diagnostics remain active. Two origins
support screening, not precise ranking uncertainty. Different resolution models
can disagree for substantive reasons even with calibrated priors.

Yearly aggregation of seasonal data is previous December–November, with DJF
assigned to its ending year. It is not a January–December average: the latter
cannot be recovered exactly by splitting an observed DJF mean. Forecast risk
labels and notes record this convention. Monthly forecasts retain their usual
calendar-year aggregation.

## r-largest and clustering

`ranked_extremes(series, r=3, tail='upper')` returns raw daily ranks, dates and
boundary ties for complete seasons. `rank_clustering_diagnostics` summarizes
minimum spacing, consecutive days, within-week span and common-month membership.
Ties are broken deterministically by earlier date and remain flagged.

`extreme_clusters(series, threshold, run_length=3, tail='upper')` groups threshold
exceedances separated by fewer than three non-exceeding days. A gap with three
non-exceeding days starts a new cluster. It retains cluster peaks, timing,
duration and cross-season membership. Clustering occurs on the continuous daily
record before seasonal assignment, preventing duplicate episode counts at a
boundary. The research script examines 1/3/5-day run rules and monthly 95th/5th
percentile thresholds from the declared 1961–1990 reference.

Proximity is not proof of a common meteorological event. Fixed historical
thresholds also do not remove warming, and a runs rule does not establish
independence. Inspect threshold/run sensitivity and the number of qualifying
clusters. Few-cluster seasons must not be filled with ordinary observations to
force r=3. Retaining multiple days from one episode can invalidate a standard
unclustered r-largest approximation even though its joint density already
accounts for rank dependence. A cross-summary Gaussian copula does not supply a
daily clustering model.

Seasonal blocks may improve the approximation for a block maximum, but the
third-largest value is less extreme and intra-season changes remain. Under iid
continuous F, the expected upper-tail probability is 1/(n+1) for a maximum and
r/(n+1) for the rth largest. Thus a maximum of 30 observations (1/31) and a
third-largest of 90 (3/91) probe a similar tail fraction. This illustration is
not a temperature asymptotic guarantee. Assess block length and r together.

References:

- Bader, Yan & Zhang (2017), *Statistics and Computing* 27, 1435–1451.
  https://doi.org/10.1007/s11222-016-9697-3
- Ferro & Segers (2003), *JRSS B* 65, 545–556.
  https://doi.org/10.1111/1467-9868.00401

The main revision can retain monthly models while reporting a focused seasonal
check in the supplement. Adoption of seasonal blocks should follow their
predictive adequacy and the scientific event definition, not smoother-looking
trajectories or an assumed asymptotic advantage.
