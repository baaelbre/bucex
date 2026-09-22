# Shared innovation shrinkage in BUCEX 1.8.1

The hierarchy learns how strongly to regularize a set of related structural
series. It shares prior scales, while every series keeps its own level, slope,
seasonal path, innovation SDs and observation parameters. It is distinct from
both a dynamic factor (a common realized latent process) and a copula
(contemporaneous residual dependence).

## Model and units

Let a[j,c] be the signed FS innovation coefficient for response j and component
c. The physical innovation SD is tau[j,c] = abs(a[j,c]). For selected components:

```text
q = Phi^(-1)(0.75) = 0.67448975...
a[j,c] | m[c] ~ Normal(0, (m[c]/q)^2)
log(m[c]/anchor[c]) ~ Normal(0, log_sd^2)
```

Conditional on m[c], the median of tau[j,c] is exactly m[c]. The hyperprior
anchor is the median of m[c], not the marginal median of tau[j,c]. That marginal
prior is a mixture and is drawn correctly by `draw_marginal_prior`.
The prior is proper, has no point mass at zero, and does not give a component
selection probability. With `log_sd=log(2)`, the hyperprior's 95% interval is
approximately [0.257, 3.89] times its anchor. Its upper tail is not a hard cap.

One common m[c] is estimated from all eligible responses. Weakly identified
individual SDs borrow information through it; well-supported large innovations
can raise it. This is a substantive exchangeability assumption about component
roughness. The six Uccle responses have a common temperature unit and monthly
time step, making this a plausible sensitivity model, not a guarantee that
means and extrema have identical evolution. It does not treat the responses
as six independent temperature records: their joint likelihood includes the
copula. Correlated weather and a small number of responses can still leave the
hyperparameters weakly identified.

Level innovations are in degrees Celsius per monthly transition. Slope is in
degrees Celsius per monthly time step, and its innovations change that slope
each transition. The median numbers depend on the response and time units;
rescale them if a dataset is standardized or time steps change. The release
uses physical temperatures, not response-variance standardized innovations.
Shared standardized latent innovations are not introduced.

## General API

```python
import bucex as bx

components = [bx.LocalLinearTrend(), bx.DummySeasonal(12)]
model = bx.MultiSeriesModel([
    bx.Channel("average", bx.Gaussian(scale=bx.SeasonalScale(12)),
               parameters={"mu": bx.Latent(components)}),
    bx.Channel("maximum", bx.GEV(scale=bx.SeasonalScale(12), xi_bounds=(-.5,.5)),
               parameters={"mu": bx.Latent(components)}),
], copula=bx.GaussianCopula(eta=1.))

priors = bx.MarginalPriors(
    channels={
        "average": bx.fs_priors("gaussian", period=12, innovation="normal"),
        "maximum": bx.fs_priors("gev", period=12, innovation="normal"),
    },
    shrinkage=bx.SharedShrinkage(
        medians={"level": .0025, "slope": .0000125, "seasonal": .02},
        log_sd=.6931471805599453,
    ),
)

# y is a complete, aligned monthly DataFrame with these two column names.
# Put this call inside main(), with an if __name__ == "__main__" guard
# when chain_workers > 1 in a standalone Python script.
fit = bx.fit(y, model, priors=priors, parameterization="fs",
             mcmc=bx.MCMC(chains=4, chain_workers=4, warmup=500, draws=500))

shared = bx.compare_shared_shrinkage(fit)
individual = bx.compare_innovation_priors(fit, channel="average", level=.95)
future = fit.forecast(60, draws=2000, seed=174)
```

Omit `shrinkage=` to retain ordinary independent marginal priors. Omit
`copula=` to remove residual correlation while retaining the hierarchy.
`medians` can select level, slope and/or seasonal innovations. The research
configuration now pools all three, with a separate shared median for each.
Selecting only `level` and `slope` retains the 1.8.0 research specification.
Initial slopes and initial seasonal patterns remain as declared per channel.
Each pooled component needs
at least two channels with that innovation active. Fixed-zero innovations do
not contribute to its hyperparameter conditional. Components must use
continuous, zero-mean normal FS priors; combining this hierarchy with SSVS or
local-mixture priors is rejected explicitly.

The seasonal hyperparameter governs **evolution of the seasonal pattern**.
Shrinking it towards zero approaches fixed repeating seasonality, not an
absence of seasonality: initial monthly effects can still be large and differ
across responses. It neither pools the initial seasonal vector nor the
observation scale. Repeating monthly observation scales describe seasonal
weather dispersion and retain their separate priors. The level, slope and
seasonal shared medians are never constrained to be equal to one another.

These are hierarchical shrinkage hyperparameters. The signed FS coefficients
have conditional normal priors; integrating over the shared medians produces
normal scale mixtures. This is partial pooling of regularization, not a shared
seasonal state or a deterministic common smoothness parameter.

The univariate `Model`, `fs_priors` and `fit` signatures are unchanged. A single
series cannot learn a cross-series hierarchy through a univariate fit.
An executable example is in `docs/examples/shared_shrinkage.py`.
Existing general parameter-evolution, factor and hierarchy APIs are retained;
this new feature is scoped to FS location innovations and is not silently
applied to latent scale/shape parameters.

## Computation and posterior feedback

The private FS sampler updates each channel's latent path and coefficients
conditional on the current shared scales and current copula. GEV path proposals
retain the exact Metropolis-Hastings correction. It then updates the shared
scales and the copula. For u = log(m/anchor), the exact hyperparameter
conditional, up to constants independent of u, is:

```text
log p(u | a) = -J*u
              - 0.5*sum((q*a[j]/anchor)^2)*exp(-2*u)
              - 0.5*(u/log_sd)^2
```

J counts active channels. The `-J*u` term is the coefficient-prior
normalization and is essential. A stepping-out slice update targets this
one-dimensional density. Paths are conditionally standardized in FS
coordinates, so they do not supply additional scale-normalization terms here.
No plug-in marginal fits or empirical-Bayes hyperparameter estimates are used.

The saved `shrinkage.shared.level`, `shrinkage.shared.slope` and
`shrinkage.shared.seasonal` draws remain
paired with channel states, observation parameters and copula parameters.
Forecasting uses those posterior draws and new future state innovations;
static hyperparameters are not independently redrawn for each future month.
Both common-scale learning and copula feedback can alter trajectories and
uncertainty. Neither guarantees narrower intervals. This is fully Bayesian
conditional on the stated model and hyperpriors; calibrating anchors using the
same observed record is a modeling choice whose sensitivity remains relevant.

Chains can run in parallel. Seed streams and chain ordering match serial
execution. Archives retain the hierarchy and shared-scale draws; warm starts
restore those scales with the states and observation parameters. The archive
schema remains 2.12.0, as in 1.8.0. Old archives
without this field retain their original independent-prior meaning; older
BUCEX versions are not expected to read the new hierarchy.

## Reports and the paper decision

`save_shared_shrinkage_report` exports hyperprior/posterior intervals, physical
scale traces and R-hat/ESS. `compare_innovation_priors(fit, channel=...)` compares
each SD against its unconditional prior, integrating m. It does not evaluate
that prior at a fitted hyperparameter. Use `draw_marginal_prior` to inspect joint
prior draws with the correct common-scale dependence.

`SensitivityReport` compares compact univariate or joint research reports.
For joint reports map each response to its common `joint/` directory. Global
hyperparameter, joint-score and joint-target tables are included once per fit;
per-response tables select the correct channel. Exact forecast case matching
is required. Monthly log-scale contrasts are now included in scalar R-hat/ESS
and parameter traces rather than omitted because they form a vector.

Follow [START_HERE](../START_HERE.md) for the full experiment commands. The
seasonal study compares three-component pooling against level/slope-only
pooling, then halves and doubles the seasonal hyperprior anchor (.01 and .04
versus .02) while holding the level/slope anchors fixed. The adequacy study
also compares fixed location seasonality; that case has no seasonal innovation
and therefore declares only level/slope pooling. Comparison figures mark
absent hyperparameters as `not pooled`; their absence is not a posterior at
zero. All historical fits re-estimate their hierarchy using training data only.

A useful final common specification requires:

- Stable scientific targets across the quarter and half hyperprior anchors,
  allowing for Monte Carlo error, without requiring identical nuisance SDs.
- Robustness to seasonal pooling and its anchor, particularly for month-specific
  risk and late-period seasonality. Pooling should be supported by prediction
  and sensitivity, not selected solely for narrower intervals.
- Useful chain movement and adequate ESS for scientific targets and nuisance
  parameters, including initial slopes, monthly scales and common medians.
- Acceptable held-out marginal and joint prediction, seasonal coverage and
  PIT diagnostics; inspect the 2019 forecast failures rather than averaging
  them away.
- Plausible posterior replications, residual serial/cross-series dependence
  and ordering-violation diagnostics. A Gaussian copula alone does not enforce
  the deterministic ordering of minimum, mean and maximum summaries.

Posterior displacement from a prior median is descriptive, not a calibration
objective. Near-zero innovations can be supported by data; a posterior similar
to its prior may reflect weak information. Neither justifies changing the prior
until a desired displacement appears. Acceleration concerns period slope
contrasts under the stated decomposition. If it is sensitive to anchors, report
that sensitivity while keeping robust warming and risk results central.

This release does not claim the final model fits the data adequately. The
included software checks validate density calculations, archive/parallel
contracts and execution, not the substantive temperature conclusions.
