# Shared components in 1.6.1

The shared model implements a dynamic factor with **fixed loadings**. The first
scientific specification uses unit loadings and additive channel departures.
It estimates common change in the original measurement units, alongside the
extent to which each marginal moves differently. It does not estimate free
factor loadings. An optional residual Gaussian copula can be added separately; see `COPULA_AND_ORDERING.md`.

## Model and identification

For the six Uccle summaries, the original-orientation location predictor is

\[
\eta_{i,t}=a_i+s_{i,t}+f_t+d_{i,t},\qquad
\sum_{i=1}^{6}w_i d_{i,t}=0,\quad \sum_iw_i=1.
\]

Here `a_i` is a private baseline, `s_i` is a private seasonal cycle, `f` is a
common trend, and `d_i` is the channel departure. Gaussian channels use this
predictor as a mean; GEV channels use it as a location parameter. These are
different distributional summaries. The model's common path is therefore a
descriptive average location change, not automatically a mean temperature or a
causal estimate of externally forced warming.

The common and departure trend levels are fixed at zero in the initial state,
immediately before the first observation. Baseline differences belong to the
private intercepts. Initial slopes may have proper Gaussian uncertainty. With
unit loadings and static private intercepts, the deseasonalized channel changes
satisfy

\[
\Delta\eta^{\mathrm{level}}_{i,t}=\Delta f_t+\Delta d_{i,t},\qquad
\sum_i w_i\Delta\eta^{\mathrm{level}}_{i,t}=\Delta f_t.
\]

The constraint is imposed within every posterior draw, not by recentering fitted
curves afterwards. For a fixed weight vector `w`, an orthonormal matrix `B`
spans its null space: `w.T @ B == 0`, `B.T @ B == I`. The compiler builds five
contrast processes `z_t` for six channels and reconstructs `d_t = B @ z_t`.
The same construction enforces the constraint in forecasts.

Each innovation type has one SD shared by all contrasts. Their Gaussian prior
is consequently invariant to the arbitrary orthonormal basis. These are
correlated channel departures with group shrinkage, not six independent
departure paths or six independent selection indicators. Equal weights give
`Cov(d innovations) = q_d**2 * (I - 11.T / K)`; thus `q_d` is a contrast SD and
the per-channel innovation SD is `q_d * sqrt(1 - 1/K)`.

The generic `Shared(..., loadings={...})` declaration accepts other **known**
loadings. Omitted channels then have zero loading. `Departures(..., weights=...)`
accepts fixed nonnegative weights naming every channel. Weights define the
scientific average and should be chosen before comparing fits. They are not
automatically estimated precision weights. Obviously redundant shared groups
with identical dynamics are rejected. More complex overlapping state designs
still require an identifiable scientific specification; adding private dynamic
trends can make common/departure allocation heavily prior dependent.

## Construction through the general API

```python
import bucex as bx

model = bx.MultiSeriesModel(
    channels=(
        bx.Channel("mean", bx.Gaussian(), (
            bx.LocalLevel(mode="static", initial_mean=10, initial_sd=3),
            bx.DummySeasonal(12, mode="static", initial_sd=3),
        )),
        bx.Channel("minimum", bx.GEV(), (
            bx.LocalLevel(mode="static", initial_mean=0, initial_sd=3),
            bx.DummySeasonal(12, mode="static", initial_sd=3),
        ), tail="lower"),
    ),
    shared=(
        bx.Shared("warming", bx.LocalLinearTrend(
            initial_level=0, initial_level_sd=0, initial_slope_sd=0.005,
        )),
        bx.Departures("departure", bx.LocalLinearTrend(
            initial_level=0, initial_level_sd=0, initial_slope_sd=0.003,
        )),
    ),
)
```

Observations supplied to `fit()` remain in their original units, including
minima. For this shared API, declared private initial means also use original
orientation. The compiler handles the reflection needed by the lower-tail GEV
likelihood; positive common warming remains positive in returned scientific
components. The established univariate transform conventions remain unchanged.

Shared declarations support `LocalLevel`, `LocalLinearTrend`, and
`DummySeasonal`. Private channels also support regression with per-channel
`exog` matrices. Supply future covariates explicitly when forecasting regression
models. A private `MultiSeriesModel` selects its route by prior type: `MarginalPriors`
for private FS/SSVS, hierarchical priors for pooling, or `JointPriors` for the
continuous joint backend. A univariate `Model` retains its scalar result API.

The Uccle helpers `make_uccle_shared_model()` and `fit_uccle_shared()` construct
these same objects and call the same inference API. They are conveniences, not
alternative models or numerical implementations. Use the general declarations above for construction. The active six-series
specification in `research/serra/models.py` now uses private FS/SSVS paths.

## Priors and seasonality

Initial-state priors belong to the component declarations. `JointPriors`
supplies process SD priors, observation scale priors for each channel, and GEV
shape priors. All keys must match the declared model exactly.

```python
priors = bx.JointPriors(
    process={
        "shared.warming.level": bx.HalfNormalSD(0.02),
        "shared.warming.slope": bx.HalfNormalSD(0.00005),
        "departure.departure.level": bx.HalfNormalSD(0.01),
        "departure.departure.slope": bx.HalfNormalSD(0.000025),
    },
    observation_sd={name: bx.InverseGammaVariance(2, 2)
                    for name in model.channel_names},
    shape={"minimum": bx.UniformPrior(-0.5, 0.5)},
)
fit = bx.fit(data, model=model, priors=priors, engine="laplace_mh",
             mcmc=bx.MCMC(draws=1000, warmup=1000, chains=4, seed=42))
```

These numerical scales are explicit monthly demonstration settings, not a
claim of universally suitable priors. Level innovations have units of Celsius
per transition; slopes are Celsius per model step and their innovations change
that slope at each transition. The example shrinks departures more strongly
than the common signal. Check sensitivity to this choice, including setting
departure innovation SDs to zero and varying their shrinkage scales.

For a local linear trend written as
`f_t = f_(t-1) + b_(t-1) + e_t` and `b_t = b_(t-1) + u_t`, the innovation-only
variance of an `h`-step level change is

\[
h q_f^2+\frac{h(h-1)(2h-1)}{6}q_b^2.
\]

Uncertain starting slope adds `h**2 * Var(b_0)`. This makes prior simulation of
decadal changes more informative than judging small-looking SDs individually.
The default initial slope SD also matters over long records. Calibrate all
these quantities together in the observation frequency used for the paper.

`FixedSD(0)` removes an innovation, but a trend with uncertain nonzero initial
slope can still change linearly. For a truly absent departure, omit its
declaration; fixing its process noise alone does not necessarily remove it.
Continuous priors imply no atom at zero, so do not report their posterior as a
probability that a departure is exactly absent. Univariate and hierarchical
SSVS remain available through their existing prior classes.

When `priors=None`, the shared backend records proper data-adaptive convenience
priors, including PC priors for process SDs. These use the observed data for
calibration. Research comparisons should specify priors and use only training
data for any calibration during predictive validation.

**Each series can have its own seasonal cycle.** The first Uccle configuration
uses `DummySeasonal(12, mode="static")` privately in every channel. Its seasonal
coefficients are estimated with uncertainty; static means they repeat over
years. `mode="dynamic"` allows evolving private seasonality and requires the
corresponding process priors. An explicit shared seasonal declaration is also
available, but a shared cycle plus unrestricted private cycles needs additional
identifying choices. It is not the default warming specification.

## Inference and scientific output

All-Gaussian shared models use FFBS. Any GEV channel uses a joint Laplace path
proposal with an exact-likelihood MH correction, followed by a fixed sweep of
Gaussian-prior elliptical slice updates over private and shared state blocks.
This supplementary update helps move local intercepts and seasonal coefficients
when a full joint path proposal has low acceptance. All contrasts in a departure
group are updated together, retaining invariance to their orthonormal basis.
The slice algorithm follows [Murray, Adams and MacKay (2010)](https://proceedings.mlr.press/v9/murray10a.html).

`SharedSampler(elliptical_slice_steps=1, maximum_slice_evaluations=200)` records
the default one full supplementary sweep. Set `elliptical_slice_steps=0` for an
explicit sampler ablation. The sweep has a fixed schedule; it is not triggered
only when MH rejects. Gaussian FFBS fits need no supplementary slice updates.
The shared backend supports these exact-target routes only. Its scale updates
combine centered conditional updates with noncentered scale/path moves to
reduce state-scale dependence.
Adaptation stops after warmup. Numerical update failures abort instead of
silently retaining a failed draw as though the update had succeeded.

```python
common = fit.shared_draws("warming")
departure = fit.departure_draws("minimum", name="departure")
level = fit.component_draws("level", channel="minimum")
season = fit.component_draws("seasonal", channel="minimum")
fit.diagnostics()
fit.plot("shared", name="warming")
fit.plot("departures", name="departure")
future = fit.forecast(12, seed=43)
future.shared_draws("warming")
fit.save("shared.bucex")
```

Component methods return joint posterior draws, combining chains by default.
Use `combine_chains=False` for chain-specific assessment. A scientifically
useful contrast is the change in departure between two dates, with its interval
and `P(change > 0)`. Compute contrasts within draws so covariance is retained:

```python
import numpy as np
change = departure[:, -1] - departure[:, 0]
interval = np.quantile(change, [0.05, 0.5, 0.95])
probability_increase = np.mean(change > 0)
```

Retain chain axes when assessing Monte Carlo exploration of those contrasts:

```python
common_by_chain = fit.shared_draws("warming", combine_chains=False)
departure_by_chain = fit.departure_draws("minimum", combine_chains=False)
fit.contrast_diagnostics({
    "common_change": common_by_chain[..., -1] - common_by_chain[..., 0],
    "minimum_departure_change": departure_by_chain[..., -1] - departure_by_chain[..., 0],
})
```

`bx.summarize_draws()` also accepts any named mapping of `(chains, draws)`
arrays. Constant or nonfinite quantities are flagged rather than assigned
misleading convergence diagnostics. The engine's `joint_state_mh_acceptance`
describes the Laplace-MH proposal alone; slice movement, angle and likelihood
evaluation summaries describe the supplementary state updates separately.

This contrasts the first and last **observed** states. `include_initial=True`
also returns the anchored state preceding the observations. Report endpoint
contrasts, meaningful window averages, and rates with their time units. Level
contrasts exclude seasonality. Predictor contrasts can include seasonal and
regression changes; these answer a different question.

Joint fitting propagates common-path uncertainty into every channel and retains
posterior covariance in contrasts and forecasts. It does not guarantee narrower
or better calibrated intervals. Without a copula, observation errors remain
conditionally independent across channels, so six summaries from the same
daily records may contribute less independent information than assumed.
Version 1.6.1 can include a Gaussian residual copula in the likelihood and
prediction; compare controlled fits and held-out calibration. This copula does
not impose physical ordering. Report pointwise credible intervals as pointwise;
see `COPULA_AND_ORDERING.md` for compound risks and ordering checks.

Current limits are aligned observation times, stationary observation scales
in multiseries models, fixed loadings, and continuous/fixed shared-scale priors.
See `VALIDATION.md` for the numerical tests, recovery pilots, and remaining
scientific experiments before using the shared model in the manuscript.

Use NaN for a missing channel observation. Each channel needs at least two
finite observations and every retained time needs at least one observed
channel; rows missing every channel are rejected. Covariates must be finite.

In 1.6.3 the active SERRA study uses private FS/SSVS trajectories, with optional
residual copula feedback. It does not use these common states or estimate free
factor loadings. Shared-state examples are retained here as general API documentation.
