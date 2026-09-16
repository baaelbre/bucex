# Parameter-specific evolution

## Declarations and compatibility

`Model(observation, parameters={...})` and
`Channel(name, observation, parameters={...})` accept the same declarations:

| Parameter | Declaration | Meaning |
|---|---|---|
| `mu` | `Latent([LocalLinearTrend(), DummySeasonal(12)])` | Private structural location with identity link |
| `mu` | `Constant()` | Unknown constant location |
| `sigma` | `Constant()` | Unknown scale, constant across time and seasons |
| `sigma` | `Latent([...], link="log", priors=EvolutionPriors(...))` | Separate structural log scale |
| `xi` (GEV) | `Constant()` or omitted | Unknown time-constant shape |

The existing `components=` argument remains a location shorthand. Do not give
both it and `parameters['mu']`. Existing `SeasonalScale`, `LogScale` and saved
models remain supported. For a fixed monthly scale pattern the concise
observation declaration remains `GEV(scale=SeasonalScale(12, prior_sd=.3))`.
`LogScale('constant', SeasonalScale(...))` still has seasonal variation: its
**secular** term is constant. `Constant()` avoids that ambiguity.

## Supported structural scale components

A scale predictor has exactly one `LocalLevel` or `LocalLinearTrend`, and
optionally one `DummySeasonal`. A local level is normalized to a trend with
slope off. `level_mode='static'`, `trend_mode='static'`/`'off'`, and
`DummySeasonal(..., mode='static')` give exact reductions. Each parameter can
have a different seasonal period. Gaps/missing observations are not supported
by the private FS backend; provide a complete regular observation grid.

Location still takes its FS prior from `fit(priors=...)`. Scale evolution uses
`EvolutionPriors`, while the baseline sigma keeps the observation-scale prior
in `fs_priors`. This separates physical temperature units from log-scale units.
An initial-state prior on an ancillary component is rejected: declare it in
`EvolutionPriors` so it cannot be silently overridden.

This release binds named latent parameters to **location and scale for Gaussian
and GEV observations**. It does not claim support for arbitrary observation
families, latent GEV shape, regression on scale, or a shared scale factor.
These bindings fail explicitly. Existing shared-location inference is retained,
but does not combine with the new private structural scale backend.

## Model and identification

On the scale predictor's own coordinate system:

    log sigma_t = log sigma + a_t + g_t
    a_t = a_(t-1) + b_(t-1) + s_level e_t
    b_t = b_(t-1) + s_slope v_t
    g_t = -sum(g_(t-j), j=1,...,p-1) + s_season w_t

The initial level is **a_0=0**. The baseline `sigma` is positive and estimated;
a second free initial level would duplicate that intercept. The initial slope
and p−1 initial seasonal coordinates have separate proper normal priors.
Seasonal initial coordinates follow the established FS convention: the state
at observation 1. Dynamic dummy seasonality satisfies its stochastic recursion,
not an exact zero-sum constraint on every observed rolling year.

`EvolutionPriors` defaults:

| Quantity | Value / interpretation |
|---|---|
| Innovation family | lasso, normal or triple_gamma |
| Median absolute level coefficient | .01 log units per step |
| Median absolute slope coefficient | .00001 log units per step |
| Median absolute seasonal coefficient | .01 log units per step |
| Initial slope SD | .001 log units per observation step |
| Initial seasonal coefficient SD | .3 log units |
| Triple-gamma shapes | a=c=.5; global multiplier fixed at 1 |

These are explicit defaults, not scientific recommendations for every series.
For monthly data, a log-scale slope b corresponds to a ten-year multiplicative
change exp(120b). Changing sampling frequency requires recalibrating priors.
Signed FS coefficients give innovation variances s²; shrinkage is continuous,
so a small SD is not a posterior probability of an exactly fixed component.

## Computation

Location retains the established Gaussian FFBS or GEV Laplace–MH update.
The ancillary Gaussian evolution is non-centred: an anchored unit-noise path
and Gaussian coefficients conditional on shrinkage mixings. An elliptical
slice updates the complete path, another updates its coefficients, and the
existing normal/lasso/triple-gamma updates update the mixings. Sign moves
preserve the predictor. The likelihood callback includes the current scale,
shape, location and copula conditional, so no frozen-residual approximation is
introduced. ASIS, if requested, acts on location; it is not added to scale.

A prior Gaussian ellipse uses the full Gaussian structural covariance,
including deterministic state coordinates. There is no added variance floor
for static components. The same target is retained near GEV support boundaries;
invalid proposals have zero likelihood. Blocked exact updates do not guarantee
fast mixing of weakly identified location/scale decompositions. Inspect both
scale and location scientific quantities and perform recovery for new uses.

## Posterior access and forecasts

```python
fit.parameter_path("mu", channel="TXx", combine_chains=False)
fit.parameter_path("sigma", channel="TXx", link_scale=True)
fit.parameter_component_draws("sigma", "slope", channel="TXx")
fit.parameter_component_draws("sigma", "seasonal", channel="TXx")
future = fit.forecast(120, seed=17)
future.sigma_draws(channel="TXx")
```

Omit `channel` for a univariate fit. Location paths are returned in the original
orientation; lower extremes retain their reflected-maxima GEV shape convention.
Scale components are on log units. The returned scale `level` includes the
baseline log sigma, so level + seasonal equals log sigma_t.

Forecasts start from each draw's final scale state and sample every future
innovation using that draw's process SDs. Saved fits preserve the scale states,
FS coefficients and shrinkage variables, including warm starts. Forecast
uncertainty therefore includes scale evolution; future sigma is not held at its
last fitted value. `docs/examples/parameter_evolution.py` is a short runnable
Gaussian proof of concept with figures, diagnostics and a saved fit.

`simulate` supports structural scale through `scale.sd.level`,
`scale.sd.slope`, `scale.sd.seasonal`, optional `scale.initial.slope` and
`scale.initial.seasonal`. Append `.CHANNEL` for multiseries parameters.
The initial level remains anchored. `prior_predictive_targets` samples the
declared scale-evolution prior for univariate structural models.

## Extension points

- `parameters.py`: validate family parameter bindings, then lower to canonical declarations.
- `observation/scale.py`: serializable structural scale specification.
- `priors/evolution.py`: prior calibration in link units.
- `inference/fit/evolution.py`: Gaussian evolution against a likelihood callback.
- `inference/fit/marginal.py`: combine that callback with the current observation/copula conditional.
- `api/predict.py` and `core/fit.py`: propagate and expose parameter paths.

A new binding needs its parameter domain/link, exact likelihood evaluation,
serialization, posterior access and future propagation. Merely accepting a new
keyword is not a supported statistical extension.
