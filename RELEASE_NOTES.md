# BUCEX 1.8.2

This release adds physically interpretable hierarchical prior calibration,
shared initial-slope regularization and unrestricted normal GEV shape defaults.
The SERRA workflow now prioritizes a preliminary supervisor draft over a large
sensitivity campaign. See START_HERE.md for commands.

- `SharedShrinkage` now learns a separate common SD for the zero-centred initial
  slopes by default, alongside selected innovation prior scales. Individual
  slopes remain distinct. `initial_slope_sd=None` opts out; research JSON uses
  `pool_initial_slope: false`. Active slope coefficients only enter its exact
  log-scale slice update. Normalizing constants are retained.
- `SharedShrinkage.from_effects`, `calibration`, and `innovation_response_gains`
  translate between per-update priors and horizon changes. They distinguish
  physical-SD medians, signed-normal SDs and fully hyperprior-marginal RMS effects.
  Seasonal calibration uses the actual dummy transition.
- Reports include initial-slope prior/posterior comparisons, all four shared
  scales, physical effect tables, traces and shape-support diagnostics.
- `GEV()` and `fs_priors('gev')` default to unrestricted normal shape support.
  Standard FS convenience profiles now use Normal(0,.3²) shape priors; the
  explicitly historical `manuscript_gev_priors` retains its historical uniform.
  Finite/one-sided bounds remain supported; JSON represents unbounded endpoints
  with null. Uniform priors require finite endpoints. GEV observation support
  is enforced in every likelihood update, including the legacy FS kernel.
  General disturbance proposals use identity/exponential/logit coordinates
  for unrestricted/one-sided/two-sided supports with the correct Jacobians.
  Scalar GEV Laplace starts can shift an uncertain intercept into support
  deterministically, preserving transition constraints and the exact MH target.
- Fit schema 2.13.0 stores the new hierarchy. Older archives retain their old
  priors; missing initial-slope pooling is decoded as disabled, not added.
- `config/draft/` provides a six-series seasonal-copula candidate through
  August 2026, a complete parallel smoke check, historical validation, R=I and
  independent fallbacks, and separately declared supplementary comparisons.
  Main and validation posteriors are saved for later reporting. An additional
  39.7°C TXx risk export matches the reviewer's 2019 threshold question.
- Empty requested figure windows (such as a partial forecast year) produce a
  labelled placeholder in non-strict mode. Exact independence conditionals
  skip unnecessary score derivatives, avoiding zero-times-overflow artifacts.

The existing modular parameter-evolution API and separate univariate fitting
remain available. This release does not replace private trajectories by shared
states or introduce discrete model selection. It does not implement mid-chain
checkpoint/resume, so use tmux/nohup on biobot.

Scientific limits: four pooled scales do not guarantee better mixing, and
seasonal dependence cannot by itself repair marginal skewness, temporal
dependence or ordering violations. Unrestricted shapes permit nonfinite
observation moments; empirical averages are not proof of finite theoretical
moments. Quantiles and risk probabilities remain primary summaries. Short
execution tests validate software behavior, not temperature findings or
publication readiness. Verification details are in validation/release_1.8.2.json.
