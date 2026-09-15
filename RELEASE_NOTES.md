# BUCEX 1.6.3

This release supports the revised paper's private structural models and their
joint residual-dependence extension. No empirical paper findings are asserted.

- `SeasonalScale(period=12, prior_sd=.3)` adds identifiable, zero-sum seasonal
  log-scale effects to Gaussian and stationary-scale GEV observations. Each
  channel has its own effects; dated monthly forecasts use calendar months.
- `MarginalPriors` retains independent FS/exact-SSVS priors for private channels.
  A Gaussian copula now enters all conditional state, structural, scale, shape
  and correlation updates. This is full posterior feedback, not a copula fitted
  to fixed marginal residuals.
- Seasonal starts remain finite when a short record does not cover all phases.
- Private Gaussian states use conditional FFBS; GEV states use support-aware
  Laplace–MH. The structural proposal is anchored independently of the
  coefficients it updates, with the complete proposal correction.
- The default slope probabilities are `(0.10, 0.45, 0.45)` for absent, fixed and
  dynamic slopes. Explicit existing priors remain explicit. Probabilities must
  sum to one; `(0.1, 0.4, 0.4)` is not silently accepted or normalized.
- Seasonal scales are included in prediction, PIT, densities, endpoints,
  return levels, risks, simulation, archives and restarts. Scalar result names
  and component-summary methods remain compatible.
- Added residual normal-score dependence checks, practical-correlation
  probabilities and paired block-bootstrap score comparisons. SSVS transition
  counts exclude artificial transitions between independent chains.
- Clean SERRA runners cover univariate and copula analyses, matched identity
  baselines, prior/shape sensitivity, endpoint assessment, forecasts, replicated
  recovery, and held-out marginal, joint and compound-event scores.
- Primary paper configs end in 2022. The August 2026 data extension remains
  available under a separate config with explicit provenance qualifications.
- Removed duplicated obsolete tests, unreachable particle code, conference
  demonstrations, the redundant old daily CSV and an unused spreadsheet.
  Maintained API tests and historical archive fixtures remain.

## Correction affecting previous hierarchical fits

The older mixed/GEV **hierarchical structural-selection** update constructed
its Gaussian independence proposal at the current predictor without rebuilding
that proposal for the reverse move. It now uses a fixed observation anchor,
so the implemented MH ratio is valid. Refit previous results from that affected
hierarchical exact-SSVS route before scientific use. The existing univariate
GEV structural proposal already used the fixed anchor and was not affected.

## Scope

The new `MarginalPriors` route requires complete finite aligned observations,
private local linear trends with optional dummy seasonality, and FS/exact SSVS.
Shared components and hierarchical pooling retain their separate established
routes. Free factor loadings, a tail-dependent copula, hard ordering, residual
serial dependence, and combining seasonal scale with dynamic GEV `phi` are not
implemented. Existing univariate dynamic-`phi` models remain available.

Result archives use schema 2.10.0; historical readers are retained. Short workflow
checks and posterior-reference tests establish implementation evidence, not
full-record convergence, empirical coverage or journal acceptance.
