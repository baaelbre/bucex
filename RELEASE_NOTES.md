# BUCEX 1.7.0 — explicit parameter evolution and the SERRA protocol

## General API

- `Model` and `Channel` accept named `parameters` declarations using `Constant`
  and `Latent`. Existing `components=` and observation-scale APIs remain valid.
- `Constant()` means estimated but time-constant; it does not fix a value.
- Location and log observation scale can have separate level, slope and dummy
  seasonal states. `EvolutionPriors` supplies separately calibrated continuous
  normal/lasso/triple-gamma shrinkage for log-scale innovations and regularizes
  its initial slope/seasonal coefficients. The scale initial level is anchored.
- A reusable Gaussian-evolution kernel takes a likelihood callback. Scale path
  and coefficient slices use the exact observation/copula conditional; the
  location sampler remains Gaussian FFBS or GEV Laplace–MH.
- Added fitted/forecast `parameter_path`, fitted `parameter_component_draws`,
  parameter path plots, structural-scale simulation and prior-predictive targets.
  Saved fits preserve ancillary states/mixings; forecasts propagate future
  innovations and support warm starts.
- GEV shape remains constant. New observation families, scale regressions,
  shared scale states and shared-location plus structural-scale combinations
  are not implemented; unsupported declarations are rejected explicitly.

## SERRA research

- The primary data window is **January 1892–August 2026** (1616 months), as
  requested. The 1892–2022 configurations remain historical comparisons.
- Primary models use estimated constant scale/shape, continuous lasso FS
  shrinkage, LKJ(1) residual dependence and pure NCP (ASIS off).
- Reports add paired 30-year period contrasts, period-average slopes,
  month-specific location/risk changes, posterior sign probabilities and
  numerical screening. The recent window is September 1996–August 2026;
  historical runs retain 1993–2022. Endpoint summaries remain secondary.
- Prior sensitivity supports both six separate fits and actual joint copula
  refits. Five core prior comparisons, targeted nuisance checks and the small
  fixed/evolving seasonality × constant/monthly scale supplement are separate.
- Matched R=I/copula validation, constant-scale joint recovery, endpoint and
  full generating-shape-grid experiments are aligned. Long validation/recovery
  jobs checkpoint numerical results after each completed case.
- `preflight` prints resolved dates/models/priors and a memory estimate.
  `START_HERE.md` and the research guide give an explicit run order. The general
  evolving-scale example is outside the paper's research directory.

The full joint centered state array alone is approximately 8.07 GB at the
configured budget. Numerical execution checks do not establish convergence,
coverage or prior robustness of production runs. See
`validation/RELEASE_VALIDATION.md` for checks actually performed. No preliminary
outputs are bundled as manuscript results.
