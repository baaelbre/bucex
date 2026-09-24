# 1.8.6 (2026-09-24)

- Add predeclared monthly and seasonal expanding-window validation runs for reviewer comment 5, covering seven distinct five-year windows and reporting central 90%, 95% and 99% intervals.
- Export the number of misses below and above each interval, directional 5%/1% quantile exceedances, and observed versus expected counts for fixed risk thresholds, with denominators by response, season and forecast year.
- Compute event-probability scores from the conditional predictive CDF integrated over posterior and future-state draws instead of rounded ensemble event frequencies. Save individual cases and incremental tables after every completed origin.
- Retain existing reference priors and old results; convergence and sparse 99% events still govern scientific interpretation.

# 1.8.5 (2026-09-23)

- Correct the seasonal reference initial-slope shared-scale median to 0.003 per season, giving a marginal prior SD of about 0.194°C per decade.
- Set seasonal level, slope and seasonal innovation medians to 0.01, 0.0001 and 0.01 per seasonal update; leave monthly reference priors intact.
- Add a resumable ten-setting seasonal level/slope prior screen with up to ten concurrent, isolated setting jobs, two-chain fits, held-out CRPS, per-response and joint scores, convergence reports and a source-data checksum.
- Keep unassessed or poorly mixed pilot rankings provisional. No new climate conclusions are included.

# 1.8.4 (2026-09-23)

- Focus the paper API on continuous private marginal FS fits, optional shared prior-scale shrinkage, and optional residual copulas.
- Separate `research/monthly` and `research/seasonal`; retain periodic-versus-constant scale adequacy tests.
- Remove public shared latent factors, hierarchical SSVS, dynamic GEV phi, and evolving observation scales.
- Correct copula-aware private multiseries forecast simulation and scoring.

# 1.8.3 (2026-09-23)

- Separate `research/seasonal` workflow: complete daily-derived seasonal means/extrema, four-phase scales and copula, shared shrinkage and independent fallbacks.
- Retain JJA 2026; exclude only January-February 1892 in research analyses. Audit 1,614 monthly / 538 seasonal blocks.
- Apply COMPSTAT marginal-moment prior calibration and match physical effects across update frequencies; moment-preserving hyperprior-width sensitivity.
- Analytic aggregate Gaussian/extreme CDFs and densities; matched historical forecasts scored on identical seasonal targets.
- Correct seasonal slope units, calendar grouping, yearly definitions, trace/risk/PIT reporting and prior-sensitivity summaries.
- Conditional-score posterior predictive tail/skewness/lag checks alongside existing dependence and forecast diagnostics.
- Raw r-largest ranks and runs-cluster diagnostics; r>1 likelihood remains a documented future extension.
- Resume completed comparison folds with configuration, data and version checks; no mid-chain checkpoint.

# 1.8.2 (2026-09-23)

- Shared initial-slope regularization with exact conditional updates and archive compatibility.
- Physical horizon calibration, conditional versus marginal SD accounting, and initial-slope reports.
- Unrestricted normal shape defaults with explicit optional bounds and enforced GEV support.
- Draft-first SERRA commands, seasonal-copula candidate, saved fits and separate appendix comparisons.

# Changelog

## 1.8.1 — 2026-09-22

- Pool seasonal innovation shrinkage alongside level and slope in the current
  SERRA specification, using three distinct shared hyperparameters. Preserve
  each response's initial seasonal pattern, process SDs and latent trajectory.
- Keep explicit level/slope-only pooling and fixed-prior univariate analyses.
  General `SharedShrinkage` declarations still select components explicitly;
  the joint sampler and archive schema are unchanged.
- Add matched seasonal-pooling and seasonal-anchor comparisons, residual
  independence/copula comparisons, structural adequacy and focused reviewer
  prior checks. Historical forecasts learn each hierarchy on training data.
- Handle omitted pooled components in sensitivity figures as `not pooled`,
  rather than failing or displaying a fictitious zero hyperparameter.
- Include model, observation and copula settings in dry-run study plans.
  Provide a complete command sequence and output interpretation in START_HERE.
- Extend seasonal hierarchy, parallel/archive, fixed-seasonality and workflow
  regression coverage. Software verification is separate from publication
  convergence and predictive adequacy on the full record.

## 1.8.0 — 2026-09-22

- Add `SharedShrinkage` to `MarginalPriors` for uncertain common FS normal-prior
  scales, retaining private innovation SDs, paths and the univariate API.
- Update shared scales inside the exact joint private FS/copula sampler with
  normalized log-scale conditionals; preserve archive, restart and parallel
  chain semantics. Fixed-zero innovations do not count as hierarchy members.
- Export shared-scale traces, R-hat/ESS, hyperprior/posterior intervals and
  unconditional individual prior comparisons. Include monthly scale contrasts
  and initial seasonal vectors in scalar diagnostics; include monthly scale
  effects in traces.
- Extend compact sensitivity reports to joint fits, joint predictive scores,
  compound events and shared-hyperparameter sensitivity without double-counting
  global tables or mixing response labels.
- Add explicit calendar forecast origins and event counts. The new SERRA pilot
  uses all six responses through August 2026 and includes a 2016–2020 held-out
  block to cover the 2019 temperature record.
- Provide matched fixed-half, fixed-quarter, pooled-quarter and pooled-half
  specifications; preserve the six-univariate and fixed-copula fallback.
- Keep existing fixed normal-prior defaults and earlier configurations intact.
  The focused hierarchy workflow has its own explicit anchors and makes no
  automatic model choice or scientific convergence claim.

## 1.7.4 — 2026-09-21

- Integrate exploratory manuscript Figures 1 and 2 into `research.monthly.explore`.
- Add the general `explore_monthly` / `MonthlyExploration` API for empirical
  seasonal cycles and within-month, within-era detrended interquartile ranges.
- Keep numerical calculations, scoped plotting and report persistence in
  separate package modules; research code only loads data and selects settings.
- Declare date windows, colours, panel layout and PNG/PDF formats in JSON.
- Export source observations, sample/missing counts, plotted CSVs and metadata.
  Reject duplicate months, unavailable windows and insufficient samples.
- Preserve original temperatures, prior defaults, parallel-chain execution,
  inference kernels and existing fitted-model figure commands.

## 1.7.3 — 2026-09-21

- Add `MCMC(chain_workers=...)` through one process executor used by all five
  inference backends. Preserve seed streams, chain order, diagnostics, warm
  starts and archives; cap numerical thread pools to prevent oversubscription.
- Add a focused normal-prior assessment driver with one candidate list for
  posterior sensitivity and historical forecasts, using JSON-only settings.
- Export slope/risk/level paths and compact traces independently of large fit
  storage; record execution settings in reports and held-out fits.
- Add `innovation_prior_diagnostics`, `compare_predictive_scores` and
  `SensitivityReport`, with manuscript-style comparison figures and strict
  forecast-case matching. Small numbers of origins receive descriptive
  comparisons, not misleadingly precise bootstrap intervals.
- Save historical predictive bands, held-out observations, horizons, calendar
  coverage/PIT diagnostics and convergence by forecast origin.
- Keep scientific priors and transition kernels unchanged. New pilot settings
  use four parallel chains and 500+500 iterations as an explicit screening
  budget; no simulation study or automatic prior selection is run.

## 1.7.2 — 2026-09-20

- Add a scoped manuscript plotting style and multi-format figure saving.
- Add `ReportCollection` and recipe-based publication figures from compact CSV
  reports; reject ambiguous fits, mislabelled intervals and risk thresholds.
- Export unthinned chain tables, including initial level/slope, physical process
  SDs, observation scale/shape and scientific-target traces.
- Add calendar-month PIT/normal-score diagnostics and held-out month-specific
  coverage tables, preserving counts and PIT boundary observations.
- Make endpoint intervals and forecast-width plots follow `credible_interval`.
- Honor `save_fits: false` in ordinary research reports; include scale parameters
  in convergence screening; preserve a saved fit's channel name in forecast checks.
- Add matched revision configurations and the manuscript figure driver. Copula
  CLI accepts `--scale`; model comparisons accept `--series`.
- Keep the 1.7.1 inference kernels, archive schema and normal-prior defaults.

## 1.7.1 — 2026-09-17

- Correct inefficient GEV continuous-FS coefficient preconditioning: optimize a
  deterministic conditional-mode Gaussian reference, then retain the exact
  likelihood/reference elliptical-slice correction. The reference never starts
  from the current coefficient vector. Conditional copula curvature is included.
- Keep this proposal construction in its own inference module, used by the
  existing univariate and joint continuous kernels. Gaussian exact draws remain
  unchanged; no ASIS or model-selection layer is added.
- Record optimizer convergence, iterations, fallback, support repair and
  covariance regularization beside coefficient slice cost in saved diagnostics.
- Default to normal FS innovations and a .01 level-SD prior median; retain
  .00005 slope and .02 seasonal medians. Existing explicit priors are preserved.
- Update the SERRA configurations, add level-only prior sensitivity and an
  old-level-prior comparison, and document the reviewer/paper workflow.
- Add independent numerical integration, copula-conditional, endpoint-support
  and long-record coefficient-mixing regression tests.

## 1.7.0

Named parameter declarations, structural log-scale evolution with continuous FS
shrinkage, reusable exact-likelihood evolution updates, parameter path APIs,
simulation/forecast/archive support, and SERRA fixed-scale workflows through
August 2026. Added climate-period estimands, actual joint prior sensitivity,
preflight and a revised run guide. See RELEASE_NOTES.md for scope and validation.

## 1.6.5

- Full SERRA runs follow the latest bundled month (August 2026); named historical configurations preserve 1892–2022. Actual fitted dates are printed and saved.
- Added draw-wise calendar averages/extremes, leap-year day weights, DJF ending-year labels, explicit incomplete windows and analytic aggregate risks.
- PNG reports now include slopes, physical parameter/scale/target traces, PIT/Q-Q/residual diagnostics, month panels, annual/seasonal forecasts and risk curves.
- Saved-fit report regeneration and fixed-origin checks against later observations are available as short research scripts.
- Inference kernels and prior defaults are unchanged from 1.6.4. A report cannot update the posterior to new data.
- Validation: 276 source tests and 21 installed-package checks passed; full-date TXm and mixed copula execution checks completed.

## 1.6.4

Continuous private FS inference, median-matched priors, secular/seasonal observation scale, pooled seasonal copulas, exact bivariate risk quadrature and revised SERRA workflows. See RELEASE_NOTES.md and validation/RELEASE_VALIDATION.md.


## 1.6.3

Added private FS/exact-SSVS inference with joint Gaussian residual copula feedback,
seasonal observation scales, paired predictive comparisons, and concise SERRA
workflows. Preserved scalar APIs and historical archive readers. Corrected the
older mixed hierarchical structural-MH proposal anchor; affected fits need rerunning.
See `RELEASE_NOTES.md` for the full scope and `docs/VALIDATION.md` for evidence.

## 1.6.2

Updated all six bundled monthly Uccle series through August 2026 (1,616
months). Extended the existing daily loader and aggregator to accept explicit
sources, use all complete months by default, and optionally export summaries
with a quality report. Daily validation now requires coverage of every compared
month. Full-record configurations use the latest bundled data. One short
preparation script delegates to the package; duplicate dated configurations and
summary copies are removed. See `RELEASE_NOTES.md` for the API and migration.
Independent GEV research runs use a zero-shape initial value when permitted by
the configured bounds, avoiding an invalid finite starting endpoint.

## 1.6.1

Consolidated SERRA-only research workflows; optional Gaussian residual copula
with exact likelihood correction; original-scale ordering diagnostics; six
independently runnable univariate analyses; configured reviewer experiments;
90/95/99% held-out calibration and memory-bounded sequential refits. See
`RELEASE_NOTES.md` and `docs/VALIDATION.md` for scope and evidence.

## 1.6.0

Introduced shared components with fixed loadings, weighted zero-sum departures,
centered joint inference, Gaussian reference validation, component/risk/forecast
results and shared-model persistence. Retired particle inference from the
active API. Earlier release archives retain their own detailed history.
