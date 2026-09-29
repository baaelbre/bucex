# 1.9.6.1

The half-normal reference now has slope scale 0.0002. A 3×4 level/slope grid includes
slope 0.001; broader seasonal checks reach 0.10 and include a fixed cycle. Observation
scale and shape checks remain. Heavy-tail hyperpriors and leave-one-out experiments
are absent from the active matrix. Six matched private HN fits remain for the pooling
comparison; broader private checks are deferred.

Initial levels have SD 10; initial seasonal contrast coordinates have SD 10. The FS
sampler retains lag coordinates but now supports their equivalent full covariance.
This change affects coefficient updates, prior draws and archive round trips. A bridge
fit preserves the old SD-20 independent lag initialization at the new slope setting.

All four central-level slope settings have 11 expanding-window origins, 1970–2020,
and up to 140 held-out seasons, truncated only at the end of available observations.
Reports retain horizon bands and unique verifying-date counts. No future observations
are used in training or learning the shared scales.

Forecast simulation for independent-residual multiseries models is vectorized in
batches. Predictive quantiles can invert the mixture CDF to integrate observation
noise, while process and posterior uncertainty are still simulated. Seasonal full-fit
forecasts use 50,000 paths over 120 steps. Annual return-level calculations use a
recorded subset of whole joint paths. The public empirical-quantile default remains
available; this study explicitly selects CDF inversion.

A BIOBOT screen queue defaults to 48 chain workers with memory-aware scheduling;
HPC uses native Slurm arrays and an automatic dependent report collection job.
100 fits map to 95 HPC array elements. Both launch paths include startup checks,
prior simulation, provenance checks and completion/diagnostic manifests.

This is a computational release, not a claim that the new scientific runs converge
or confirm prior robustness. The manuscript's results must be updated from those runs.

---

## 1.9.5.1

- Add full joint prior simulations and scientific figures, with no posterior conditioning.
- Check FS seasonal ordering, minimum reflection, horizon variances and joint pooling.
- Integrate prior checks into screen/paper runners and compact exports.
- Revise the manuscript on calibrated evolution versus broad initialization.
- Preserve all 1.9.5 model and posterior sampling settings.

# 1.9.5

Pooled half-normal hierarchical shrinkage, calibrated half-t/Cauchy sensitivity,
106 fits per tier and a single resource-bounded BIOBOT queue. Adds explicit
annual risk/return-level reports and full pre-2019 evaluation. See
`RELEASE_NOTES.md` and `BUCEX-1.9.5-commands.md`.

# 1.9.4 — matched shrinkage comparison

Direct Normal-SD hierarchies; matched independent/shared mixtures; separate initial rates; 49 screen experiments; longer chains; mixed CPU routing; pooled/separate collection and influence checks. See RELEASE_NOTES.md.

# 1.9.3 correction — manuscript prior SDs (paper_sd_20260928)

- Match the latest manuscript directly: Normal innovation SDs (0.01, 0.0001, 0.01), initial-slope variance 0.0001 and hence SD 0.01. Remove the earlier median-to-SD conversion.
- Update reference verification, paper gates, sensitivity anchors and calibrated prior-effect checks. The 23-setting grid and HPC resources are unchanged.
- Use a fresh `serra_193_paperpriors` results tree. Existing submitted jobs must be stopped before replacing their source/configuration files. Earlier results remain a wider-prior experiment.

# 1.9.3 — fixed Normal shrinkage and parallel HPC experiments

- Remove shrinkage hyperpriors from the active seasonal grid; use median-absolute calibration (0.01, 0.0001, 0.01, 0.01), including the initial slope.
- Add direct `innovation_sd` inputs, four-component SD multipliers and fixed-prior calibration reports; retain coefficient prior/posterior comparisons.
- Replace learned-width/matched-moment controls with global half/double SD controls; preserve stronger seasonal shrinkage, slope combinations, shape priors and model alternatives.
- Run six separate responses per experiment with two parallel chains each, reserving twelve cores for both screen and paper. Longer paper chains and stricter diagnostics remain.
- Submit using only the login host's standard library. Add Gallade compute-node environment creation, startup probe dependencies, per-response failure tracking and compute-node collection.
- Provide 23 full-record settings and 88 experiment/origin bundles across all batches (508 response fits). Retain 30-year forecasts and both validation designs.
- Freeze 1.9.2 configuration snapshots for historical tests; no new empirical performance claim is made.

# 1.9.2 — separate-response seasonal workflow

- Add single-response `IndependentShrinkage`, preserving the normal–lognormal marginal innovation prior while removing cross-response borrowing.
- Split every sensitivity and validation fit into one task per applicable response; keep initial rates separately Normal regularized.
- Add quarter-seasonal and double-slope × half/quarter-seasonal controls; omit redundant mean/shape fits.
- Make the root launchers local to biobot; add bounded job concurrency, per-task logs, provenance-checked resume, preserved failed attempts and compact export.
- Extend saved-fit, report and validation paths to private hyperparameters, 30-year forecasts and per-response score comparisons.
- Freeze the previous joint reference for historical tests and tools. No new empirical conclusions are claimed by this software release.

# BUCEX 1.9.1 (2026-09-28)

- HPC startup correction: resolve the Python executable's parent directory to
  its physical path before export, preserving the virtual-environment symlink.
  Report the requested interpreter and host on failure; check Python >= 3.10
  before loading the package. No scientific settings or Python model code change.
- Seasonal workflow: identity residual dependence in every active task; separate
  fixed initial-rate priors, baseline SD 0.5 C/decade and sensitivity .25 / 1.0.
- Retain three shared innovation-scale hyperparameters. Add fixed-anchor width
  controls alongside matched-moment controls; 23 full-record fits in total.
- Separate HPC posterior and biobot validation wrappers. Native Slurm preference,
  physical paths, explicit environment propagation and startup diagnostics.
- 52 five-year validation tasks, with 2010 added to sensitivity comparisons;
  optional original 60/80/90% ten-year checks in a separate three-task batch.
- Export forecast-origin levels alongside rates; suppress main-workflow
  cross-summary contrasts and compound-risk figures. Shared-scale figure
  adapts to the three active hyperparameters.
- Fix calendar phase labels in identity/constant residual dependence checks.
- Fix prior-effect simulation for separate initial rates: the random initial
  coefficient is multiplied by the horizon without an extra Gaussian shock.
- Keep exact likelihood/MH correction, saved fits and 30-year forecasts.

# BUCEX 1.9.0 (2026-09-27)

- Adopt the tested log(3) seasonal hyperprior width with all four anchors
  matched to the old marginal coefficient second moments. Freeze the original
  1.8.9 reference as `research/seasonal/config/reference_189.json`.
- Integrate 25 unique posterior fits: reference plus width/anchor, shape,
  seasonal scale, initial seasonality and copula checks. Identity dependence
  retains the same shared hierarchy. Static seasonality uses the new width.
- Provide separate screen and paper launchers. Both save fits, use 95% bands
  and forecast 30 years. Give the paper reference its longer chain budget.
  Keep pre-2019 and validation batches separate from default submissions.
- Export posterior innovation contributions at 10 and 30 years and
  probabilities of effects below/above .05, .10 and .20°C.
- Fix the displayed initial-slope hyperparameter's median-to-SD conversion;
  preserve the legacy SD-anchored API. Align serial/copula predictive envelopes
  with the declared 95% level. The likelihood/sampling target is unchanged.
- Add source/config/data provenance checks, batch-aware collection, paired
  validation origins and launcher dry runs. Avoid silently overwriting failed
  attempts or pooling incompatible completed fits.

See [FINAL_RUN.md](FINAL_RUN.md) for exact settings and commands, and
[release verification](validation/RELEASE_VALIDATION_190.md) for software checks.
The release does not contain completed production MCMC or validation results.

# 1.8.8 (2026-09-26)

- Use manuscript seasonal median-absolute anchors `(0.01, 0.0001, 0.01, 0.001)` with a matching initial-slope hierarchy; preserve historical SD-anchored fits.
- Add complete paper command queue, 30-year prior-effect simulation, time-resolved additive-gap contrasts, physical anchor sensitivity, and supporting-run convergence gate.
- Align main exploratory, scale, and smoothed-PIT figures with the selected manuscript layouts.

# 1.8.7 (2026-09-24)

- Lock the seasonal manuscript reference to the resolved scientific settings,
  thresholds, contrasts and seeds of
  `uccle_copula_20260923T222238_406159Z.zip`; retain the 1.8.6.1
  dummy-seasonal initialization correction.
- Increase only the final MCMC budget to 3,000 warm-up and 8,000 retained
  iterations per chain. Priors are not tuned to the earlier run's diagnostics.
- Add prospective JJA-2019 record-event configurations and export forecast
  probabilities for every declared additional threshold.
- Add a convergence-gated manuscript-figure builder, focused prior/adequacy
  configurations, and a single final-run command guide.
- Repair a nonzero user-supplied initial GEV shape by enlarging only the
  starting scale when needed to enter finite support; the prior and retained
  likelihood are unchanged.

# 1.8.6.1 (2026-09-24)

- Fix dummy-seasonal initial states: map chronological phase effects to the lag-ordered coordinates used by the state transition in single-series and joint fits. Match the time-one convention for the initial level.
- Reproduce the earlier 2015 TXn shape/scale failure without the copula and show that the corrected initializer removes its drifting low-likelihood chain in a focused two-chain test. The posterior likelihood and priors are unchanged.
- Add reference mixing diagnostics with per-chain traces and scientific targets. Gate new seasonal prior-grid runs on passing four-chain reference checks at both origins; use a fresh output directory to avoid old fits.

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
# 1.8.9 (2026-09-27)

- Seasonal manuscript priors: initial seasonal coefficient SD 20°C and shared
  initial-slope median absolute coefficient 0.01 per season; the prior-rate SD
  is approximately 0.96°C per decade after integrating the shared scale.
- Full reference: 30-year forecast, 12,000 predictive draws, 95% bands, and
  quantile-based uncertainty widths for each of the six responses.
- Split sensitivity, seven validation origins and matched monthly/seasonal
  forecast comparisons into 106 independent PBS array fits; retain separate
  full reference, pre-2019, dependence and monthly supplement jobs.
- Add a convergence-aware collector, mid-century pairwise gap contrasts and
  consolidated manuscript forecast uncertainty panels.
