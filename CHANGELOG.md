# Changelog

## 1.3.1

- Added six complete schema-1 simulation configurations, one for each
  structural truth, under `examples/config/simulations/`.
- Converted the supplied successful run manifests into clean input files and
  retained their scenario-specific simulation, prior, and MCMC settings.
- Removed the example configuration-wrapper module. Every numbered example now
  exposes `DEFAULT_CONFIG_FILE`, accepts `--config PATH`, and calls
  `bx.load_config` directly.
- Documented the six local one-by-one commands and added regression checks for
  JSON completeness, scenario-key order, and compatibility with every matched
  simulation engine.
- Kept the all-scenario configuration, state equations, inference targets,
  public API, and archive schema unchanged.

## 1.3.0

- Replaced duplicated example settings with authoritative JSON files shared
  across the simulation, Uccle, and centered inverse-gamma analyses.
- Standardized the effective simulation and Uccle priors across Laplace,
  PGAS, and Laplace-MH scripts without hidden environment overrides.
- Added `--config PATH` selection with resolved
  settings-file provenance in every fitting run manifest.
- Removed the Laplace-sensitivity reducer, shell grid, PBS jobs, and dedicated
  documentation, plus obsolete launchers for the former example-07 name.
- Made titles optional and off by default, adopted mathematical legend labels,
  simplified structural-selection and process-SD plots, and added paired level,
  decade-slope, seasonal, posterior-predictive, and forecast figures.
- Replaced positional/environment HPC controls with one JSON path and a generic
  parallel-chain fan-out/combine runner.
- Added configuration, cleanup, and plot-title regression coverage while
  keeping the public inference API and archive schema 2.6.2 unchanged.

## 1.2.0

- Standardized the simulation truths across examples 02, 03, 04, and 08 and
  the fitted model, calibrated SSVS prior, MCMC defaults, predictive settings,
  and output names across examples 03, 04, and 08.
- Standardized the Uccle model, prior, MCMC defaults, forecast settings, and
  output names across examples 05, 06, and 09.
- Kept every scientific setting explicitly defined in each script and expanded
  parity tests to compare all approximate and exact engines.
- Added one-based `phase=` filtering to fitted predictor and latent-level plots.
- Added `phase=`, `target="level"`, and `component_draws()` to forecasts and
  forecast summaries, including phase-matched history plotting.
- Added phase-specific and seasonally adjusted forecast tables/figures to all
  simulation and Uccle fitting examples.
- Renamed the centered inverse-gamma example and its launchers to
  `07_centered_ig` and standardized its output/log stem.
- Raised the numbered fit-script, Bash-runner, and PBS defaults to 1,000 warmup
  plus 1,000 retained iterations per chain.
- Kept the exact inference kernels and archive schema 2.6.2 unchanged.

## 1.1.4

- Added a PBS fan-out/fan-in workflow for example 08. The default six
  scenarios and four chains are mapped bijectively onto 24 independent
  one-core array tasks.
- Added a single submission helper,
  `bash_scripts/qsub_08_simulation_laplace_mh.sh`, which submits the fit array
  and a dependent `afterok` finalizer and prints both job IDs and the shared
  output directory.
- Changed the final HPC simulation profile to 1,000 warm-up iterations and
  1,000 retained draws per chain, with six-hour fit-task and one-hour finalizer
  resource requests. The ordinary standalone example retains its local
  defaults and remains backward compatible.
- Added collision-free shared output: each worker writes one
  `tasks/chainXX/fits/<scenario>/combined.bucex` and one task manifest; the
  finalizer alone writes combined fits, simulations, tables, and figures.
- Added `BUCEX_SCENARIO_KEYS` to select a reproducible subset without editing
  the scientific example. Scenario-specific seed offsets remain those of the
  full six-scenario design.
- Refactored example 08 into reusable simulation, prior, and one-scenario fit
  functions. The internal array worker imports these definitions rather than
  maintaining a second model or prior specification.
- Added task-map, shared-layout, resource-default, real worker, and real
  multi-chain finalization regression coverage.
- Kept the public inference API, Laplace-MH kernel, and archive schema 2.6.2
  unchanged.

## 1.1.3

- Repaired exact FS Laplace-MH proposal initialization for finite-endpoint GEV
  models. When the zero non-centred trajectory violates GEV support, bucex now
  constructs a deterministic support-feasible trajectory without using the
  current chain state.
- Added a one-step-ahead repair for integrated stochastic slopes, preserving
  their deterministic accumulation coordinate and singular transition law.
- Preserved the independence-MH target: the repaired proposal remains a fixed
  function of observations and static parameters, and endpoint-invalid draws
  from that proposal remain ordinary MH rejections.
- Exact Laplace-MH and PGAS iterations now fail immediately on unrecoverable
  numerical errors. They never retry an identical deterministic calculation or
  save a restored state as a new posterior draw.
- Added `initial_support_repair_rate` to Laplace diagnostics and retained the
  per-draw repair indicator in result archives.
- Added regression coverage for a negative-shape GEV whose zero FS path is
  invalid while a full integrated-slope trajectory is valid, plus a fail-fast
  test proving that exact samplers do not emit duplicate draws.
- Kept the public modelling API and archive schema 2.6.2 unchanged.

## 1.1.2

- Rebuilt `08_simulation_laplace_mh.py` as the exact-engine counterpart of
  `03_simulation_laplace.py`: the same six truths, observations, model,
  calibrated priors, MCMC defaults, posterior summaries, and PDF/PNG figures.
- Rebuilt `09_uccle_laplace_mh.py` as the exact-engine counterpart of
  `05_uccle_laplace.py`, including the median-centred level prior, calibrated
  slope/innovation slabs, fixed-versus-dynamic model odds, predictive checks,
  forecasts, endpoint plots, and complete tables.
- Standardized simulation outputs under `simulations/`, `fits/`, `tables/`,
  and `figures/`, and Uccle outputs under `fits/`, `tables/`, and `figures/`.
- Added aggregate Laplace-MH acceptance/support diagnostics while retaining
  per-series and per-scenario algorithm tables.
- Replaced the abbreviated Laplace-MH HPC launchers with independent-chain
  runners that match the Laplace runner contracts and combine results only
  after all chains succeed.
- Removed temporary output-directory and random-walk-only overrides from
  `03_simulation_laplace.py`.
- Added regression tests that enforce scientific-setting, artifact-layout,
  and runner-default parity between examples 03/08 and 05/09.
- Kept inference internals and archive schema 2.6.2 unchanged.

## 1.1.1

- Added automatic conjugate inverse-gamma Gibbs updates for Gaussian process
  variances under the centered parameterization.
- Kept inverse-gamma GEV observation-scale updates and non-centred process-scale
  updates on their nonconjugate MH paths.
- Added explicit parameter-update-method diagnostics and removed Gibbs/fixed
  updates from MH acceptance summaries.
- Changed `07_centered_ig_random_walk_gev.py` to use approximate Laplace by
  default, with `BUCEX_ENGINE=laplace_mh` and `BUCEX_ENGINE=pgas` as exact
  validation options.
- Updated the example runner, PBS job, validation, documentation, and release
  metadata. Laplace-MH performance internals are unchanged.
- Kept archive schema 2.6.2 for backward compatibility.

## 1.1.0

- Added `engine="laplace_mh"` for exact-invariant whole-trajectory state
  updates in univariate GEV models using FS, centered, or disturbance
  parameterizations.
- Refactored Laplace inference into deterministic proposal construction,
  Gaussian smoother draws, and reusable exact-over-Gaussian correction weights.
- Kept the exact and proposal state laws identical so singular transition
  measures cancel from the MH ratio; projected numerical draws onto integrated
  slope and dummy-seasonal affine recursions without adding process jitter.
- Added `Laplace.mh_steps`, per-draw state acceptance/support diagnostics,
  per-chain acceptance summaries, exactness metadata, and proposal metadata in
  `InferencePlan`.
- Routed `laplace_mh` through exact GEV structural-parameter and SSVS model
  updates rather than the approximate pseudo-observation regression update.
- Added structural-simulation and Uccle Laplace-MH examples with matching Bash
  and PBS launchers.
- Added identity, singular-support, public API, SSVS, persistence, and
  negative-shape GEV regression coverage.
- Kept archive schema 2.6.2 for backward compatibility.

## 1.0.1

- Added posterior predictive replication and out-of-sample forecasting to the
  public `FitResult` API, with common plotting support for observations,
  recent history, predictive intervals, and latent-predictor summaries.
- Added observation-free level figures alongside the seasonally adjusted
  versions saved by every fitting example.
- Added conditional slope plotting in degrees per decade, including an
  optional dashed fixed-slope posterior and its uncertainty beside the
  stochastic-slope posterior.
- Changed seasonal figures to show the seasonal component alone and simplified
  the season, slope, structural-selection, and process-SD figure labels.
- Recalibrated the structural and Uccle example priors to distinguish slowly
  varying trends from level innovations more clearly.
- Kept the seven core examples as direct, sequential uses of the public API
  and added predictive tables and figures to all four component-selection
  fitting scripts.
- Added an eighth diagnostic example: an exact-PGAS random-walk GEV with
  centered states, inverse-gamma priors on both variances, independent-chain
  HPC execution, and trace/ACF/ESS/R-hat/particle diagnostics.
- Kept archive schema 2.6.2 for backward compatibility.

## 1.0.0

- Established the clean `bucex` repository as the stable continuation of
  former version 2.6.2.
- Added dedicated public level and slope plots and saved both from every
  Laplace and PGAS fitting example.
- Moved normal runners to `bash_scripts/` and PBS submissions to
  `job_scripts/`, retaining one pair per numbered analysis.
- Added parallel-chain orchestration, fit combination, virtual-environment
  validation, complete submission documentation, and a PBS seed/slab
  sensitivity grid.
- Retained compact signed result directories, complete run configurations,
  theoretical GEV tail/scale figures, 1892 Uccle coverage, LOESS smoothing,
  and phase-specific seasonal trajectories.
- Added clean-repository ignore rules for generated fits, PDFs, results, logs,
  caches, environments, and build products.
- Kept archive schema 2.6.2 for backward compatibility.

## 2.6.2

- Replaced the presentation workflow layer with seven standalone, direct-API
  examples under `examples/` and seven matching PBS jobs.
- Added optional timestamp-indexed result directories shared through
  `BUCEX_RUN_ID`.
- Added robust NumPy-only `loess_smooth()` and phase-specific
  `fit.plot("season")` trajectories.
- Put matched shape and scale simulations on common comparison ranges and
  retained separate time-series figures.
- Removed `bucex.workflows`, scenario factories, result workflow helpers, and
  the `bucex-presentation` console script.
- Advanced safe archives to schema 2.6.2 with backward readers.

## 2.6.1

### Configurable simulations

- Added period-aware scenario objects and public tail, scale, and structural
  scenario factories.
- Replaced the seven structural truths with the requested six-design sequence:
  stationary, linear trend, random walk, local linear trend, stationary plus
  changing seasonality, and local linear trend plus fixed seasonality.
- Changed defaults to period 4, 800 structural blocks, and smaller process and
  seasonal amplitudes.

### Examples and execution

- Exposed simulation controls and SSVS hyperparameters at the top of the
  standalone presentation scripts.
- Added stale-artifact checks for changed simulation, prior, period, and MCMC
  controls.
- Synchronized the workflow CLI and PBS scenario arrays with the new design.
- Advanced safe archives to schema 2.6.1 with backward readers.

## 2.6.0

### Presentation workflow

- Replaced the broad 2.5 presentation workflow with the requested Uccle TXx
  opening, GEV-tail illustrations, controlled structural simulations,
  Laplace-versus-PGAS comparison, and TXx/TXn/TNx/TNn analyses.
- Added seven fixed-truth SSVS recovery scenarios with common negative shape
  and scale.
- Strengthened the simulated structural innovations, widened the GEV shape
  contrast to `-0.30/0/+0.30`, and added a matched `sigma=0.75/1.50/3.00`
  experiment.
- Replaced simulation grids with one figure per time series and one separate
  three-panel decomposition per structural scenario.
- Standardized result tables, figures, paths, manifests, runtime profiles, and
  CLI behavior across simulated and observed fits.
- Replaced the presentation examples with exactly seven standalone direct-API
  scripts; retained the complete PBS dependency graph for batch execution.

### Inference and diagnostics

- Added true univariate `FitResult` warm starts, including the complete centred
  latent path and signed FS coefficients.
- Added Laplace-to-PGAS warm-start provenance to saved results.
- Added reference-ancestor change diagnostics for standard PGAS with singular
  or deterministic transition directions.

### Release

- Restored the complete top-level package API.
- Advanced safe archives to schema 2.6.0 with backward readers.
- Added v2.6 workflow tests and validation scripts.

## 2.5.0

### Scientific workflow

- Made componentwise fixed/dynamic level, zero/fixed/dynamic slope, and
  fixed/dynamic seasonality the primary univariate and hierarchical SSVS story.
- Changed hierarchical defaults to pooled selection; retained joint trend
  classes and shared slab magnitudes as explicit sensitivity models.
- Added the staged TXx-to-six-series presentation workflow with runtime
  profiles, deterministic paths, manifests, tidy exports, and report-only mode.
- Added controlled TXx structural analogues and leave-future-out model
  comparison using proper scores and PIT diagnostics.

### Execution and release

- Added `bucex-presentation` and the reusable `bucex.workflows` API.
- Added full PBS/Torque arrays for benchmark, validation, univariate,
  hierarchical, sensitivity, combination, and reporting stages.
- Fixed chain combination for deserialized dataclass priors containing NumPy
  arrays, which previously made some SSVS HPC archives incomparable in Python.
- Advanced checksummed archives to schema 2.5.0 with backward readers.
- Added end-to-end workflow validation and package/release build checks.

## 2.4.1

### Hierarchical model space

- Added the default four-class joint level/slope innovation space: linear
  trend, RW1 with drift, RW2 smooth trend, and full local linear trend.
- Kept initial slope estimated in every class and retained componentwise
  no-slope SSVS only as an explicit legacy sensitivity.
- Added posterior summaries and plotting for shared trend-class probabilities.
- Added expert-scale calibration and implication helpers for normal/half-t
  hierarchical slabs.

### Mixed/GEV inference

- Added exploratory hierarchical Laplace SSVS for mixed and all-GEV models.
- Added validated `FitResult.warm_start()` and `init=<laplace fit>` support for
  exact PGAS.
- Added `HierarchicalSampler` controls for Laplace initialization and optional
  concurrent channel updates.
- Vectorized GEV particle weights, complete-path likelihoods, and Laplace
  pseudo-data calculations.
- Grouped channel-specific GEV shape values in progress output.
- Renamed changed fraction to the scientifically explicit path-update fraction,
  retaining a deprecated compatibility alias.

### Workflows and persistence

- Added a Laplace-to-PGAS example, four-process HPC chain example, Slurm array,
  and chain-combination example.
- Updated the checksummed archive schema to 2.4.1 while retaining readers for
  prior supported schemas.

## 2.4.0

### Uccle workflow hotfix

- Aligned `ssvs_gaussian_priors()` and `ssvs_gev_priors()` with Example 09 by
  accepting direct level, trend, season, and slab SSVS settings as well as an
  explicit `SSVSPrior` object.
- Made one-series Uccle selections robust to a bare string, so `"TXm"` cannot
  be accidentally interpreted as the three names `"T"`, `"X"`, and `"m"`.
- Made a requested single-series CSV authoritative without requiring the
  explicit directory to contain all six Uccle summaries.
- Restored the zero-restoration metadata contract for successful
  centered/disturbance PGAS fits.

### Model surface

- Unified univariate `Model` and hierarchical `MultiSeriesModel` under
  `bx.fit()` and `FitResult`.
- Removed obsolete multichannel implementations, compatibility aliases,
  specialised samplers, result methods, plots, examples, tests, and documents.
- Added `HierarchicalPrior(pool="selection"|"slab"|"both")` with normal
  dynamic slabs, Dirichlet population allocation probabilities, and optional
  half-t pooled slab multipliers.
- Made monthly seasonality fixed/dynamic by default; absence is available only
  when explicitly requested.

### Inference

- Added joint Gaussian FFBS and mixed/GEV PGAS hierarchical inference.
- Added audited random sign switches for signed FS innovation coefficients.
- Estimated and stored initial level, slope, and seasonal coefficients for
  univariate and hierarchical models.
- Hardened guided disturbance PGAS on singular affine support by using the
  scaled transition loading, projecting the reference trajectory, and safely
  rejecting invalid optional ancestor moves.
- Applied the same conditioned-predecessor fallback to FS PGAS.
- Retained Joseph-form Gaussian covariance updates for long or nearly
  deterministic series.

### Diagnostics and plots

- Added population allocation, pooled slab, structural transition, and scoped
  channel summaries.
- Made the level--slope plot compare latent level with seasonally adjusted
  observations.
- Standardised progress output across engines with chain, `it`, phase, saved
  draws, scientific parameters, elapsed time, and ETA.
- Retained trace, ACF, analytic-prior, forecast, score, PIT, and direct-save
  plotting APIs.

### Workflows

- Rewrote examples as sequential, editable user scripts.
- Rebuilt the Uccle workflow around independent baselines and hierarchical
  pooled-selection/slab analyses.
- Added release validation for the PGAS regression, initial states, hierarchy,
  predictive checks, plotting, persistence, and package builds.
