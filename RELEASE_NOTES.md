# BUCEX 1.9.8

This release prepares the broader half-normal calibration study. It changes the scientific settings and requires a fresh results root.

## Scientific design

The reference is (A_alpha, A_beta, A_gamma) = (0.1, 0.002, 0.1). The 3×3 level/slope grid uses (0.05, 0.1, 0.2) × (0.001, 0.002, 0.004), with A_gamma=0.1. Central seasonal checks use 0.05 and 0.2. All eleven calibrations have matched private full-record fits, and the private reference has five matched validation origins. Private fits remain regularized by half-normal hierarchies with response-specific scales.

Initial rates, initial seasonal contrasts, observation priors, residual independence and the exact half-normal GIG sampler remain as declared in the command guide. Variant-specific reproducible seeds give different random streams across calibrations. These are new fits; overlap in alpha/beta values with an older grid does not justify reusing its posterior.

## Execution and reports

Screen chains retain 2,000 draws after 1,000 warmup each. Gallade runs 111 fits (46 groups); BIOBOT runs 51 recent-origin fits. Focused private arrays now use the six-response bundle runner, whose task registry includes focused tasks. The default private array cap is four bundles, separate from sixteen pooled elements. All runs preserve manifest/provenance checks and saved archives.

Reports retain all six private responses and never pass an incomplete bundle. Grid heatmaps are 3×3 within each response and exclude central seasonal checks. Seasonal and matched pooling comparisons have separate figures and tables. Prior figures resolve the current reference instead of assuming A_alpha=0.01. Manual and automatic collection logs use the results root.

## Verification and limits

See RELEASE_VALIDATION.json for the precise checks and local runtime. Configuration checks, short actual fits, predictive/report exports and selected regression tests validate execution. They do not establish posterior convergence or a stable calibration region. Production HPC/BIOBOT jobs were not submitted here and their running time is unmeasured.

The source retains historical tests from releases with different fixed-prior and copula experiment plans. An exploratory run of the entire historical suite encountered obsolete version/plan expectations; it is not reported as an all-suite pass. The release gate selects current sampler, prior simulation, forecast, workflow and 1.9.8 design regressions explicitly. Historical result fixtures and archived settings are not new 1.9.8 results.
