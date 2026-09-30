# BUCEX 1.9.8.1

This release adds the separate fixed-Normal analysis: signed innovation coefficients
retain calibrated Normal priors, with no hyperparameters to estimate. Pooled and
separate half-normal hierarchies remain available as distinct comparisons.

The level grid is 0.05, 0.1, 0.5; the slope grid is 0.001, 0.002, 0.005, 0.01.
The 12 combinations use seasonal scale 0.1. Two central seasonal checks use 0.05
and 0.2. All 14 settings have all three full-record constructions. Pooled and
fixed-prior validation uses five matched origins; separate HN validation covers
the reference. Equal scales match prior variances, not full marginal distributions.

The screen uses two parallel chains, 1,000 warmup and 2,000 retained iterations
per chain. Gallade runs 390 individual fits in 100 array elements; BIOBOT runs
312 individual fits. Commands, resources and result locations are in
BUCEX-1.9.8.1-commands.md. Use a fresh output root and preserve the source when resuming.

Reports include fixed-prior versus hierarchical trajectory/forecast comparisons,
matched-calibration paired CRPS, 3×4 heatmaps for every construction, and existing
risk, allocation and convergence diagnostics. Fixed-prior local sensitivity uses
the squared coefficient score; no hyperparameter trace is fabricated. Prior
simulations now cover pooled HN and fixed Normal priors across the full grid.

RELEASE_VALIDATION.json records the targeted regression, startup, forecast,
prior-simulation, report, syntax, wheel-build and launcher checks. Short local
fits verify execution, not convergence or scientific conclusions. Production
HPC/BIOBOT jobs have not been submitted and their runtime has not been measured.
Historical fixtures are retained; the verification claim concerns the explicitly
listed release tests rather than every historical test expectation.
