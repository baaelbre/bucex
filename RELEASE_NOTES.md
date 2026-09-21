# BUCEX 1.7.3 — parallel chains and focused prior assessment

This release adds process-parallel MCMC and a compact posterior/predictive
comparison of innovation priors. Scientific defaults, model declarations,
FS transition kernels, likelihoods and the archive schema are preserved.

- `MCMC(chain_workers=4)` runs independent chains in spawned processes. The
  Python default is serial; research configurations request four workers.
- All five inference backends use one executor. Seed streams and chain order
  are preserved across worker counts, with correctly combined diagnostics,
  initial states and archives. Numerical thread pools use one thread per chain.
- Execution metadata records workers, process IDs and complete seed states.
  Failed workers raise without returning an incomplete posterior.
- `prior_assessment` uses one JSON candidate list for full-record sensitivity
  and matched historical forecasts. Stages can run separately.
- The TNm pilot compares reference, half-level, and half-level/half-slope normal
  priors with four 500+500 chains and three five-year forecast blocks.
  Full-record data end in August 2026. All six independent margins can use the
  workflow; process-parallel execution also supports the joint copula fits.
- `innovation_prior_diagnostics` describes posterior displacement and interval
  contraction. `compare_predictive_scores` pairs forecast cases and treats a
  small number of origins descriptively. Neither selects a prior.
- `SensitivityReport` builds manuscript-style PNGs and comparison tables from
  CSVs: innovation intervals, level/slope/risk overlays, scientific contrasts,
  held-out forecasts, scores, PITs and monthly coverage.
- Compact traces and path summaries remain available without large archives.
  Validation records actual cutoffs, predictive bands, observed values and
  convergence per origin. Figures can be regenerated without MCMC.

The short-chain budgets screen execution and sensitivity; they do not guarantee
convergence. No simulation study or final scientific assessment was run for
this release. The strongest apparent acceleration is not a selection criterion.
Use predictive adequacy and sensitivity to guide a defensible prior choice.

Start with [START_HERE.md](START_HERE.md). API details and interpretation are in
[docs/PRIOR_ASSESSMENT.md](docs/PRIOR_ASSESSMENT.md). Executed software checks
are listed in [validation/RELEASE_VALIDATION.md](validation/RELEASE_VALIDATION.md).
