# BUCEX 1.9.0 release verification

This records software checks, not production posterior or validation findings.
The final source passed **289 tests**, with one existing provenance warning
about two retained daily TN > TX observations. The full run took 86.53 seconds
in the available Python 3.12 runtime. The retained test log is
`release_190/tests.txt`.

## Checks completed

- Resolve and compile all 66 declared tasks under both screen and paper tiers.
  Verify full-record data (538 complete seasons), prospective record windows,
  complete five-year folds, saved fits and 95% interval declarations.
- Verify that all 25 full-record fits forecast 120 seasons. Check one-time
  application of variant multipliers, matched marginal coefficient second
  moments, retention of original anchors in the fixed-anchor control, and
  log(3) width in the static-seasonal alternative.
- Verify the expected PBS dry-run submission sizes: screen 25 standard tasks;
  paper 24 standard tasks and one longer reference. Verify equivalent native
  Slurm declarations. Check Bash syntax of all seven launch/job scripts.
  No actual scheduler jobs were submitted.
- Exercise all six responses in a short real-data run (22 seasonal blocks),
  using four parallel chains, three warm-up and four retained iterations per
  chain. Verify the exact-posterior target, archive save/load, 120-season
  forecast, all six predictive-width reports, innovation effects and
  sensitivity-report generation. This short run is deliberately flagged
  `needs_review`; it is not a converged scientific result.
- Independently test the median-absolute and legacy normal-SD initial-slope
  conventions through fitted models and the report writer. Check the 40/q
  versus 40 conversion, the 40/120 seasonal-update horizons, practical-effect
  probabilities against saved posterior draws, and 95% predictive envelopes.
- Check task idempotence, changed-source rejection, preservation of individual
  validation cases and duplicate-origin rejection. Verify that collecting
  posterior work does not require the deferred validation batch.
- Build source and wheel distributions. Check that the source distribution
  includes the launchers, job scripts, experiment declarations and daily data.
  Install the wheel in an isolated target and import version 1.9.0 / read its
  bundled data outside the source tree.
- Generate the final comparison tables and 43 manuscript-style diagnostic
  figures from the short saved fit. Inspect the shared-scale figure's labels;
  initial slope is labelled as a median absolute coefficient when applicable.

The source preserves the declared exact likelihood and MH correction; this
release changes research priors, experiment orchestration and reporting.
The updated package still requires the screen and paper fits and scientific
assessment. Numerical success alone is not evidence of predictive adequacy.
No production validation conclusions are made here.

The two `release_190/*/resolved_settings.csv` files record every effective task
configuration. `release_190/smoke_summary.json` records the computational smoke
check. Historical release-verification files remain historical records.
