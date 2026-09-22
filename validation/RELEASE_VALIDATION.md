# BUCEX 1.8.0 validation

Executed on Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0
and pandas 2.2.3. These checks validate software behavior, not
publication conclusions or scientific convergence.

## Regression and install checks

- Complete source regression: 357 passed; two tests still asserted version
  1.7.4 when pytest collected them. Their assertions now expect 1.8.0.
- Focused check: 16 passed, including both corrected version assertions, the
  new hierarchy tests, and the added joint-report regression.
- Installed-wheel check outside the source tree: 49 passed. Import resolved to
  the installed 1.8.0 package, archive schema 2.12.0. This also exercises the
  corrections, hierarchy, process-parallel equivalence, historical archives,
  seasonal scale/copula margins, and public fit/restart contracts.
- Across these runs, **360 distinct tests have a passing latest result**;
  there are no unresolved failures. The raw XML files and reconciliation
  details are retained in `verification-1.8.0.json`. No statistical tolerance
  was weakened to obtain these results.
- The existing bundled-data check reports two retained daily TN > TX pairs;
  their handling is unchanged and documented in the dataset quality report.

## New numerical contracts

The shared-scale conditional is compared to an independently expressed product
of normal densities and checked against numerical quadrature. The test retains
the scale normalizing constant and verifies the slice update. Joint prior draws
use one common hyperparameter per draw; conditional squared coefficients have
the reference normal moment and responses share prior magnitude dependence.
Static innovations are excluded from the hyperparameter update. Unsupported
SSVS/local-mixture combinations fail explicitly.

Mixed Gaussian/minimum-GEV fits with seasonal scales check conditional updates,
R=I equivalence, serial/parallel chain identity, distinct chain seeds, save/load
and warm restart, finite predictive density, unconditional prior comparisons,
and inclusion of monthly scale and shared-median diagnostics. Compact report
tests prevent response-label mixing and duplicate global hyperparameter tables.

## Research execution

`hierarchy/smoke.json --stage all` completed four full-record and four historical
joint fits, each with six responses and four process workers. It generated
635 PNGs, including 55 comparison figures. Shared-scale interval figures were
visually checked. All convergence summaries correctly remain `needs_review`
for this deliberately tiny run. No full posterior archives are requested by
the pilot; numerical tables and compressed traces are retained.

The pilot plan resolves January 1892–August 2026 (1,616 months), four candidates,
four full-record fits and sixteen historical refits. The explicit origins
include the 2016–2020 forecast block. Final preflight estimates 16.14 GB for
centered state arrays alone; process transfers, merging and diagnostics require
additional memory. The standalone shared-shrinkage API example also completed.

No full-record production chains, prior winner or substantive acceleration
claim were produced for this release. Run START_HERE.md before interpreting
this model as the final paper specification.
