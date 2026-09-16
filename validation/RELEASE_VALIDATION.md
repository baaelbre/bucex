# BUCEX 1.7.0 validation

This file records software checks actually performed for this release. Short
chains are execution checks only; their posterior values are not paper results.

## Automated checks

- Full source regression run: **287 passed**, including new independent
  Gaussian/GEV conditional log-scale-slope quadrature checks, analytic future
  level/slope innovation variance, three shrinkage families, mixed copula
  structural scale, save/reload/restart, and paired period estimands.
- Final focused run: **33 passed**, including four additional guards for the
  August-2026 primary window, historical climate periods, five core sensitivity
  variants and reviewer shape grid. These overlap the full suite; do not add
  these counts as independent tests. The final source tree contains 291 tests.
- Installed wheel, imported outside the source checkout: **25 passed** on
  ancillary evolution, period estimands, calendar forecasts and risks.
- Source/wheel construction succeeded. The installed import was verified as
  1.7.0. Historical archive compatibility is covered by the source regression.

Machine-readable evidence is in the three `pytest-1.7.0-*.xml` files,
`installed-import.json`, `environment.json` and `source-manifest.json`.
The final small metadata/plotting additions were also exercised by the runnable
example and installed-package checks; the whole regression suite was not
needlessly repeated after documentation/configuration-only updates.

## Executed workflows

See `workflows-1.7.0.json` for the complete list. Checks included:

- Six independent summaries and a six-margin copula with report generation.
- An actual joint LKJ sensitivity refit and a univariate normal-prior case.
- A matched R=I held-out fold and constant-scale joint recovery.
- All eleven generating shapes from −.5 to .5, with wider fitting support.
- Reference, wider-support and uniform-prior July-2019 endpoint cases.
- The general evolving-scale example, including PNG figures and forecasts.
- A six-margin full-length fit on **1616 months, ending August 2026**, with
  1 warm-up and 2 retained draws. Each sampled scale was constant within time;
  163 endpoint/period/paired estimands were constructed successfully.

The last check validates execution on the requested data window. It cannot
validate mixing from two draws. Default production budgets, full recovery
replication, held-out calibration, sensitivity and ordering adequacy remain
research tasks. The new full structural scale option needs separate scientific
validation for any use beyond the fixed-scale SERRA reference model.

## Known boundaries

Latent evolution currently binds to Gaussian/GEV location and log scale.
GEV shape remains constant. Ancillary scale uses exact-likelihood elliptical
slice updates and private trajectories; shared-location plus structural scale,
scale regressions and arbitrary new observation families are not implemented.
No residual serial-dependence process, residual t copula or ordering constraint
is added. Declared constant sigma is estimated; it is not fixed to a number.

The tests assess numerical targets, propagation and software contracts.
They do not establish novelty, adequate application fit, or journal acceptance.
