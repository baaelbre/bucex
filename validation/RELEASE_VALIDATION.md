# BUCEX 1.6.5 release validation

This is a reporting and calendar-prediction update. It does not change inference kernels or prior defaults from 1.6.4, and it does not establish convergence of a substantive Uccle fit.

## Checks

- Final full source regression suite: **276 passed** in 261.87 seconds; one expected warning records retained daily TN > TX pairs. See `pytest-1.6.5-source.xml`.
- Twelve new calendar-prediction tests use independent Gaussian/Gumbel formulas: leap-year weighting, observation-variance propagation, original-scale reflected minima, within-draw probability products, partial-window exclusion, DJF ending years, joint/scalar consistency, month views and invalid calendar input.
- The first regression run passed 274 tests; two checks still pinned the preceding release version. Those version pins were updated, both tests passed on rerun, and a final full suite was run. The earlier XML is retained separately for transparency.
- An isolated wheel installation was imported from outside the source tree. Twenty-one calendar, prediction and reviewer-experiment checks passed against the installed package. Its path/version/data endpoint are in `installed-import.json`.
- Wheel and source-distribution builds completed. The user-facing source ZIP includes research scripts, tests, docs and validation records.
- A short TXm execution check fitted all 1,616 months, January 1892–August 2026, then exported forecasts/reports. A mixed Gaussian/reflected-GEV copula execution check fitted 36 months through August 2026 and exercised channel aggregation/reporting. Both use only four retained draws: no scientific inference follows from them.
- Existing uploaded 1.6.4 TXm archive loaded and generated posterior, scale, trace, residual, calendar-prediction and risk outputs without refitting. A fixed-origin forecast check used later observations through August 2026.
- PNG visual checks included slopes, parameter traces, monthly scale, PIT/Q-Q, 12-month panels, annual/seasonal forecasts, held-out annual outcomes and lower-tail aggregate risk curves. Calendar year labels and one-period interval rendering were checked.

## Scientific scope

Annual/seasonal Gaussian means use calendar-day weights; extrema use max/min inside each posterior predictive path. Analytic risk integration assumes conditionally independent monthly residuals. Posterior parameter/state dependence is retained; a cross-series copula does not add residual serial dependence. Incomplete periods are labelled and excluded by default. No partial year is silently treated as complete.

The uploaded TXm result is one chain with 500 retained draws. Endpoint slope and monthly observation scales have low effective sample sizes, and residual lag-one dependence remains. Generated figures are preliminary. Full-record multichain refitting, convergence assessment, rolling-origin calibration, scale-model comparison and aggregate-uncertainty checks remain necessary before final scientific claims.
