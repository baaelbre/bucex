# BUCEX 1.8.3 verification

Verified on 23 September 2026. The final source suite passed **406 tests**
with zero failures or errors in 477.60 seconds. One data-quality warning
records the two retained daily TN > TX pairs. The JUnit report and machine-readable
verification record are included beside this file.
After the final calendar-end label correction, a further 45 targeted
seasonal, calendar, prior-assessment and publication-reporting tests passed.

The tests cover complete-season boundaries and leap-year weights, exclusion of
January-February 1892 and retention of JJA 2026, agreement of daily-derived
seasonal summaries with independent aggregation of the monthly records,
meteorological scale/copula phases, GEV maximum/minimum predictive densities
against direct order-statistic formulas, support endpoints, physical prior
calibration, runs-cluster definitions, and a complete seasonal adequacy-report
workflow. Existing univariate, shared-state, hierarchical, copula, archive,
inference and reporting regressions also passed.

The built wheel was installed in a separate directory; all 123 package Python
files matched the source byte for byte. An installed-package check reloaded the
six-channel seasonal posterior archive, generated finite forecasts, retained
season labels through JJA 2027, and exported conditional predictive discrepancies.

Research execution checks completed a six-response seasonal fit with four
parallel chains and a matched monthly-versus-seasonal forecast comparison.
Each smoke chain used 3 warmup and 4 retained draws. The forecast comparison
used a November 2023 cutoff and four complete held-out seasons. Completed-fold
resume preserved the existing completion records. Data preparation and daily
rank/clustering diagnostics also ran. All 14 Bash command blocks in START_HERE
passed syntax checks; 18 referenced configurations resolved and all ten named
research command-line entry points accepted --help. Seasonal forecast/risk and
copula figures were visually inspected.

These short posterior runs validate execution, not convergence, temperature
change, predictive superiority or asymptotic adequacy. Their results are excluded
from the release. Full scientific fits and predictive/sensitivity assessments
remain to run. Seasonal fitting uses r=1 extrema; r>1 fitting is not implemented.

See `release_1.8.3.json` for exact execution settings and coverage.
