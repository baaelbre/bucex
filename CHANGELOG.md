# 1.7.1 — 2026-09-17

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

# Changelog

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
