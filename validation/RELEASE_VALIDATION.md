# BUCEX 1.7.1 release validation

Validation separates posterior-target checks, implementation checks and
scientific production inference. The short fit results below are **not paper
results**, and do not certify convergence of the production study.

## Executed checks

- Full source test suite: **301 passed**, zero failures/errors; see
  `pytest-1.7.1-source.xml`. The one warning documents retained TN>TX source
  records already covered by the existing data-quality report.
  This covers existing APIs, inference contracts, Gaussian references, copula
  math/feedback, priors, structural scale, forecasts, risks and saved archives.
- New independent numerical checks: GEV coefficient posteriors agree with
  one-dimensional integration at xi=-.45, 0, .45, both independent and conditional
  on nontrivial Gaussian-copula scores. Deliberately limiting optimization to
  one iteration still preserves the exact target. A long-record regression
  checks that the corrected reference supports substantial coefficient moves.
  Deterministic GEV-support repair is checked for both signs of shape.
- An isolated wheel was built and installed outside the source tree. All nine
  coefficient-reference tests passed against that installed package. Imported
  version, the normal/.01 default and bundled data through August 2026 were
  checked (`installed-import.json`, `pytest-1.7.1-installed.xml`).
- All 43 research JSON configurations resolved. Three command-line workflows
  completed: TXn with monthly scale and PNG reports; all-six copula smoke fit;
  and the three level-only sensitivity variants. Archive round trips and saved
  reference diagnostics were checked (`workflows-1.7.1.json`).
- Full-record preflight: six channels, January 1892–August 2026, 1616 months;
  normal priors, level median .01, constant scale and ASIS off. The full joint
  centered state array alone is about 8.07 GB (`preflight-1.7.1.json`).

## TXn coefficient benchmark at the unchanged old prior

Three states from the supplied 1.7.0 TXn archive were held fixed, including
paths, scale, shape and the old .02 level prior. Each coefficient kernel ran
for 1000 updates at each conditional state. The production 1.7.1 reference
converged in six optimization iterations for every case, without fallback,
support repair or covariance regularization.

Level-coefficient lag-one correlation changed from .983–.988 to .086–.093;
slope-innovation coefficient correlation from .988–.993 to .017–.089. Mean
slice evaluations fell from 6.38–7.03 to 1.08–1.13. The old reference was
10.79–14.04 local Mahalanobis units from the conditional mode.

These checks isolate the proposal change from prior changes. They do not
measure full-chain ESS or prove that every posterior block mixes rapidly.
Timing for repeated slices with a held-fixed reference excludes rebuilding it
at every outer iteration. Original archive provenance/hash and numerical
results are in `TXn-coefficient-benchmark.json`; the large user archive is not
redistributed in the package. The synthetic regression/integration tests do
not require that archive.

## Short full-record TXn execution check at the new prior

A fresh fit used all 1616 months, normal prior medians (.01,.00005,.02), constant
scale, ASIS off, four chains, **60 warmup + 100 retained per chain**. It completed
640 outer iterations in about 426 seconds in this environment.

- Path MH acceptance: .945.
- Mean coefficient slice evaluations: 1.115.
- Reference convergence: 100%; mean optimization iterations: 6.04.
- Reference fallback, support repair and covariance regularization: zero.
- Path support rejections: zero.

This short run is **not converged sufficiently for reporting**. For example,
initial-slope R-hat was about 1.218 (bulk ESS 22); level-SD R-hat 1.033 (ESS 76);
slope-SD R-hat 1.049 (ESS 100). A larger production budget and scientific-target
checks remain necessary. Do not compare these posterior summaries directly
with the old fit as a sampler-only experiment: the level prior also changed.
Settings, per-chain computational metrics and diagnostics are retained in
`TXn-short-run.json` and `TXn-short-run-diagnostics.csv`.

## Limits

No four-chain 2000+2000 full-record fit or full joint production fit was completed
as part of this release. No new claim of prior robustness, acceleration,
forecast calibration, coverage or publishability follows from passing tests.
The original reviewer experiments and prior/scale comparisons remain research
runs to execute. A constant-scale model may still fail calendar-specific fit
checks after the sampling issue is corrected.

The reference optimizer can fail or need regularization on other data. Those
cases retain an exact-target corrected update but may mix poorly; inspect the
new saved metrics. The Gaussian copula does not fix residual serial dependence,
miscalibrated margins or ordering violations. See `docs/REVIEWER_MATRIX.md`.
