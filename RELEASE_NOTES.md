# BUCEX 1.6.5 — calendar forecasts and diagnostics

- Full SERRA runs follow the latest bundled month (August 2026); named historical configurations preserve 1892–2022. Actual fitted dates are printed and saved.
- Added draw-wise calendar averages/extremes, leap-year day weights, DJF ending-year labels, explicit incomplete windows and analytic aggregate risks.
- PNG reports now include slopes, physical parameter/scale/target traces, PIT/Q-Q/residual diagnostics, month panels, annual/seasonal forecasts and risk curves.
- Saved-fit report regeneration and fixed-origin checks against later observations are available as short research scripts.
- Inference kernels and prior defaults are unchanged from 1.6.4. A report cannot update the posterior to new data.
- Validation: 276 source tests and 21 installed-package checks passed; full-date TXm and mixed copula execution checks completed.

See `docs/FORECASTS.md` for the API, output files, aggregation assumptions and commands. The uploaded one-chain TXm analysis remains preliminary; this release does not repair its mixing or add residual serial dependence.

## Earlier release notes

# BUCEX 1.6.4 — continuous private trajectories and joint dependence

This release implements the revised paper's main workflow:

- Continuous FS normal, lasso and triple-gamma priors in private joint fits;
  exact copula feedback in every path/parameter update. SSVS is optional.
- A public `fs_priors` constructor with deterministic median calibration,
  proper initial-state priors, and a normal GEV shape prior by default.
- Prior-whitened Gaussian regression, corrected GEV coefficient elliptical
  slices, and location-scale ASIS through invariant NCP rescaling.
- Nondegenerate lasso/PC local-scale draws at zero coefficients and removal of
  triple-gamma variance floors.
- Private static components; seasonal, linear and RW observation log scales;
  pooled harmonic/four-season/monthly Gaussian-copula dependence.
- Calendar-correct forecasting and archive/restart support for these models.
- Physical innovation-effect summaries, bulk/tail ESS, residual-score serial
  and seasonal dependence checks, and bivariate compound-risk quadrature.
- Concise SERRA scripts with resolved configuration inheritance, prior/shape
  sensitivity, staged predictive model comparison, simulation, July 2019
  endpoint checks, joint recovery and forecast uncertainty decomposition.

Reference priors: median innovation SDs (level, slope, season) =
(0.02, 0.00005, 0.02), initial level N(0,20²), initial slope N(0,.0025²),
initial seasonal coordinates N(0,2.25²), baseline observation variance IG(2,2),
xi ~ N(0,.3²) truncated to [-.5,.5]. These are explicit starting assumptions;
inspect prior predictions and sensitivity. Lasso lambda²=1; TG a=c=.5 and
global multiplier=1 are fixed. All local mixing variables are sampled.

The release keeps the univariate, shared-state and hierarchical APIs. It does
not convert their past fits to the new model. There is no residual AR process,
t copula or ordering-constrained likelihood. Seasonal copula priors depend on
channel order; the baseline LKJ prior alone is permutation invariant.

The test report distinguishes numerical target checks and execution checks
from research-length convergence and coverage. New full Uccle results are not
included or asserted. The manuscript's preliminary SSVS figures remain a
separate pilot and must be replaced before claiming continuous-prior results.
