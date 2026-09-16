# BUCEX 1.6.4 release validation — 15 September 2026

## Executed checks

- **264 source tests passed** in 237.51 seconds. Machine-readable evidence:
  `pytest-1.6.4-source.xml`.
- **30 installed-wheel tests passed** in 10.10 seconds, from outside the source
  directory: `pytest-1.6.4-installed.xml`. The installed import and the bundled
  primary data shape (1572 months × 6 series) are recorded in `installed-import.json`.
- Source distribution and wheel built successfully. Release source, wheel and
  sdist contents are checked for accidental nested packages and run outputs.
- Nine research workflow groups executed with deliberately tiny budgets. See
  `workflows-1.6.4.json`; all use four warmup and four retained draws in one chain.
- Seasonal copula panels and continuous scalar report figures rendered. A
  final x-tick rotation improves panel readability; it does not change inference.

The source suite reports one expected data-quality warning: the Uccle source
contains two retained daily TN > TX pairs. The corresponding test verifies the
documented aggregation behavior. This is not an inference failure.

## What the numerical checks establish

| Target | Independent check |
|---|---|
| Whitened Gaussian coefficient posterior | Direct dense Gaussian conditioning, including tiny prior SDs and induced initial-level/slope covariance |
| Continuous GEV coefficient update | Posterior location mean compared with numerical quadrature |
| Calibrated priors | Simulated normal/lasso/TG physical SD medians agree with the declared calibration |
| Lasso zero-coefficient local scale | Nondegenerate Gamma conditional moments, rather than freezing the local scale |
| Continuous ASIS | Centred path remains invariant even with innovation scales below 1e-14 |
| Independence reduction | Normal/lasso/TG private fits with R=I give the same draws as the absent-copula kernel under identical seeds |
| Seasonal dependence | SPD matrices, zero-effect reduction, seasonal grouping, calendar forecasting, joint scores and serialization |
| Observation scale | Constant/linear/RW paths in mixed fits, forecasting and warm starts |
| Rare compound events | Deterministic bivariate quadrature agrees with independent/product and known Gaussian orthant cases |
| Established inference | Existing copula, shared-state, hierarchy, shape/scale, endpoint, posterior and predictive tests remain passing |
| Compatibility | Historical archives, generic static automatic inference, scalar APIs and named FS priors |

These checks give evidence for implemented targets and API behavior in tractable
cases. They do not prove every model is numerically reliable for every dataset.

## Execution checks, not paper findings

The short workflows covered:

1. Gaussian and upper/lower GEV univariate fits;
2. all six private margins with a harmonic residual copula;
3. matched normal/lasso/TG and normal/uniform shape sensitivity;
4. known-truth mixed-margin constant-copula recovery, comparing independence;
5. static seasonality, absent slope, linear and RW observation-scale prediction;
6. all eleven generating shapes from -.5 through .5, fitted on [-.75,.75];
7. July 2019 endpoint analysis and forecasts fitted before the event;
8. latent/location/observation forecast uncertainty and annual aggregation;
9. matched independence/constant/harmonic copula held-out workflows.

Numerical pilot outputs are intentionally not included as empirical results.
The release retains only compact execution evidence, not dozens of saved pilot
fits or generated figures. The configs specify substantially longer studies.

## Remaining scientific work

Run the full-record multichain analyses and assess physical SDs/variances,
climate changes, shape, correlations and risks. Investigate R-hat >1.01, low
bulk/tail ESS, support problems and high coefficient slice costs. Assess
sensitivity using converged fits. Complete replicate-based recovery and held-out
coverage/score studies with uncertainty over independent replicates/blocks.
The legacy approximate/exact benchmarks, all full sensitivity settings, and
research-length studies have not been declared completed by this release.

Seasonal dependence additionally needs prior-width/channel-order sensitivity
and recovery under truly seasonal dependence; a passing calendar/API check is
not a seasonal recovery study. RW scale needs its own mixing and forecasting
assessment. Inspect remaining serial residual correlation and ordering
violations. The current observation model does not enforce summary order or
have Gaussian-copula asymptotic tail dependence.

No new continuous-prior scientific fit can be inferred from the prior 1.6.3
SSVS pilot. Passing tests, fixed iteration budgets or plausible-looking paths
do not guarantee publishability.
