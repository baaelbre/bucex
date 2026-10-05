# Release verification

Verification is performed locally on Python 3.12. Source parsing also checks
Python 3.10 syntax. CI is configured for Python 3.10, 3.11 and 3.12; remote
CI execution and a particular HPC installation are separate from these local
checks.

The release suite includes **30 tests**, covering:

- Gaussian static-level and multivariate regression posteriors against analytic
  distributions, including posterior covariance.
- Constant and time-varying-design smoothing against dense Gaussian conditioning.
- GEV coefficient and Laplace–MH path targets against independent quadrature.
- GEV derivatives, reflected-minimum probabilities, finite endpoints and support.
- Gaussian-likelihood MH acceptance and exact deterministic lag transitions.
- GIG shared-scale conditionals against direct density integration, including
  tiny amplitudes without an innovation-variance floor.
- Initial seasonal covariance and analytical prior/forecast innovation variance.
- Fixed/static components, regression, TVP, a fixed-period cycle, and an external
  component using the public contract.
- Future-covariate alignment, named outputs, native plots and JSON/NPZ round trips.
- Identical serial/parallel/separately submitted chain streams, with incompatible
  fits and duplicated streams rejected.
- All five paper model configurations, block counts, calendars, monthly prior
  calibration, diagnostics and the JSON sensitivity grid.

Execution checks additionally exercise the five analysis pipelines and their
figure generation, historical/recent validation output, a sensitivity
comparison, the actual 509-block pre-2019 training record, the post-1970 linear
benchmark, and regression/custom-component examples. The paper prior-
calibration export uses 50,000 replications. Representative figures are
inspected for readability and absence of titles.

A wheel and source distribution are built with the declared setuptools backend.
The wheel is installed into a separate environment and the numerical suite is
run against that installation. Package contents are checked to keep research
data and scripts out of the wheel and include them in the source release.

These are numerical and execution checks, **not convergence results for the
paper**. Short smoke fits have too few draws for scientific inference. The
production main fit, full sensitivity grid and held-out validation runs must
be rerun and assessed before replacing manuscript numbers. Existing research
1.9.x and pre-publication schema-1 results are not relabelled as new fits.
