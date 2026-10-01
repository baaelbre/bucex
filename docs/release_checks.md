# Release checks

The 1.0.0 implementation was checked in the release workspace on Python 3.12.
The source also passes Python 3.10 syntax parsing. CI is configured for Python
3.10, 3.11 and 3.12; those remote CI jobs have not been run as part of this
local preparation.

The local suite contains **22 passing tests**, including:

- Gaussian static-level posterior versus the analytic distribution.
- Gaussian smoothing means/covariances versus dense conditioning.
- GEV location posterior versus independent likelihood quadrature.
- GEV derivatives, original-scale lower-tail reflection and exact support.
- Known-scale GEV initialization near a finite endpoint.
- Laplace–MH correction identically zero for a Gaussian likelihood.
- Exact deterministic state recursions in singular FFBS.
- Half-normal shared-scale draws versus direct density integration, including
  very small coefficients without a variance floor.
- Orthonormal seasonal prior covariance and physical-horizon innovation variance.
- Fixed/static components, different channel structures, saved-object round trips.
- Identical draws for serial, parallel and separately submitted chains;
  incompatible or duplicate chains rejected by the combiner.
- All five analysis specifications, complete block counts, calendar forecasts,
  monthly calibration, diagnostics, native plots and the JSON sensitivity plan.

Additional execution checks cover the complete smoke workflow and figure
regeneration, validation tables and plots, joint prior simulations, the
509-block MAM-2019 refit, sensitivity comparisons, and all three demo scripts.
Generated figures were inspected for layout and the absence of titles. The
pre-2019 smoke run used the real training record but only a few iterations.

These checks establish implementation behavior, not convergence of the paper
posterior or successful execution on a particular HPC installation. The final
paper fit, full sensitivity grid and validation fits must be rerun with the
publication settings and assessed before replacing scientific results.
