# BUCEX 1.8.0 — common innovation shrinkage with joint inference

The new `SharedShrinkage` prior learns common normal-prior medians for selected
FS innovation coefficients while retaining separate trajectories and process
SDs for every response. It integrates with `MarginalPriors`, the joint Gaussian
copula sampler, parallel chains, archives and warm starts. No SSVS or factor
model is needed; the existing univariate API is preserved.

The SERRA workflow now compares fixed-half, fixed-quarter, pooled-quarter and
pooled-half priors under the same copula likelihood, monthly observation scales
and data window through August 2026. Common scales are updated from training
data in each historical forecast fit. One origin covers 2016–2020, including
the 2019 record. The new reports distinguish unconditional individual priors
from shared hyperpriors and export both sets of posterior diagnostics.

Monthly observation-scale contrasts and initial seasonal vectors are included
in scalar convergence diagnostics; monthly scale effects also appear in trace
exports. Forecast reports retain threshold-event counts, marginal and joint
scores, compound-event scores, seasonal calibration and numerical warnings.

Read [START_HERE](START_HERE.md) for the exact commands, and
[SHARED_SHRINKAGE.md](docs/SHARED_SHRINKAGE.md) for equations and API examples.
The pilot is deliberately short. Smoothness and narrow intervals do not certify
adequacy, learning or acceleration. Publication conclusions still require the
specified convergence, sensitivity and predictive checks on the full record.

Validation actually performed for this release is recorded in
[RELEASE_VALIDATION.md](validation/RELEASE_VALIDATION.md).
