# Inference in 1.7.4

| Model/prior route | Sampler | Scope |
|---|---|---|
| Scalar Gaussian, legacy constant scale | Existing FS FFBS or generic engine | Established API preserved |
| Scalar GEV, legacy constant scale or `phi` | Existing FS Laplace–MH; optional approximate Laplace | Original API preserved |
| Scalar Gaussian/GEV with `SeasonalScale` or `LogScale` | Private FS kernel adapted to scalar results | Continuous normal/lasso/TG; optional exact SSVS |
| Explicit static FS components | Private FS kernel | Continuous priors; fixed innovation terms omitted |
| Private multiseries with `MarginalPriors` | Copula-conditional FFBS / Laplace–MH | Continuous normal/lasso/TG; optional exact SSVS |
| Constant or seasonal Gaussian copula plus `MarginalPriors` | Same private kernel; correlation slice updates | Full joint feedback, complete aligned observations |
| Shared states or `JointPriors` | Existing general joint state sampler | Constant copula only; not private FS/SSVS |
| `HierarchicalPriors` | Existing structural hierarchy | Separate optional analysis, not the SERRA reference |

The private kernel requires exactly one local-linear trend, at most one dummy
seasonal block, no regressions/shared states, and at least two complete rows.
Normal/lasso/TG priors are the main supported research comparisons. Existing PC
and regularized-horseshoe priors remain accepted by the private implementation,
but are not part of the release's paper study matrix.

For GEV paths, the deterministic Laplace approximation is only a proposal;
Metropolis–Hastings corrects to the exact conditional likelihood. Continuous
GEV coefficients use a fixed Gaussian-reference elliptical slice. Its target
contains the exact likelihood and prior divided by that reference. The anchor
is held fixed during each conditional update; it is not changed based on the
incumbent coefficient vector. Gaussian coefficient updates are conjugate in
prior-whitened coordinates. Initial-level/slope prior covariance induced by
time centering is retained.

ASIS updates active continuous location innovation scales, preserving centred
paths through NCP rescaling. They do not interweave the observation-scale RW.
ASIS is rejected with exact SSVS point masses. Static component reductions omit
the corresponding coefficient and its local shrinkage variable. Continuous
priors never produce posterior inclusion probabilities.

Scale baseline, seasonal contrasts, secular scale terms, GEV shape and copula
coefficients use their full conditional likelihoods. RW log-scale paths use an
elliptical slice over the anchored Gaussian path. Missing-data private inference,
residual AR models and a residual t copula are not implemented.

Use `fit.plan.to_dict()` to record the actual backend. Approximate Laplace is
available only through the historical univariate route; it is not accepted as
a shortcut for the private copula model. No PGAS engine is present.

## Ancillary structural evolution

`StructuralScale` / named latent sigma uses private FS inference with independent
normal/lasso/triple-gamma evolution priors and exact-likelihood elliptical slice
updates of its NCP path and coefficients. Location keeps FFBS/Laplace–MH.
Every update includes the copula conditional when present. Shape is constant.
Shared-location plus structural scale and scale regressions are not supported.
See [PARAMETER_EVOLUTION.md](PARAMETER_EVOLUTION.md).
