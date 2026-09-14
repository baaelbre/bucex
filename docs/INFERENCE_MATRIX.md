# Inference in 1.6.1

| Model | Maintained exact-target route | Structural prior |
|---|---|---|
| Univariate Gaussian | FFBS | FS shrinkage/selection or supported generic priors |
| Univariate GEV | Laplace–MH | FS shrinkage/selection or supported generic priors |
| Gaussian hierarchy | FFBS | Pooled selection or slab scales |
| Mixed/GEV hierarchy | Laplace–MH | Pooled selection or slab scales |
| Gaussian shared components, independent residuals | FFBS | Continuous `JointPriors` |
| Mixed shared components | Laplace–MH plus Gaussian-prior block slice sweeps | Continuous `JointPriors` |
| Gaussian residual copula, optional shared states | Exact copula-corrected state and parameter updates | Continuous `JointPriors`, LKJ correlation prior |

Particle inference is retired. The optional univariate `engine="laplace"`
remains explicitly approximate. Research mixed-model scripts select
`engine="laplace_mh"`. Exact invariant targets are not evidence that a finite
chain has converged; inspect scientific contrasts as well as static parameters.

`bx.plan(model, ...)` and `fit.plan` record supported inference combinations.
Dynamic GEV log scale (`stationary`, `linear`, `rw`, `ssvs`) remains a univariate
FS capability. Joint models require stationary marginal scales. Shared/copula
models do not currently support hierarchical SSVS or estimated factor loadings.

The `joint` research baseline uses a fixed identity Gaussian copula. Estimating
its correlation via `--copula` retains the same private components, prior
scales and centered sampler. This is the controlled residual-dependence
comparison; comparing it to independent FS-SSVS changes additional assumptions.
