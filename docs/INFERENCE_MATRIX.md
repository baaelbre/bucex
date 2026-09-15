# Inference in 1.6.3

| Model declaration | Priors and parameterization | Maintained inference |
|---|---|---|
| Scalar Gaussian, constant scale | Existing FS shrinkage/SSVS or generic priors | FFBS |
| Scalar GEV, constant or dynamic `phi` | Existing FS shrinkage/SSVS | Laplace–MH |
| Scalar Gaussian/GEV with `SeasonalScale` | FS exact SSVS; `asis=False` | New private-margin kernel, adapted to scalar results |
| Private multiseries, with or without Gaussian copula | `MarginalPriors`, FS exact SSVS; `asis=False` | Conditional Gaussian FFBS / GEV Laplace–MH; all updates include dependence |
| Gaussian selection hierarchy | Hierarchical priors, FS | FFBS and pooled selection/slab updates |
| Mixed/GEV selection hierarchy | Hierarchical priors, FS | Corrected Laplace–MH and pooled updates |
| Shared states, optionally residual copula | Continuous `JointPriors`, centered | Existing joint Gaussian / Laplace–MH and slice updates |

`fit.plan` records the actual route. `bx.plan(model, parameterization="fs")`
can inspect the private route before fitting. `MarginalPriors` is the explicit
prior type that chooses it during fitting. Private models can use one channel.
A copula does not introduce shared temporal states or pooled prior parameters.

The new route requires complete, finite, aligned data and private local linear
trends with optional dummy seasonality. Regression, shared states, hierarchy,
continuous lasso priors and dynamic `phi` are not accepted on that route.
The existing centered `JointPriors` route retains its own missing-data support;
that support must not be inferred for the new FS route.

Every channel can have a different declared location/scale seasonal period.
With monthly dated data and period 12, scale dummies refer to calendar months.
`SeasonalScale` with GEV `phi="linear"`, `"rw"` or `"ssvs"` is rejected.

`engine="laplace"` remains an explicitly approximate univariate comparison
for constant-scale/dynamic-phi models. It is not used for the new copula/seasonal
scale route. No PGAS implementation remains. Exact targeting refers to the
invariant posterior, not a guarantee that a finite MCMC run has converged.
