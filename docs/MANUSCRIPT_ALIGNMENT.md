# SERRA alignment in BUCEX 1.7.0

The primary runs use the user's requested complete record through **August
2026**. Update manuscript date labels accordingly: reference January
1892–December 1921 versus recent September 1996–August 2026 (360 months each).
The original 1892–2022 record is retained as a named historical sensitivity,
with its 1993–2022 recent window. The mixed source provenance after 2022 remains
visible; no automatic homogenization claim is made.

| Scientific element | Implementation/configuration |
|---|---|
| Six monthly marginal summaries | Two Gaussian means, four GEV extrema, lower extrema reflected |
| Structural location | Per-channel level, slope, dummy seasonal, period 12 |
| Primary observation scale/shape | Unknown constant sigma/xi; no monthly scale effects by default |
| Continuous shrinkage | Median-matched lasso, normal, triple_gamma |
| Main priors | Lasso reference medians (.02,.00005,.02); initial level N(0,20²), slope N(0,.0025²), seasonal coefficients N(0,2.25²); sigma² IG(2,2); xi N(0,.3²) truncated to [-.5,.5] |
| Joint dependence | Constant Gaussian residual copula, LKJ(1); every conditional update has feedback |
| Computation | FS NCP; Gaussian FFBS; GEV Laplace–MH; exact-likelihood coefficient updates; ASIS off |
| Main estimands | Period-average level/location changes, average slopes, month-specific location changes; paired cross-series changes |
| Main risks | Original-tail monthly and annual events, forecast uncertainty and compound heat |
| Core prior sensitivity | Reference, all process medians ×.5, ×2, matched normal, matched TG; independent and joint drivers |
| Targeted sensitivity | Initial coefficients, IG observation prior, shape family/SD/support, LKJ(2/4) |
| Supplementary structure | Fixed/evolving location seasonality × constant/monthly observation scale |
| Model comparison | Small matched rolling-origin comparisons with proper scores and paired uncertainty |
| Reviewer validation | Shape grid, zero/weak dynamics, endpoint stress, approximate/exact benchmark, joint recovery, annual aggregation |
| Fallback | Six independent runs remain fully supported |

`period_contrasts.csv` and `contrast_definitions.json` provide the numerical
inputs for manuscript period comparisons. `*_period_risks.csv` describes
month-specific changes. `convergence.json` screens scientific quantities and
parameters; trace inspection and adequate Monte Carlo precision remain needed.
The run configuration, rather than prose defaults, is authoritative.

The new general `Latent`/`StructuralScale` API is a software extension. The
paper does **not** need to claim a full distributional location-and-scale
model or benchmark it to use this release. Shape remains constant. Factor
models, sparse post-selection, residual t copulas, residual serial dependence
and ordering-constrained likelihoods are not part of the revision's primary
implementation. Existing shared-location APIs are preserved separately.

Software checks are not scientific results. No release smoke output replaces
a manuscript results placeholder. Final convergence, prior robustness,
predictive performance and reviewer simulations still require production runs.
