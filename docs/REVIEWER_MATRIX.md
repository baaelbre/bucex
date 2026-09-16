# SERRA reviewer map and remaining experiments

The rewritten paper and software supply a concrete revision. They do not turn
unexecuted studies into results. Preliminary pilot figures are not evidence for the new continuous-prior model;
replace them only after adequate sampling.

| Reviewer issue | Revision and runnable study | Evidence still needed for submission |
|---|---|---|
| 1. Independent bulk/tail fits overstated as unified | Methods distinguish six independent private continuous FS models from a joint residual copula; `univariate`, `copula --independence`, `copula` | Converged private fits; if retaining copula claims, matched joint fits, feedback and held-out evidence |
| 2. Near-zero variances may be prior driven | Median-matched continuous priors and explicit static model comparisons; `sensitivity` and zero/weak-dynamics `simulate` configs | Prior/posterior plots for SDs **and variances**, all three processes, initial coefficients, trajectory/risk stability and replicate uncertainty |
| 3. Laplace approximation near negative-shape endpoints | Exact likelihood MH correction; Gaussian and Gumbel integration reference tests; paired `endpoint` and endpoint stress `simulate` | Full record and pre-July-2019 fits, local support gaps, risk changes, failures, acceptance, ESS and across-chain stability |
| 4. Shape-prior bounds may influence results | `sensitivity/targeted.json` varies fitted bounds; separate generating-shape grid `-.5,-.4,...,.5` with wider fitting support | Shape-bound mass, endpoint/return-level/risk stability; sufficient converged recovery replicates |
| 5. Central 90% coverage misses tail inadequacy | `validate`: 90/95/99% central intervals, .005–.995 direct quantiles, PIT, proper/threshold scores | Adequate held-out event counts, calendar/series/horizon breakdown, dependence-aware uncertainty and failure reporting |
| 6. Flat long-horizon interval widths | Forecast derivation and `forecast_check`: latent/predictor/observation intervals, future innovations, total-variance accounting | Long-horizon simulations and plots with model/parameter uncertainty; discuss when GEV moments do not exist |
| 7. Annual maxima aggregation | Exact complete-block annual identity, conditional product before posterior averaging; `forecast_check` | Numerical aggregation agreement and clear monthly versus annual interpretation; state conditional temporal-independence assumption |
| Minor 1–3: wording, undefined parameter and small text | Complete prose rewrite, defined bases/periods/coefficients, readable vector figure settings | Final author proofreading and rendered final figures after placeholders are replaced |

## Added methodological claims need their own evidence

| Addition | Implementation/protocol | Remaining result |
|---|---|---|
| Seasonal observational variability | Constant primary scale; `sensitivity/structure.json` and `joint_structure.json` compare monthly scale effects | Recovery, seasonal PIT, marginal trajectory/risk sensitivity |
| Full copula feedback into private FS | Conditional Gaussian-score likelihood in every block update; identity-kernel and integration checks | Recovery of R and innovation magnitudes; converged matched Uccle fits; practical-correlation intervals; LKJ(1/2/4) sensitivity |
| Distributional contrasts | Matched-draw private components and `contrast_diagnostics` | Hot/cold/mean differences with posterior covariance, explicit distinction between location and quantiles |
| Compound risks | Joint prediction and compound heat Brier scores; `compare` | Enough events for finite-threshold calibration; report Gaussian-copula tail limitation |
| Physical summary compatibility | Predictive crossing diagnostics, original-scale draws unchanged | Acceptable crossing frequencies for the intended use, or revise to an order-compatible observation model |
| Later data extension | Primary fit through August 2026; named historical comparison and mixed-source provenance | Source comparability and sensitivity to including post-2022 data; no unverified homogenization claim |

## Publication decision

The independent six-series analysis remains defensible if it answers the
reviewer's prior, endpoint and forecasting questions with adequate evidence.
If the joint sampler does not mix adequately, retain that analysis and defer
the copula results; remove unsupported joint empirical claims from the paper.
If it mixes and shows useful calibrated joint inference, it strengthens the
methodological story. Neither adding a copula nor passing tests guarantees
acceptance. Novelty, converged experiments, actual figures and author/editorial
review remain necessary.

A negative-shape endpoint is a model support boundary, not a physical upper
limit. Do not discard nonnegative shape draws to manufacture a finite endpoint
interval. Do not report pointwise bands as simultaneous bands, a 100-monthly-
block level as a 100-year level, or a predictive Monte Carlo average as a
posterior credible interval for risk. These distinctions are explicit in the
revised text and result APIs.

The staged `model_comparison` workflow evaluates location seasonality, slope reductions, constant/monthly scale, and constant/harmonic dependence on matched held-out folds. Seasonal copula prior-width and channel-order sensitivity remain required study choices; they are not implied by a successful smoke run. Residual short-memory and t-copula extensions are deferred.
