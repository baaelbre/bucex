# Validation evidence for 1.6.1

This release extends the shared-state engine with an explicit Gaussian residual
copula and consolidates scientific workflows under `research/serra`.
`validation/` contains only evidence generated for this release; the original
1.6.0 evidence remains in that release's archive.

## Installed-package checks

The installed-wheel suite passed **215 tests**, with no failures, errors or
skips, in 162.4 seconds (`validation/pytest-installed.xml`). Imports were
verified to come from the installed wheel, outside the source checkout.
After the final correction to the default Gaussian-copula Kalman covariance,
**43 targeted installed-wheel regression checks also passed**, including the
independent dense covariance reference, copula math, inference, prediction,
and ordering (`validation/pytest-final-regressions.xml`). All source package
modules matched the final installed files byte for byte. Environment records
are retained beside these XML results. These are repeated checks, not 258
different tests.

## Source integration check

An initial source-suite integration check passed **198 tests, with no errors,
failures or skips**, in 179.3 seconds. This record predates the final helper and
full-copula-curvature changes; the final installed-wheel suite supplies the
release-wide result. The machine-readable record is
`validation/pytest-source.xml`. This includes the historical archive fixtures,
independent numerical references, and the new copula/ordering contracts.

## Executed analysis workflows

All **nine configured workflow jobs** completed successfully after the report
layer was corrected to request scale-selection probabilities only for a
scale-selection fit. The final record is `validation/smoke-workflows.json`:

- Six fitting routes: independent, hierarchical, joint with independent
  residuals, shared states, joint with an estimated copula, and shared states
  with an estimated copula.
- Held-out independent and copula analysis jobs, including central/tail
  coverage, scores, PITs, and fold-specific parameter/target diagnostics.
- The combined tutorial job covering Gaussian, GEV, hierarchy and shared-state
  examples through the public package API.

Two additional univariate scale-model execution checks also passed: TXx with a
linear log-scale and TNn with stationary/linear/random-walk scale selection.
Their archives, scale reports, risks and forecasts were inspected. The record
is `validation/log-scale-smokes.json`. These optional scale models remain
available on the independent route; the joint copula backend uses stationary
marginal scales.

These jobs use four retained draws per chain. They verify that the specified
routes and reports execute; their intervals, correlations and diagnostics
cannot be used as scientific findings. Output locations in the record identify
the development-run directories, which are not bundled with the source release.

## Executed reviewer-workflow checks

The five reviewer workflows were exercised on small configurations. Their
complete settings and output locations are recorded in
`validation/reviewer-smokes.json`:

| Workflow | Executed checks | Interpretation |
|---|---:|---|
| Prior and shape-bound sensitivity | 21 TXx fits, including SSVS/lasso hyperparameters, initial-slope uncertainty and changed shape bounds | Confirms that every configured variant fits and writes prior/posterior, risk and diagnostic outputs. It does not establish robustness of the Uccle conclusions. |
| Real 2019 record | Four TXx fits: two engines by two shape-bound specifications, with July 2019 verified as 39.7 C | Compares available outputs around the actual event. The event is included in fitting; these are retrospective posterior quantities. |
| Forecast uncertainty and annual aggregation | TXm, TXx and TNn, each with 24 forecast months and two complete annual groups | Exercises Gaussian, upper- and lower-GEV uncertainty/aggregation, including the distinction between location and conditional mean. |
| Generating-shape grid | 22 fits: 11 shapes from -0.5 to 0.5, paired approximate Laplace and corrected Laplace–MH | Confirms execution across the requested shape range; one simulated dataset per shape and four retained draws do not estimate coverage. |
| Near-endpoint stress | Six fits: shapes -0.5, -0.3 and -0.1, two engines, one observation replaced by its generating 0.999 quantile | A deliberately constructed stress experiment, not unconditional sampling from the generating model. |

All listed smoke fits completed without a recorded failure. These checks retain
only four MCMC draws per chain. They are deliberately separated from the longer
recovery pilot and the unexecuted paper-level studies.

## Longer shape-recovery pilot

A larger univariate pilot completed **30 fits**: three independent simulated
60-month data sets, five generating shapes (-0.5, -0.3, 0, 0.3, 0.5), and two
engines. The same data were used for each engine comparison; innovations and
uniforms were paired across shapes within a replication. Each fit used two
chains, 100 warmup iterations and 100 retained draws per chain, with an
estimation shape prior on [-0.75, 0.75]. The full record is
`validation/shape-recovery-pilot.json`, with per-replication recovery and
scientific-target diagnostics in the adjacent CSV files.

**This budget was insufficient for trustworthy posterior comparisons.** Across
level-change, event-risk and shape targets, the largest R-hat was 1.488 for
uncorrected Laplace and 2.286 for Laplace–MH; the smallest bulk ESS was 4.72 and
2.63, respectively. Those diagnostics take precedence over attractive curves
or aggregate errors.

For transparency, the average 90% pointwise location coverage was 0.961/0.914
and risk coverage was 0.827/0.823 for Laplace/Laplace–MH. These are descriptive
summaries of poorly mixed short fits and only three independently generated
data sets per shape. They neither estimate reliable nominal coverage nor
establish an advantage of either engine. Longer, diagnostically adequate runs
and more independent replications are required; time points and paired shapes
must not be counted as independent coverage trials.

## Actual six-series copula pilot

A copula model without a shared latent factor was fitted to all six Uccle
monthly summaries for 2020–2022 (36 months, 78 state coordinates), using two
chains with 80 warmup and 80 retained iterations each. The resolved configuration
is `validation/copula-uccle-pilot-config.json`; it is runnable through the same
`research.serra.run --config ...` entry point as the ordinary analysis.

The first proposal used only marginal observation curvature. Its retained
joint-state MH acceptance was **zero**, despite frequent supplementary slice
movement. That result motivated the full joint copula gradient/Hessian and
full-covariance Gaussian pseudo-observations in the release sampler. The
superseded record is preserved as `validation/copula-uccle-pilot-marginal.json`.

A matched run with the full-curvature proposal and a 30-iteration Laplace cap
raised retained joint-state MH acceptance to **4.375%**, but only **13.125%** of
recorded mode optimizations satisfied the convergence tolerance. Scientific
level-change R-hat reached 1.564 with minimum ESS 4.07; correlation R-hat reached
2.055 with minimum ESS 2.88. The record is `validation/copula-uccle-pilot.json`.

**This copula pilot still does not provide usable scientific posterior
precision.** The likelihood and numerical reference contracts are verified,
but this short joint analysis remains difficult to sample. Supplementary slice
movement and marginal-parameter acceptance must not hide poor joint-state or
scientific-target exploration. A further six-case check rebuilt proposals at
saved parameter draws with iteration caps of 30 and 100: none converged at either
cap. More iterations improved the objective, but cost roughly three times as
much and did not resolve the issue (`validation/copula-mode-budget.json`).
Increasing this cap alone is not an established remedy. Improved sampling and
adequate multi-chain diagnostics are required before interpreting joint
correlations, warming changes or compound risks. The independent six-series
analysis remains available as the primary paper route.

## What the checks establish

The software tests exercise univariate, hierarchical and shared-state model
construction, exact-target sampling, lower-tail orientation, covariance,
forecasts, risks and archive compatibility. New copula reference checks compare
against independent multivariate Gaussian calculations, rather than merely
checking that draws have the expected shape. Tests and short executions do not
establish full-record Uccle convergence or empirical coverage.

The five independent copula-inference reference tests passed after integration
of the full observation curvature (170.6 seconds total in the development
environment). Their expected targets are calculated without
BUCEX copula densities or correlation transforms:

| Numerical contract | Executed check |
|---|---|
| Correlated Gaussian state posterior | 3,000 retained draws match analytic posterior means and the variances of cross-channel sums/differences under a fixed correlation matrix. Retained log likelihoods match direct multivariate-normal densities, and the exact Gaussian state proposal has MH acceptance one. |
| Observation-scale update under dependence | 7,000 retained draws match first/second posterior moments from independent one-dimensional numerical quadrature. A marginal-only scale update would target a different posterior. |
| Correlation prior and missing observations | 5,000 retained draws in a three-channel model with only singleton observed subsets retain the LKJ(2) prior: every ordinary correlation has mean zero and second moment 1/6, within ESS-based Monte Carlo tolerances. The data contain no cross-channel information in this design. |
| Full observation covariance and missing channels | A dense random-walk Gaussian reference with alternating missing channels matches Kalman log likelihoods and smoothed means/covariances to 1e-11; its Laplace mode and Gaussian pseudo density match to 1e-9. |
| Forecast residual covariance | 12,000 draws from known Gaussian margins with unequal scales and fixed negative correlation match the declared multivariate covariance. |

These checks are in `tests/test_copula_reference.py`. Existing Gaussian
state/scale reference tests remain in the suite, alongside copula-density,
orientation, persistence, ordering and exact-update contract tests. Source and
installed-wheel evidence is recorded above. Scientific
paper configurations remain **unexecuted protocols** until their actual runs
and diagnostics are recorded; a smoke execution does not change that status.

## Running the checks

```bash
python -m pip install -e '.[plot,test]'
python -m pytest tests
python -m research.serra.tutorials --kind all --config research/serra/config/tutorial_smoke.json
```

The tutorials have deliberately tiny default budgets. They check construction,
fitting, diagnostics, figures, forecasts and persistence. Use the full tutorial
configuration and substantive analysis settings for interpretable inference.

## Scientific experiments still required

`REVIEWER_MATRIX.md` maps every reviewer comment to its workflow, output and
remaining manuscript changes. The priority is:

1. Obtain trustworthy independent fits for all six summaries, with chain and
   scientific-contrast diagnostics. These remain a complete primary analysis
   if any joint model proves computationally impractical.
2. Run process-prior and initial-state sensitivity, and change the shape-prior
   bounds around `[-0.5, 0.5]`. Inspect trajectories and risks, not only SDs.
3. Compare the approximate Laplace and corrected Laplace–MH results for the
   real 2019 event and known-truth endpoint cases. Exact-target updates can
   still mix slowly.
4. Run replicated simulations over the generating-shape grid -0.5 to 0.5,
   retaining sampler diagnostics and replication-level uncertainty in coverage.
5. Evaluate held-out central and one-sided tail coverage, PIT values and proper
   scores by channel and horizon. Keep overlapping forecasts identifiable.
6. Audit long-horizon latent/observation uncertainty and complete-year
   aggregation; explain the conditional-independence assumptions.
7. If reporting joint results, compare dependence specifications and LKJ priors,
   inspect residual and scientific-target mixing, and quantify replicated
   ordering violations. A copula fit does not enforce hard ordering.

The current Gaussian copula is contemporaneous, has fixed or estimated
correlation, and retains the Gaussian/GEV marginal specifications. It is not a
hard-ordered summary model and does not model general residual serial
dependence. Free factor loadings remain outside this release; common warming
uses specified loadings and constrained departures.

## Forecast moments versus quantiles

The forecast-uncertainty helper distinguishes the location from the GEV
conditional mean when applying the law of total variance. Its finite-sample
decomposition is a Monte Carlo summary, not a proof that the theoretical
posterior-predictive variance exists. If any retained shape is at least 1/2,
the corresponding GEV conditional second moment is infinite. Even if all
retained shapes are below 1/2, posterior integration can still diverge near
that boundary; a shape prior ending at 1/2 does not by itself guarantee
integrability. Quantile intervals remain the primary forecast-width diagnostic.
