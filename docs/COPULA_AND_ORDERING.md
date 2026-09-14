# Residual dependence and ordering

BUCEX 1.6.1 adds an optional Gaussian residual copula to `MultiSeriesModel`.
It works with private component paths alone, or with declared `Shared` and
`Departures` components. Every channel retains its own observation distribution,
baseline, and declared seasonal components. Ordinary univariate fitting remains
available independently of either extension.

## Model and interpretation

For a time block t, conditional on the states and parameters, the joint density is

```text
p(y_t | x_t, theta, R) = c_R(F_1(y_1t), ..., F_K(y_Kt)) * product_i f_i(y_it).
log c_R(u) = -0.5 log|R| - 0.5 z' (R^-1 - I) z,  z_i = Phi^-1(u_i).
```

All CDFs and correlations refer to original response units. For a minimum fitted
internally as a reflected maximum, its original CDF is the survival function of
the reflected variable. The implementation handles this reflection in both the
likelihood and prediction. A positive residual correlation therefore has a
consistent temperature interpretation across maxima, means, and minima.

A shared state describes co-movement in latent location over time; R describes
remaining dependence between observations in the same block. These are different
quantities. R is a correlation of conditional normal scores, not necessarily the
Pearson correlation of temperatures. Shared states and posterior parameter
uncertainty can induce predictive dependence even when R is the identity.

The current shared warming declaration uses specified loadings, ordinarily one,
and zero-sum departures. It estimates an identified common component in Celsius;
it does not estimate free dynamic-factor loadings or identify externally forced
warming. Common and departure trajectories must be assessed together, including
their joint contrasts and prior sensitivity.

## Construction and priors

Add the copula to the same general model declaration used elsewhere:

```python
model = bx.MultiSeriesModel(
    channels=channels,
    shared=shared_components,  # use () for private paths only
    copula=bx.GaussianCopula(eta=2.0),
)
fit = bx.fit(data, model=model, priors=joint_priors,
             engine="laplace_mh", parameterization="centered",
             mcmc=bx.MCMC(chains=4, warmup=1000, draws=1000))
```

Here `channels`, `shared_components`, and `joint_priors` are explicit scientific
declarations; the runnable construction is in `research/serra/models.py`.
`JointPriors` supplies proper process-SD, observation-SD, and GEV-shape priors.
Each seasonal component is channel specific. Specify static or dynamic evolution
per channel according to the question; a shared warming state does not force a
shared seasonal cycle.

`GaussianCopula(eta=2)` estimates R under an LKJ(2) prior. Larger eta increasingly
concentrates near independence; eta=1 is uniform over correlation matrices,
not uniform over each pairwise correlation when K>2. For K=6 and eta=2, each
pairwise prior has mean zero and SD 1/3. Compare eta=1, 2, and 4 when conclusions
depend on residual correlations. The implementation uses Cholesky partial
correlations with the full transformation Jacobian. See the official
[LKJ density](https://mc-stan.org/docs/functions-reference/correlation_matrix_distributions.html)
and [correlation transform](https://mc-stan.org/docs/reference-manual/transforms.html#correlation-matrices)
references.

`GaussianCopula(correlation=R)` fixes a positive-definite correlation matrix.
The SERRA `joint` baseline fixes R to the identity so that estimated-copula and
independence fits can use identical marginal priors, components, and inference
controls. This matched comparison is distinct from comparing the separate
univariate SSVS fits with a continuous-shrinkage joint model.

## Inference and diagnostics

Mixed copula models use a deterministic Laplace proposal built from the full
joint likelihood gradient and Hessian, including cross-channel curvature. The
Gaussian pseudo-observations have full covariance matrices. Indefinite local
curvature is stabilized in the proposal while retaining the likelihood score;
Metropolis-Hastings correction still uses the original joint likelihood.
With Gaussian margins this recovers the correlated Gaussian state posterior.
Observation scale, GEV shape, residual correlation, and interwoven process-scale
updates all account for the copula. An elliptical-slice refresh also targets
the full joint likelihood.

The initial marginal-only proposal was rejected after an actual Uccle pilot
showed zero retained state-MH acceptance under strong dependence. Its evidence
is retained alongside the revised-proposal pilot in `validation/`. Inspect
acceptance rates, R-hat, ESS, traces, and effective samples per elapsed second
for the revised proposal too. Exact targeting does not imply adequate finite-run
mixing, and a successful short-record pilot does not establish full-record
convergence.

Continuous or fixed process-SD priors are supported on this route. Structural
SSVS remains available for univariate and the existing selection-hierarchical
models; copula-plus-SSVS is not implemented. PGAS is absent.

Useful result methods are:

```python
fit.copula_summary()
fit.copula_correlation_draws(combine_chains=False)
prediction = fit.forecast(24, seed=81)
prediction.joint_log_score(held_out_observations)
prediction.ordering_diagnostics(bx.UCCLE_ORDER_CONSTRAINTS)
prediction.compound_probability({"TXx": (">", 35), "TNm": (">", 20)})
```

Joint predictive scoring mixes complete draw-level joint densities. It is not
the sum of separately mixed marginal log scores. Compound probabilities use
unchanged joint draws, integrating state, parameter, and observation uncertainty.
Keep channel alignment and posterior draws intact. Missing observations in
likelihood fitting use the relevant observed submatrix of R.

A Gaussian copula has no nonzero asymptotic tail dependence for nonsingular R.
It can model dependence at finite thresholds, but fit quality for simultaneous
rare extremes must be tested. A t-copula or other tail-dependent construction is
a future extension, not a capability of this release.

## Ordering is a separate model requirement

For summaries of the same complete blocks, check these necessary inequalities:

```text
TXn <= TXm <= TXx
TNn <= TNm <= TNx
TNn <= TXn,  TNm <= TXm,  TNx <= TXx
```

The bundled 1,572 monthly Uccle blocks satisfy all seven pairwise inequalities.
`UCCLE_ORDER_CONSTRAINTS` encodes them. The generic ordering diagnostic reports
each violation probability, its time profile, and the probability that at least
one constraint fails in a block. The latter is a union probability, not the sum
of pairwise probabilities. These checks apply to observations, not to a presumed
ordering of GEV location parameters and Gaussian means.

A copula cannot generally enforce these constraints while preserving the chosen
margins. If L <= M <= U almost surely, necessarily
F_L(a) >= F_M(a) >= F_U(a) for every a. In particular, a negative-shape GEV maximum
has a finite upper endpoint b, whereas a Gaussian mean has positive probability
above b. Thus some mean-above-maximum probability is unavoidable under those
unchanged margins, whatever the copula. Check its practical magnitude rather
than assuming correlation eliminates it.

This release diagnoses predictive crossings; it does not impose hard ordering.
Sorting draws changes their marginal distributions. Rejecting crossed draws
defines a conditional predictive model. An order-truncated joint likelihood
requires its state- and parameter-dependent normalizing probability in inference;
omitting that term does not fit the intended model.

Two defensible future hard-order models are positive gaps around a central
summary, or a daily-temperature model whose realizations are aggregated into all
six summaries. Both change the current observation model. If predictive crossing
probabilities are material, report that limitation and use one of these models
before making claims that require physical ordering.

## Evidence needed for a paper

Run the independent six-series analysis first. Compare fixed-identity and
estimated-R joint models with matched priors and seasonality, and then assess
whether adding a shared state improves inference or prediction. Compare held-out
joint and marginal scores, upper-tail calibration, interval widths and coverage,
residual correlations, compound risks, and crossing probabilities. Correlation
does not guarantee narrower or better calibrated credible intervals.

The reviewer experiments and remaining manuscript tasks are tracked in
`REVIEWER_MATRIX.md`. Numerical reference tests and short workflow runs establish
implementation evidence; repeated recovery experiments and converged Uccle fits
are still required for scientific conclusions.
