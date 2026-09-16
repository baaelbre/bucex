# Residual dependence and ordering

## The 1.6.4 private FS model

Each series has a private level, slope, location seasonality, observation scale
and (for GEV) shape. Continuous innovation priors are independent across channels; SSVS is optional.
For a contemporaneous response vector, the likelihood is

\[
p(y_t\mid x_t,\theta,R)=c_R(F_{1t}(y_{1t}),\ldots,F_{Kt}(y_{Kt}))
\prod_j f_{jt}(y_{jt}),\qquad
\log c_R=-\tfrac12\log|R|-\tfrac12z_t^\top(R^{-1}-I)z_t.
\]

Here `z = Phi^-1(F(y))` uses original-scale CDFs. Reflected minima are handled
with the corresponding survival function, so positive residual correlations
have the same temperature orientation across all six summaries.

The conditional normal score of channel j is Gaussian with mean
`R[j,-j] @ inv(R[-j,-j]) @ z[-j]` and its Schur-complement variance. That gives
an exact conditional marginal likelihood. Gaussian states admit conditional
FFBS; GEV states use a deterministic Laplace independence proposal with an
exact MH correction. Structural coefficients/indicators, seasonal log-scale
effects, baseline scale, shape and R are updated using this conditional target.

Consequently, changing R can change the trajectories and SSVS probabilities.
This is not a second-stage residual fit with frozen margins. For paired changes,
`Var(D1-D2) = Var(D1)+Var(D2)-2 Cov(D1,D2)`; the covariance matters. Individual
credible intervals can widen, narrow or shift. Better calibration is an
empirical question, not a mathematical consequence of positive dependence.

## Declarations and priors

```python
model = bx.MultiSeriesModel(channels, copula=bx.GaussianCopula(eta=2.))
priors = bx.MarginalPriors({name: own_fs_prior for name in names})
fit = bx.fit(data, model, priors=priors, parameterization="fs", asis=True,
             mcmc=bx.MCMC(chains=4, warmup=2000, draws=2000))
```

The complete Uccle construction is `research/serra/models.py`. Fix
`GaussianCopula(correlation=np.eye(K))` for a matched independence comparison.
Both models have identical components and marginal priors. Six separate
univariate jobs remain usable if the joint chain is impractical.

`eta=2` specifies LKJ(2), mildly favoring identity; eta=1 is uniform over
correlation matrices, not over each pairwise correlation for K>2. For K=6,
the LKJ(2) pairwise prior SD is 1/3. Compare eta=1,2,4. The Cholesky partial
correlation transform includes its complete Jacobian. A continuous LKJ prior
assigns probability zero to exact independence, so `Pr(rho != 0)` is not an
informative dependence measure. Use intervals and practical thresholds.

The new route supports complete aligned finite data, private local linear
trends and optional dummy seasonality, continuous FS or exact SSVS, optional seasonal/secular
observation scale, and constant or pooled periodic residual correlation. Shared states and hierarchical pooling are separate routes.
The existing continuous `JointPriors` copula model remains available, including
its own missing-data handling; that is not the new private FS backend.

## What hierarchical, shared and joint mean

| Term | What links series? | FS/SSVS in this release? |
|---|---|---|
| Independent private models | Nothing across series | Yes |
| Selection hierarchy | Pooled prior probabilities or slab scales | Existing hierarchical route |
| Shared states | An actual common latent path with specified loadings and constrained departures | Continuous `JointPriors`, not private SSVS |
| Joint residual copula | Within-month conditional observation dependence | Yes with `MarginalPriors`; full feedback |
| Shared plus copula | Both a common latent path and dependent observation residuals | Existing continuous shared route |

A residual copula does not supply free dynamic-factor loadings, common states,
or a causal warming attribution. Each declared channel retains its own
seasonality. There is no requirement that means and extreme locations share
an identical change over time.

## Evidence for residual dependence

1. Fit the private independence model after removing location/seasonal effects
   and allowing seasonal scale. Inspect residual PIT/normal scores by month
   and their temporal autocorrelation; raw temperature correlations are not
   residual-dependence evidence.
2. `bx.residual_dependence_check(fit)` compares observed score correlations
   with joint posterior replications. It integrates over fitted states/parameters
   and reports descriptive posterior predictive tail areas, not hypothesis-test
   p-values. Use adequate draws and separate checks for serial dependence.
3. `fit.copula_summary(practical_threshold=.1)` reports pairwise intervals,
   `Pr(rho>0)`, `Pr(abs(rho)>.1)`, R-hat and ESS. Assess sensitivity to LKJ eta.
4. Compare matched held-out joint/marginal scores and compound-event Brier
   scores. `research.serra.compare` keeps forecast cases paired and resamples
   whole calendar-year blocks. Try longer blocks if errors persist across years.

`future.joint_log_score(y)` is a negative log score (lower is better), obtained
by mixing complete joint conditional densities. It is generally not the sum
of independently mixed marginal log scores. `compound_probability(events)`
averages joint predictive draws; it is not a posterior credible interval for
a conditional event probability. Increase predictive Monte Carlo draws for
rare events. A Gaussian copula with nonsingular R has no asymptotic tail
dependence; finite-threshold adequacy needs direct validation.

## Ordering is an explicit limitation

For common complete monthly blocks, TXn <= TXm <= TXx and TNn <= TNm <= TNx.
Additional TX/TN inequalities require compatible daily recording windows.
The bundled monthly observations satisfy `UCCLE_ORDER_CONSTRAINTS`; source
notes flag different TX/TN windows and two daily TN>TX values in the extension.

```python
fit.ordering_diagnostics(bx.UCCLE_ORDER_CONSTRAINTS)
future.ordering_diagnostics(bx.UCCLE_ORDER_CONSTRAINTS)
```

These functions report pairwise crossing probabilities and the union probability
of any crossing. The latter is not the sum of the pairwise probabilities. Check
summary observations, not an assumed ordering of Gaussian means and GEV locations.

A Gaussian mean is unbounded, while a negative-shape GEV maximum has a finite
upper endpoint. No copula can then impose mean <= maximum almost surely while
preserving both margins. In general, almost-sure order requires stochastic
ordering of the marginal CDFs. The current joint model is a working model of
summary dependence and retains this support limitation.

Do not sort predictions or discard crossed draws. Sorting changes the margins;
rejection conditions the predictive distribution. An order-truncated likelihood
needs its state/parameter-dependent normalizing probability inside inference.
If violations are scientifically material, a positive-gap model or a daily
model followed by aggregation is the next methodological step, with changed
marginal assumptions. Hard ordering is not claimed by this release.

## Pooled seasonal dependence in 1.6.4

`SeasonalGaussianCopula` adds harmonic or zero-sum calendar contrasts to
Fisher partial-correlation coordinates. Its baseline has the LKJ prior; each
contrast coefficient has a zero-centred Normal prior. Individual monthly R
matrices do not each have LKJ priors. A harmonic coefficient pair with SD .25
gives SD .25 for its combined monthly coordinate effect; orthonormal dummy
contrasts have phase variance .25²(1−1/p).

All matrices are positive definite by construction. Channels must retain a
declared ordering because the seasonal effect prior uses ordered partial
correlations. Refit meaningful permutations to assess sensitivity. Compare
constant versus seasonal dependence on the same held-out months after allowing
seasonal marginal scales; otherwise seasonal heteroscedasticity can masquerade
as changing dependence. This model has no serial residual process.

For two threshold events, `compound_probability_draws` evaluates the Gaussian
rectangle within each parameter/state draw by deterministic quadrature. It
retains posterior uncertainty and avoids reliance on rare-event counts. It
does not enforce ordering or generate asymptotic tail dependence.
