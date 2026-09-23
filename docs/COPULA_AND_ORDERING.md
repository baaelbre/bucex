# Residual dependence and ordering

Each channel has a private level, slope, location seasonality, observation scale and, for GEV, a shape. Continuous marginal priors may share hyperparameters for their prior scales. A Gaussian copula links contemporaneous residuals through the **joint** likelihood:

\[
p(y_t\mid x_t,\theta,R)=c_R(F_{1t}(y_{1t}),\ldots,F_{Kt}(y_{Kt}))\prod_jf_{jt}(y_{jt}),
\qquad \log c_R=-\tfrac12\log|R|-\tfrac12z_t^\top(R^{-1}-I)z_t.
\]

The normal scores use original-response CDFs; reflected minima use the corresponding survival function. Conditional Gaussian states use FFBS, while GEV trajectories use Laplace independence proposals with exact MH correction. Copula correlation can therefore change fitted marginal trajectories and their paired contrasts. A copula describes residual dependence and does not create a common latent warming path.

```python
model = bx.MultiSeriesModel(channels, copula=bx.SeasonalGaussianCopula())
priors = bx.MarginalPriors({c.name: bx.fs_priors(c.family, period=12) for c in channels})
fit = bx.fit(data, model, priors=priors, parameterization="fs",
             mcmc=bx.MCMC(chains=4, warmup=1000, draws=1000))
```

`GaussianCopula(correlation=np.eye(K))` fixes conditional residual independence while retaining the joint multiseries fit. The `research/monthly/config/independence.json` and `research/seasonal/config/independence.json` alternatives use that comparison; `constant_copula.json` compares seasonal versus constant correlations.

Use `fit.copula_summary()` and residual dependence/PIT diagnostics before interpreting correlations. `future.joint_log_score(observed)` averages **joint** densities over complete posterior draws. `future.compound_probability(events)` averages unmodified joint predictive draws. Both retain fitted copula parameters and propagated state uncertainty. Matched monthly and seasonal predictions are scored on the same observed seasonal outcomes in `research/seasonal/compare.py`.

Common-block summaries obey physical inequalities such as TXn ≤ TXm ≤ TXx, but a Gaussian copula does not impose them deterministically. Use ordering diagnostics and compound-event checks to measure incompatible simulated combinations; do not sort or reject predictive draws after sampling. A Gaussian copula has no asymptotic tail dependence, so check finite-threshold tail performance empirically.
