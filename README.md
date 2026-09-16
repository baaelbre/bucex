# BUCEX 1.6.4

Bayesian unobserved-component models for means and extremes, with private or
shared states. The SERRA workflow tracks six related temperature summaries
using continuous Frühwirth–Schnatter (FS) innovation shrinkage, then adds a
residual Gaussian copula with full feedback into the marginal posteriors.

## Install and check

From this release directory:

```bash
python -m pip install -e '.[test]'
python -m pytest
python -m research.serra.univariate --series TXm
python -m research.serra.copula
```

The last two commands use tiny **execution checks**, not research-length fits.
Run commands from the extracted directory so the bundled data and `research`
modules are available. Install `.[plot]` for figures without pytest.

## One model, one fitting API

```python
import bucex as bx

model = bx.Model(
    bx.GEV(scale=bx.LogScale(seasonal=bx.SeasonalScale(12))),
    [bx.LocalLinearTrend(), bx.DummySeasonal(12)],
)
prior = bx.fs_priors("gev", innovation="lasso")
y = bx.load_uccle_multiseries(series="TXx", end="2022-12-01")["TXx"]
fit = bx.fit(y, model, priors=prior, parameterization="fs", asis=True,
             mcmc=bx.MCMC(chains=4, warmup=2000, draws=2000, seed=31))
fit.diagnostics()["parameters"]
fit.innovation_effect_draws(120)       # SD of each future location contribution
fit.exceedance_probability_draws(35)  # respects maxima/minima orientation
forecast = fit.forecast(120, seed=32)
fit.save("TXx.bucex")
```

Budgets are starting points. Assess convergence for physical innovation SDs,
variances, changes, risks and correlation coefficients; a visually smooth path
or high acceptance rate does not establish convergence. Exact posterior
**targeting** does not establish finite-run accuracy.

## Add dependence without sharing the trajectories

```python
channels = [
    bx.Channel("mean", bx.Gaussian(scale=bx.LogScale(seasonal=bx.SeasonalScale())),
               [bx.LocalLinearTrend(), bx.DummySeasonal()]),
    bx.Channel("maximum", bx.GEV(scale=bx.LogScale(seasonal=bx.SeasonalScale())),
               [bx.LocalLinearTrend(), bx.DummySeasonal()]),
]
model = bx.MultiSeriesModel(channels, copula=bx.GaussianCopula(eta=2))
priors = bx.MarginalPriors({c.name: bx.fs_priors(c.family) for c in channels})
# data is an aligned DataFrame with columns "mean" and "maximum".
# fit = bx.fit(data, model, priors=priors, parameterization="fs", asis=True,
#              mcmc=bx.MCMC(chains=4, warmup=2000, draws=2000, seed=31))
```

Each channel retains its own location, seasonality, observation scale and GEV
shape. Every marginal update includes the copula conditional likelihood. This
is a joint Bayesian model, not a copula fitted afterwards to point estimates.
Proper independent priors do not prevent dependence in the posterior.

- `fs_priors(..., innovation="normal" | "lasso" | "triple_gamma")` matches prior
  medians of physical innovation SDs. It fixes lasso lambda² and triple-gamma
  global/shape hyperparameters; local mixing variables remain sampled.
- `LogScale("constant" | "linear" | "rw", seasonal=SeasonalScale())` keeps
  observation seasonality distinct from latent location seasonality.
- `LocalLinearTrend(trend_mode="static")` and `DummySeasonal(mode="static")`
  give explicit fixed-component comparisons in the private continuous kernel.
- `SeasonalGaussianCopula(structure="harmonic", prior_sd=.25)` adds pooled
  periodic dependence. Four-season and monthly contrasts are also available.
- `forecast.compound_probability_draws({"mean": (">", 25), "maximum": (">", 35)})`
  integrates bivariate residual noise per parameter/state draw by quadrature.
  The existing `compound_probability` method remains a simulation estimate.

## Research guide

Start with [research/serra/README.md](research/serra/README.md). It lists the
necessary scripts, commands, output files, and interpretation checks in paper
order. Configurations inherit a common `base.json`; every run saves its fully
resolved configuration. The manuscript's original 1892–2022 window is explicit.

The implementation-to-manuscript mapping is in [docs/MANUSCRIPT_ALIGNMENT.md](docs/MANUSCRIPT_ALIGNMENT.md).

See [docs/INFERENCE_MATRIX.md](docs/INFERENCE_MATRIX.md) for supported routes,
[docs/REVIEWER_MATRIX.md](docs/REVIEWER_MATRIX.md) for experiments, and
[docs/MIGRATION.md](docs/MIGRATION.md) for compatibility. Shared-state and
hierarchical APIs remain available; they are separate from the private FS
paper workflow. Exact SSVS remains optional. PGAS is retired.

## Scientific limits

The new private kernel requires complete, aligned Gaussian/GEV series with
local-linear trends and optional dummy seasonality. It does not implement
serially dependent copula residuals, a t copula, regressions, or shared factors.
The broader existing model API has separate supported routes for shared states.
A Gaussian copula has no asymptotic tail dependence for nonsingular R.

Independent marginal likelihoods plus an unrestricted copula do not enforce
min ≤ mean ≤ max. Ordering diagnostics retain the original draws; sorting or
rejection would change the model. Seasonal copula priors depend on channel
ordering, unlike the constant LKJ prior. All reported uncertainties must be
qualified by model adequacy, prior sensitivity and Monte Carlo accuracy.

This release supplies software and executable studies. It does not claim that
new full-record fits, simulation coverage or reviewer experiments are completed.
See `validation/RELEASE_VALIDATION.md` for checks actually run on this release.
