# BUCEX 1.6.3

Bayesian unobserved-component models for Gaussian means and GEV extremes.
Build a model, fit it, inspect its components, and compute forecasts and risks
through one API. Each series can retain its own level, slope, seasonal location,
and seasonal observation scale. An optional Gaussian residual copula fits the
private FS/SSVS models jointly and feeds dependence back into their posteriors.

## Start here

From the extracted directory containing `pyproject.toml`:

```bash
python -m pip install -e ".[plot,test]"
python -m research.serra.univariate --series TXm TXx
python -m research.serra.copula --independence
python -m research.serra.copula
```

These defaults are tiny execution checks, with four retained draws. For the
full study and an explanation of every output, follow
[the SERRA run guide](research/serra/README.md). Every active research script is
in `research/serra`; conference examples and particle inference are removed.

## One trajectory

```python
import bucex as bx

y = bx.load_uccle_multiseries(end="2022-12-01")["TXx"]
model = bx.Model(
    bx.GEV(xi_bounds=(-0.5, 0.5), scale=bx.SeasonalScale(period=12, prior_sd=0.3)),
    [bx.LocalLinearTrend(), bx.DummySeasonal(period=12)],
)
prior = bx.ssvs_gev_priors(
    period=12, alpha_mean=float(y.iloc[:120].median()), alpha_sd=3.2,
    beta_sd=0.0025, seasonal_initial_sd=2.25,
    innovation_slab_sd={"level": 0.02, "trend": 0.00005, "season": 0.02},
    trend_probabilities=(0.10, 0.45, 0.45),
    season_probabilities=(0.0, 0.5, 0.5),
    sigma2_prior=bx.InverseGammaPrior(2, 2),
)
fit = bx.fit(y, model, priors=prior, parameterization="fs", asis=False,
             engine="laplace_mh", mcmc=bx.MCMC(chains=4, warmup=2000, draws=2000))
fit.diagnostics()
fit.component_probabilities()
fit.plot("level")
fit.sigma_draws()
fit.exceedance_probability_draws(35)
fit.forecast(12).summary()
fit.save("TXx.bucex")
```

For means use `Gaussian` and `ssvs_gaussian_priors`; the state update is FFBS.
For minima pass `tail="lower"` to `fit`, or declare it on a `Channel`.
Predictions and risks return to the original temperature orientation.
`GEV()` and `Gaussian()` still mean constant observation scale. Seasonal scale
is an explicit, optional declaration; the existing scalar API remains usable.

## Joint private trajectories

```python
channels = [
    bx.Channel("TXm", bx.Gaussian(scale=bx.SeasonalScale()),
               [bx.LocalLinearTrend(), bx.DummySeasonal(12)]),
    bx.Channel("TXx", model.observation, model.components),
]
priors = bx.MarginalPriors({
    "TXm": bx.ssvs_gaussian_priors(alpha_mean=12.),
    "TXx": prior,
})
joint = bx.MultiSeriesModel(channels, copula=bx.GaussianCopula(eta=2.))
data = bx.load_uccle_multiseries(series=["TXm", "TXx"], end="2022-12-01")
fit = bx.fit(data, joint, priors=priors, parameterization="fs", asis=False,
             mcmc=bx.MCMC(chains=4, warmup=2000, draws=2000))
fit.component_probabilities()
fit.copula_summary(practical_threshold=0.1)
bx.residual_dependence_check(fit)
future = fit.forecast(12)
future.compound_probability({"TXm": (">", 25), "TXx": (">", 35)})
```

`MarginalPriors` gives every channel its own independent structural priors.
`GaussianCopula(correlation=np.eye(K))` is the matched independence baseline.
Estimated dependence changes the likelihood during fitting, so trajectories,
selection probabilities and uncertainty can change. Interval narrowing is not
guaranteed. The model diagnoses, but does not enforce, physical summary ordering.

This private FS/copula route requires complete aligned data, a local linear
trend with optional dummy seasonality, and exact SSVS priors. It does not combine
with shared states, hierarchical pooling, regression, or dynamic GEV `phi`.
Those established model families remain separate API routes; see the
[inference matrix](docs/INFERENCE_MATRIX.md).

## Data and evidence

The bundled Uccle series cover January 1892–August 2026. Primary paper configs
end in December 2022; `extension_full.json` is a separate mixed-source extension.
See [data provenance](bucex/data/SOURCES.md) before interpreting the extension.
The original daily CSV can be reaggregated with `bx.derive_uccle_monthly`.

Run `python -m pytest` for numerical reference and API tests. The
[validation record](docs/VALIDATION.md) separates executed software checks from
scientific experiments still required. Four chains and a configured iteration
count are starting budgets, not a convergence certificate.

See [release notes](RELEASE_NOTES.md), [migration notes](docs/MIGRATION.md),
[seasonal scales](docs/LOG_SCALE.md), and the
[reviewer experiment map](docs/REVIEWER_MATRIX.md).
