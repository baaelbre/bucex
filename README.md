# BUCEX 1.7.1

Bayesian unobserved components for Gaussian summaries and GEV extremes.
Declare the observation distribution, give its parameters an interpretable
structure, choose priors, and use the same fitting and posterior APIs for a
single series or related series with residual dependence.

```bash
python -m pip install -e ".[test]"
python -m pytest
```

## One series, explicit parameter structure

```python
import bucex as bx

y = bx.load_uccle_multiseries(series=["TXx"], end="2026-08-01")["TXx"]
model = bx.Model(
    bx.GEV(xi_bounds=(-0.5, 0.5)),
    parameters={
        "mu": bx.Latent([bx.LocalLinearTrend(), bx.DummySeasonal(12)]),
        "sigma": bx.Constant(),
        "xi": bx.Constant(),
    },
)
prior = bx.fs_priors("gev", period=12, innovation="normal")
fit = bx.fit(y, model, priors=prior, parameterization="fs",
             engine="laplace_mh", asis=False,
             mcmc=bx.MCMC(chains=4, warmup=2000, draws=2000, seed=1700))

location = fit.parameter_path("mu", combine_chains=False)
scale = fit.parameter_path("sigma", combine_chains=False)
slopes = 120 * fit.component_draws("slope", combine_chains=False)
risk = fit.exceedance_probability_draws(35, return_labels=False)
future = fit.forecast(120, seed=1701)
fit.save("TXx.bucex")
```

`Constant()` means **unknown but constant in time**. The scale and shape are
estimated, not numerically fixed. The original API remains valid:
`Model(GEV(), [LocalLinearTrend(), DummySeasonal(12)])`. Existing saved fits
remain readable. The default observation-scale prior from `fs_priors` is
IG(2,2) on variance; shape is N(0,.3²), truncated by the declared bounds.
Normal innovation priors are the default, with monthly SD prior medians
(.01, .00005, .02) for level, slope and seasonality. The .01 is a prior
median, not a fixed process SD or the normal coefficient prior SD.

Version 1.7.1 corrects inefficient GEV coefficient preconditioning with a
deterministic conditional-mode reference and exact-likelihood slice correction.
See [the sampler fix and research decisions](docs/REVISION_GUIDE.md).

## Give scale its own evolution

Use the same component language for a log-scale predictor:

```python
model = bx.Model(
    bx.GEV(),
    parameters={
        "mu": bx.Latent([bx.LocalLinearTrend(), bx.DummySeasonal(12)]),
        "sigma": bx.Latent(
            [bx.LocalLinearTrend(), bx.DummySeasonal(12)],
            link="log",
            priors=bx.EvolutionPriors(
                innovation="normal",
                innovation_median={"level": .01, "trend": .00001, "season": .01},
                initial_slope_sd=.001,
                seasonal_initial_sd=.3,
            ),
        ),
    },
)
```

Location and scale have **separate states and priors**. The scale level is
anchored at zero initially, with the overall baseline supplied by the unknown
observation scale. Its initial slope is regularized separately. Static/dynamic
level and slope and fixed/evolving dummy seasonality can be combined. This
extension supports normal, lasso and triple-gamma innovation priors, exact
likelihood updates, persistence, restart, simulation and forecasts.

Read [parameter evolution](docs/PARAMETER_EVOLUTION.md) for units, examples,
identification and supported combinations. Shape remains constant in 1.7.1;
unsupported parameter/family combinations fail explicitly. Ancillary full
structural models need their own mixing and scientific validation. They are
**not part of the SERRA reference analysis**.

## Related series

`Channel` accepts the same `parameters` mapping. Put channels in
`MultiSeriesModel(..., copula=GaussianCopula(eta=1))`, assign their location and
observation priors with `MarginalPriors`, and call `fit`. Every conditional
update includes the copula likelihood; univariate fits are not frozen inputs.
Fixed identity correlation is the matched joint independence benchmark.
Shared-location declarations remain available through their existing API;
new structural scale evolution uses private marginal trajectories.

## Run the revised paper

Start with [START_HERE.md](START_HERE.md), then the
[SERRA run guide](research/serra/README.md). The primary model is six monthly
structural locations, constant unknown scales/shapes and optional constant
Gaussian copula dependence. The primary window is January 1892–August 2026. Named
`*_1892_2022.json` configurations retain the historical comparison.

Research scripts are thin users of BUCEX's public API. Configurations declare
the science and computation. No PGAS or conference directories are needed.
See [manuscript alignment](docs/MANUSCRIPT_ALIGNMENT.md),
[reviewer map](docs/REVIEWER_MATRIX.md), and
[release validation](validation/RELEASE_VALIDATION.md).
