# BUCEX 1.8.3

Bayesian unobserved components for Gaussian summaries and GEV extremes.
Declare interpretable latent components for observation parameters, fit one
series or a joint model, and retain the same diagnostics, forecast and risk API.

```bash
python -m pip install -e ".[plot]"
python -c "import bucex; print(bucex.__version__, bucex.__file__)"
```

## Shared innovation shrinkage

Use `MarginalPriors(channels, shrinkage=SharedShrinkage(...))` to learn common
normal-prior scales for selected FS innovations. Series keep distinct process
SDs and trajectories. Copula dependence and shared regularization are estimated
jointly; neither is a shared latent warming factor. Omitting `shrinkage` retains
the existing marginal prior API. Ordinary univariate analysis remains supported.

```python
import bucex as bx

# channel_priors maps the declared response names to their FS priors.
shared = bx.SharedShrinkage.from_effects(
    horizon=360, period=12, slope_time_unit=120,
    level_displacement_sd=0.3794733192,
    slope_displacement_sd=0.1967692811,
    seasonal_displacement_sd=0.1549193338,
    initial_slope_sd=0.30, calibration="marginal")
priors = bx.MarginalPriors(channel_priors, shrinkage=shared)
```

See [the API and statistical specification](docs/SHARED_SHRINKAGE.md) for a
complete model example and the exact conditional update. Conditional priors are
normal; integrating their common scale gives a normal scale mixture. The
current SERRA specification pools level, slope and seasonal innovation
shrinkage separately and also learns a common initial-slope prior SD. Set
`initial_slope_sd=None` to disable that fourth hierarchy. Seasonal pooling regularizes changes in the seasonal
pattern; it does not share the initial pattern or monthly observation scales.

## Monthly and seasonal SERRA workflows

The monthly reference retains March 1892–August 2026 (1,614 months).
The separate `research/serra_seasonal` workflow uses 538 complete seasons,
including JJA 2026. Its command guide covers complete-block auditing,
physical prior calibration, four-chain fitting, comparable seasonal forecast
scores, and daily clustering/rank diagnostics. See [START_HERE](START_HERE.md)
and [the seasonal methods guide](docs/SEASONAL_ANALYSIS.md).

## What to run for SERRA

[START_HERE](START_HERE.md) now prioritizes a supervisor draft: exploration,
physical prior calibration, one smoke check, one main fit and two historical
forecast origins. Appendix sensitivities are separate. The main six-series
candidate has repeating monthly scales, seasonal copula dependence, four
shared regularization scales and unrestricted normal GEV shapes, through
August 2026. Saved posterior archives support replotting without refitting.

```bash
python -m research.serra.preflight --config research/serra/config/draft/main.json
python -m research.serra.copula --config research/serra/config/draft/smoke.json
python -m research.serra.copula --config research/serra/config/draft/main.json
```

Four independent chains use `MCMC(chains=4, chain_workers=4, ...)`. No cluster
scheduler is required. The default Python API remains serial; guard standalone
parallel-script entry points with `if __name__ == "__main__":`.

Generate observed-data Figures 1 and 2 without fitting:

```bash
python -m research.serra.explore
```

The existing manuscript-style plotting, parameter-evolution, copula, risk and
archive APIs remain available. Run the software tests with
`python -m pip install -e ".[test]"` followed by `python -m pytest`.

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
IG(2,2) on variance; shape defaults to unrestricted N(0,.3²). Finite
`GEV(xi_bounds=...)` and `xi_max_abs` are opt-in support restrictions.
GEV observation support is always enforced. See
[physical prior calibration](docs/PRIOR_CALIBRATION.md) for horizon-based
settings and the distinction between conditional and marginal prior SDs.
Normal innovation priors are the default, with monthly SD prior medians
(.01, .00005, .02) for level, slope and seasonality. The .01 is a prior
median, not a fixed process SD or the normal coefficient prior SD.

Version 1.7.1 corrects inefficient GEV coefficient preconditioning with a
deterministic conditional-mode reference and exact-likelihood slice correction.
See [the sampler fix and research decisions](docs/REVISION_GUIDE.md).

## Publication figures and reports in 1.7.2

The inference kernels and prior defaults are unchanged from 1.7.1. This release
adds a reproducible figure workflow, consistent interval reporting, calendar
PIT/coverage diagnostics, and compact parameter-chain exports including initial
slopes. Existing 1.7.1 fits can be re-reported without refitting.

```python
with bx.publication_style():
    figure, axis = fit.plot("level", credible_interval=.95)
    bx.save_figure(figure, "figures/TXx_level", formats=("png", "pdf"))

# Figure recipes and the series palette belong to the study configuration.
spec = bx.load_config("research/serra/config/revision/figures.json")
reports = bx.ReportCollection.from_directories("results/my_independent_run",
                                               series=spec["series_order"])
bx.save_publication_figures(reports, "figures/manuscript",
    recipes=spec["recipes"], colors=spec["colors"], formats=("png",))
```

The figure exporter uses existing numerical summaries, preserves their declared
interval level, and records source checksums. Missing trace data produce a
labelled placeholder; they are never reconstructed from a trace image. Use
`strict=True` for a final export. See [figure recipes](docs/FIGURES.md) and the
[publication run sequence](docs/PUBLICATION_RUNS.md).

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
identification and supported combinations. Shape remains constant in 1.7.2;
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
