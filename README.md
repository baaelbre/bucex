# bucex 1.0.0

Bayesian structural time series for environmental means and extremes.

Build a model from a level, a slope and an optional seasonal component. Use
Gaussian observations for means or GEV observations for block extrema. Fit
one response directly, or combine named channels with shared innovation-prior
scales. Inspect, forecast and plot the resulting objects with the same API.

## Install

Python 3.10 or newer is required. From this release directory:

```bash
python -m pip install '.[plot]'
```

For the paper scripts and development checks:

```bash
python -m pip install -e '.[dev,research]'
python -m unittest discover -s tests -v
```

The wheel installs `bucex` only. The source release also contains `research/`,
`examples/`, `tests/` and documentation. Importing `bucex` does not import
Matplotlib or execute research scripts.

## One series

```python
import bucex as bx

model = bx.Model(
    observation=bx.GEV(tail="upper", scale=bx.SeasonalScale(period=4)),
    components=[bx.LocalLinearTrend(), bx.DummySeasonal(period=4)],
    priors=bx.Priors(
        level=bx.Normal(0, 0.1),
        slope=bx.Normal(0, 0.002),
        seasonal=bx.Normal(0, 0.1),
    ),
)

# y is a finite vector or pandas Series, ordered in complete seasonal blocks.
fit = bx.fit(y, model=model, steps_per_year=4,
             mcmc=bx.MCMC(draws=1000, warmup=1000, chains=4, workers=1))
print(fit.summary())
bx.plot(fit, type="level")
bx.plot(fit, type="slope")  # original response units per decade
forecast = fit.predict(120, draws=5000)  # 120 seasonal blocks = 30 years
bx.plot(forecast, type="forecast", history=fit)
fit.save("results/my_fit.bucex")
restored = bx.load("results/my_fit.bucex")
```

The defaults reproduce the **seasonal temperature calibration**, not a
universal choice of weakly informative priors. Change them for different
units or observation frequencies. `Normal(mean, sd)` always takes an SD.
`InverseGamma(shape, scale)` is on the **baseline observation variance**.

## Named channels and pooling

```python
components = [bx.LocalLinearTrend(), bx.DummySeasonal(4)]
scale = bx.SeasonalScale(4, prior_sd=0.3)
model = bx.MultiSeriesModel(
    channels=[
        bx.Channel("TXm", bx.Model(bx.Gaussian(scale), components)),
        bx.Channel("TXx", bx.Model(bx.GEV("upper", scale), components)),
        bx.Channel("TNn", bx.Model(bx.GEV("lower", scale), components)),
    ],
    pooling=bx.Pooling(
        level=bx.HalfNormal(0.1),
        slope=bx.HalfNormal(0.002),
        seasonal=bx.HalfNormal(0.1),
    ),
)
# data is a DataFrame with exactly these column names and aligned blocks.
fit = bx.fit(data, model=model, steps_per_year=4)
bx.plot(fit, channel="TXx", type="normal_qq")
bx.plot(fit, channel="TXx", type="risk", threshold=35, phase=1)
```

For each pooled component, `s[c,j] | tau[c] ~ Normal(0, tau[c]**2)` and
`tau[c] ~ HalfNormal(A[c])`. The variance of a state innovation is `s[c,j]**2`.
Pooling learns **three shared prior SDs**, while every response retains its
own innovation amplitude, initial level, initial rate and seasonal cycle.
Pooled scales replace the corresponding private Normal SDs in each channel;
those private SDs are used when pooling is absent. `pooling=None` gives
separate prior specifications. Individual components can also be left private. Channels may use different
structural components; pass `steps_per_year` explicitly when their seasonal
structures differ. Shared amplitudes must be calibrated in compatible units.

The hierarchy is a joint Bayesian model for parameters. Conditional on states
and parameters, the observation likelihood is a product across channels.
There is **no residual copula or cross-channel observation correlation model**.
Pooling does not identify a level–slope decomposition by itself: calibrate and
check both innovation scales together.

## Dynamic, static and known components

```python
# Smooth evolving slope, with no direct level innovations:
bx.LocalLinearTrend(level_mode="static", trend_mode="dynamic")

# A straight trend with an estimated, constant rate:
bx.LocalLinearTrend(level_mode="static", trend_mode="static")

# An evolving level with no slope:
bx.LocalLevel(mode="dynamic")

# An estimated repeating cycle without seasonal innovations:
bx.DummySeasonal(period=4, mode="static")

# Known initial rate or known observation variance:
bx.Priors(initial_slope=bx.Fixed(0), variance=bx.Fixed(1))
```

`static` means estimated but without innovations. `Fixed` means known.
Omit `DummySeasonal` or use `mode="off"` to remove seasonality. Remove
`SeasonalScale` to use constant observation dispersion. There is no time
trend in the observation scale or GEV shape in this release.

## Inference and results

- Gaussian paths: forward filtering and backward sampling (FFBS).
- GEV paths: a deterministic Laplace–FFBS independence proposal, always with
  a Metropolis–Hastings correction using the exact GEV likelihood and support.
- Structural coefficients: Gaussian updates or preconditioned elliptical
  slice sampling with the Gaussian reference divided out of the target.
- Shared half-normal scales: direct generalized inverse Gaussian Gibbs draws.
- Observation parameters: conjugate Gaussian variance or slice updates;
  zero-sum seasonal log-scale contrasts have Gaussian priors.

Finite Laplace tolerances affect proposal efficiency, not the invariant target.
An exact-target transition does not imply that a finite MCMC run has converged.
Use rank-normalized split R-hat, bulk/tail ESS, trace plots and path diagnostics.
The paper sampler and mathematical conventions are described in
[docs/inference.md](docs/inference.md).

Results retain chain and draw axes. `fit["TXx"].path("slope")` has shape
`(chains, draws, time)`. Parameters and metrics are dictionaries of arrays.
Physical states, lower-tail probabilities and plots use the original response
scale; lower GEV extrema are reflected only inside inference.

Plots have **no titles by default**, use posterior means, and accept `ax=` for
multi-panel manuscript figures. Bands are pointwise central 95% intervals by
default. See [docs/plotting.md](docs/plotting.md) and the runnable
[examples](examples/).

## Reproduce the paper

Run commands from this source directory. The configurations and results are
separate from the installed package.

```bash
# Quick execution/installation check; these draws are not scientific results.
python -m research.main --profile smoke

# Main seasonal fit.
python -m research.main --profile screen --workers 2
python -m research.main --profile paper --workers 4

# Four structural comparisons.
python -m research.monthly --profile screen --workers 2
python -m research.private --profile screen --workers 2
python -m research.constant_dispersion --profile screen --workers 2
python -m research.monthly_constant_dispersion --profile screen --workers 2

# Show the complete JSON-controlled sensitivity grid.
python -m research.sensitivity --list
python -m research.sensitivity --profile screen --variant 0 --workers 2
python -m research.sensitivity --profile screen --compare

# Calibration, validation, and the pre-2019 fit.
python -m research.prior_calibration --profile screen
python -m research.validation --profile screen --workers 2
python -m research.validation --profile screen --recent --workers 2
python -m research.pre2019 --profile screen --workers 2

# Regenerate figures without refitting.
python -m research.figures results/main/screen
```

The paper profile is **4 chains × (2,000 warmup + 4,000 retained draws)**.
The screen profile is 2 × (1,000 + 1,000). These are configurable budgets,
not convergence guarantees. Default results go under `results/<analysis>/<profile>/`.
A resolved JSON record prevents accidentally reusing an output directory for
a different specification. Source data and provenance are in `research/data/`.

See [research/README.md](research/README.md) for the full analysis/figure map
and the matched monthly calibration. The monthly + constant-dispersion,
private-prior configuration recreates the **structural comparison** with the
original manuscript; it does not reinstate its old shrinkage priors or
uncorrected sampler.

## Independent chains on a workstation or cluster

Set `workers=chains` to run chains concurrently. A whole multichannel chain
occupies one process; its shared-scale updates must stay inside that chain.
Use an `if __name__ == "__main__":` guard in Python scripts. Each process
limits BLAS to one thread. More workers than chains do not add parallelism.

For separate scheduler tasks, use the same seed and distinct `--chain-id`
values, then `python -m research.combine ...`. Exact commands are in
[docs/parallel.md](docs/parallel.md). No machine paths, module loads, BIOBOT
launcher, scheduler wrapper or auto-submission code is part of `bucex`.
Optional local templates in the source ZIP are under ignored `local/`.

## Release scope

1.0.0 is the first publication API. The earlier 1.9.x numbers designated
research snapshots and are intentionally not an API compatibility promise.
[CHANGELOG.md](CHANGELOG.md) and [docs/migration.md](docs/migration.md) explain
the changes. Old serialized development fits must be read using their matching
version; they are not silently interpreted as 1.0.0 objects.

The release excludes copulas, alternative shrinkage families, model selection,
particle methods, dynamic observation scales/shapes and experimental samplers.
The software is MIT-licensed. The supplied observations retain their separate
source provenance; see `research/data/SOURCES.md`.
