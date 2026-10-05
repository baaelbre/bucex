# BUCEX 1.0.0

**Bayesian structural time series for environmental means and extremes.**
BUCEX combines additive latent components with Gaussian or generalized
extreme-value (GEV) observations. Fit an individual response first, then use
named channels when several responses should share innovation-prior scales.
The observations retain their Gaussian or extreme-value distribution.

The package uses a non-centred state representation. Gaussian paths use FFBS;
GEV paths use a Laplace/FFBS proposal with a Metropolis–Hastings correction.
Posterior simulation targets the specified model; finite chains still require
convergence and Monte Carlo error assessment.

## Install

Python 3.10 or later. From this source release:

```bash
python -m pip install '.[plot]'
```

For the paper workflows and editable development:

```bash
python -m pip install -e '.[research,dev]'
python -m unittest discover -s tests -v
```

NumPy, SciPy, pandas and threadpoolctl are core dependencies. Matplotlib is
optional and imported only when plotting. The wheel installs `bucex`; the
source archive additionally contains `research/`, examples, documentation and
tests. Use the source tree to reproduce the paper.

## Fit one series

```python
import numpy as np
import bucex as bx

y = 10 + .02 * np.arange(60) + np.random.default_rng(7).normal(size=60)
model = bx.Model(
    observation=bx.Gaussian(scale=bx.SeasonalScale(4)),
    components=[bx.LocalLinearTrend(), bx.DummySeasonal(4)],
)
fit = bx.fit(
    y, model=model, steps_per_year=4,
    mcmc=bx.MCMC(draws=1000, warmup=1000, chains=4, workers=1, seed=42),
)
print(fit.summary())
ax = bx.plot(fit, type="level")
forecast = fit.predict(40, draws=2000, seed=43)  # 40 blocks = 10 years
bx.plot(forecast, type="forecast", history=fit)
fit.save("results/example.bucex")
restored = bx.load("results/example.bucex")
```

A pandas Series supplies its channel name and, when present, regular dates.
Without dates, specify `steps_per_year` for meaningful rate plots. Values must
be finite and block dates complete and regularly spaced; the current sampler
does not impute missing observations or compress gaps.

Use `bx.GEV()` for maxima or `bx.GEV(tail="lower")` for minima. The public API
always uses the original temperature scale. `xi` follows the EVT convention
(SciPy's `genextreme` uses `c=-xi`). Negative `xi` gives a finite upper endpoint
for maxima or a finite lower endpoint for minima. GEV scale is not its standard
deviation. A `SeasonalScale(p)` estimates a repeating dispersion pattern,
independently of whether the location contains a seasonal component.

## Build the model from components

| Component | Behaviour | Main choices |
|---|---|---|
| `LocalLevel` | Constant level or random walk | `mode="static"` / `"dynamic"` |
| `LocalLinearTrend` | Level with an evolving or constant rate | `level_mode`, `trend_mode`; rate may be `"off"` |
| `DummySeasonal(p)` | Repeating or evolving dummy seasonality | `mode="static"`, `"dynamic"`, `"off"` |
| `Regression(features)` | Covariate effects | Static coefficients or `mode="dynamic"` random walks |
| `Cycle(period)` | Two-state harmonic cycle | Known period, damping, static or dynamic |

“Static” means estimated but with no new innovations. `Fixed(value)` means
known. For example, a linear trend with no stochastic departures is:

```python
model = bx.Model(
    bx.Gaussian(),
    [bx.LocalLinearTrend(level_mode="static", trend_mode="static")],
    bx.Priors(initial_level=bx.Normal(0, 10), initial_slope=bx.Normal(0, .1)),
)
```

Component-specific priors override the corresponding model defaults:

```python
level = bx.LocalLevel(initial_prior=bx.Normal(15, 5),
                      innovation_prior=bx.Normal(0, .2))
cycle = bx.Cycle(period=20, damping=.98,
                  innovation_prior=bx.Normal(0, .1))
model = bx.Model(bx.Gaussian(), [level, cycle])
```

Components need distinct names and output names. Multiple seasonalities or
regression blocks can use custom names. Do not add redundant intercepts
without considering identifiability. Cycle period and damping are fixed
configuration values in 1.0.0, not estimated parameters.

## Regression and time-varying coefficients

```python
import pandas as pd

x = pd.DataFrame({"wind": np.sin(np.arange(60) / 5)})
model = bx.Model(
    bx.Gaussian(),
    [bx.LocalLevel(), bx.Regression(("wind",), prior=bx.Normal(0, 2))],
)
fit = bx.fit(y, model=model, exog=x,
             mcmc=bx.MCMC(draws=500, warmup=500, chains=2))
future_x = pd.DataFrame({"wind": np.sin(np.arange(60, 68) / 5)})
forecast = fit.predict(8, exog=future_x)
bx.plot(fit, type="component", component="regression.wind")
bx.plot(forecast, type="component", component="regression")
```

`mode="dynamic", innovation_prior=bx.Normal(0, .05)` gives each coefficient
its own random-walk innovations. Named DataFrame columns are aligned by name;
arrays use the declared order. Future covariates must be supplied for every
forecast step. They are treated as known scenarios, without modelling their
uncertainty. Standardize covariates deliberately and calibrate coefficient
priors in those units. See [examples/regression.py](examples/regression.py).

## Named distribution parameters

The component shortcut defines the location predictor. The equivalent explicit
form, with a known scale and an estimated constant shape, is:

```python
model = bx.Model(
    bx.GEV(),
    parameters={
        "location": bx.Latent([bx.LocalLinearTrend(), bx.DummySeasonal(4)]),
        "scale": bx.Fixed(1.5),
        "shape": bx.Constant(bx.Normal(0, .3)),
    },
)
```

`Constant` is estimated and time invariant; `Fixed` is known. A `Constant`
scale can take an inverse-gamma prior **on its square**. Seasonal dispersion
is declared through the observation family instead of a constant scale.

The model/compiler interfaces can describe separate latent predictors for
scale or shape. **The 1.0 sampler supports latent location only**, with an
identity link. Attempting to fit stochastic scale or shape raises a clear
`NotImplementedError`. Supporting those parameters requires their likelihood
derivatives, link treatment and conditional updates; declarations alone do
not make them statistically implemented. See [docs/extending.md](docs/extending.md).

## Several channels, optional innovation-prior pooling

```python
components = [bx.LocalLinearTrend(), bx.DummySeasonal(4)]
model = bx.MultiSeriesModel(
    channels=[
        bx.Channel("TXm", bx.Model(bx.Gaussian(bx.SeasonalScale(4)), components)),
        bx.Channel("TXx", bx.Model(bx.GEV(scale=bx.SeasonalScale(4)), components)),
    ],
    pooling=bx.Pooling(level=bx.HalfNormal(.1),
                      slope=bx.HalfNormal(.002),
                      seasonal=bx.HalfNormal(.1)),
)
# data is a DataFrame with exactly the columns TXm and TXx.
# fit = bx.fit(data, model=model, mcmc=bx.MCMC(chains=4, workers=4))
```

For component `c`, `s[c,j] | tau[c] ~ Normal(0, tau[c]**2)` and
`tau[c] ~ HalfNormal(A[c])`. Each channel retains its own amplitude, initial
state, trajectory, observation scale and shape. Set `pooling=None` for private
fixed Normal prior SDs. A subset of innovation scales can be pooled; custom
innovation names use `Pooling(groups={"cycle": HalfNormal(.1)})`.

This is a hierarchical model with a product conditional observation
likelihood. Pooling induces dependence through shared scales, but **does not
model residual cross-channel association**. There is no copula or pooled
initial-rate prior. Channels in a pooled chain are updated together.

## Priors and physical units

`Normal(mean, sd)` and `HalfNormal(sd)` use **standard deviations**. Innovation
priors act on signed amplitudes; the innovation variance is their square.
`InverseGamma(shape, scale)` has density proportional to
`v**(-shape-1) * exp(-scale/v)`. Fixed zero innovation amplitudes remove that
stochastic block exactly; no numerical variance floor is added.

The convenient `Priors()` defaults are the paper's **seasonal Celsius
calibration**, not universal defaults. The private innovation SDs are .1,
.002 and .1; initial level and seasonal-contrast SDs are 10; initial-rate SD
is .01 per block. Observation variance is IG(2,2), and shape is Normal(0,.3²).
For pooling, the `HalfNormal` arguments replace the fixed innovation prior SDs
with calibrated shared-scale hyperpriors. Use `bx.prior_samples` and
`bx.prior_predictive` to inspect assumptions before fitting. See
[docs/inference.md](docs/inference.md) for formulas and sampler details.

## Results, risk and plots

`fit["TXx"].path("level")` has shape `(chain, draw, time)`. Initial states
are retained separately at index zero of `.states`. A named component or
state is accessible through `.path(name)`; `.component_names` lists them.
Scalar posterior parameters have shape `(chain, draw)`.

`fit.summary()` reports posterior means, central intervals, rank-normalized
R-hat and bulk/tail ESS. `include_paths=True` also examines component paths.
There is no automatic convergence declaration.

```python
# risk = fit["TXx"].risk(35, tail="upper")
# mean_probability = risk.mean(axis=(0, 1))
# uncertainty = bx.summarize(risk)
# forecast.risk(35, channel="TXx")  # conditional risk in each future draw
```

Average conditional probabilities to obtain predictive event probabilities.
Intervals across those probabilities describe uncertainty about risk, while
prediction intervals concern observations. `.return_level(years, tail=...)`
returns draw-wise seasonal quantiles: select the specified season, which
occurs once per year. It does not compute a maximum over all seasons.
`bx.window_risk` and `bx.block_extremes` aggregate within continuous simulated
paths. `1 / mean_probability` is a stationary-equivalent period, not an
expected waiting time under a changing climate.

`bx.plot` and `.plot()` accept an existing Matplotlib `ax`, return that axis,
use means and pointwise 95% bands, and set no titles. Types include `level`,
`slope`, `location`, `component`, `seasonal`, `seasonal_cycle`, `normal_qq`,
`pit`, `acf`, `trace`, `posterior`, `prior_posterior`, `observation_scale`,
`risk`, `risk_curve`, `return_level` and `forecast`. Forecast figures include
the evolving location. `cycle` remains an alias for the seasonal-cycle
comparison; use `type="component", component="cycle"` for a stochastic Cycle.
See [docs/plotting.md](docs/plotting.md).

## Parallel chains and the paper

Use `workers=chains` in a script with an `if __name__ == "__main__":` guard.
Separate jobs can use distinct `chain_ids`, then `bx.combine_fits` combines
compatible independent chains. Numerical worker threads are limited to one.
See [docs/parallel.md](docs/parallel.md) for complete commands.

[research/README.md](research/README.md) documents the main analysis, monthly
blocks, private priors, constant dispersion, their combined comparison,
JSON sensitivity, prior calibration, the 2019 refit and validation. Scientific
choices live there; the installed package imports none of that code.

```bash
python -m research.paper --profile screen --dry-run
python -m research.main --profile smoke --workers 1
```

## Package structure and extension

| Module | Responsibility |
|---|---|
| `components/` | Local state matrices, priors, innovation loadings and output names |
| `parameters/` | Fixed, constant and latent distribution-parameter declarations |
| `models/` | Channels and compilation into physical and non-centred systems |
| `observations/` | Gaussian/GEV probabilities, quantiles and location derivatives |
| `priors.py` | Supported prior specifications and innovation pooling |
| `inference/` | FFBS, Laplace–MH, coefficient, observation and shared-scale updates |
| `results/` | Posterior containers and named paths |
| `plots/` | Extensible object-aware plotting handlers |
| `prediction.py`, `simulation.py` | Generic compiled-state propagation |
| `serialization.py` | Versioned JSON/NPZ storage, explicit trusted type registry |
| `research/` | Paper-only configurations, experiments and figures |

An external component can implement `build(steps, priors, exog)` and return a
`ComponentBlock`; it need not modify the sampler or forecast code. See
[examples/custom_component.py](examples/custom_component.py) and
[docs/extending.md](docs/extending.md). New nonlinear state dynamics would
require a different backend; modularity does not remove that mathematical
requirement.

Code is MIT licensed. Observation data have separate source terms, documented
in `research/data/`. See `CITATION.cff`, `CHANGELOG.md` and
[docs/release_checks.md](docs/release_checks.md). The release checks distinguish
software/numerical verification from production MCMC convergence.
