# bucex 1.3.0

`bucex` fits Bayesian unobserved-components models to Gaussian and generalized
extreme-value observations. The package combines a declarative structural
model API, componentwise SSVS, approximate Laplace state updates, exact
Laplace independence-MH updates, and exact-density PGAS state updates.

Version 1.3.0 makes the scientific examples easier to inspect and harder to
misconfigure. Every setting is read from a short JSON file; the numbered
Python scripts contain the sequential model, fit, and plotting workflow rather
than a second layer of environment-variable overrides. Examples 02/03/04/08
share one simulation JSON, examples 05/06/09 share one calibrated Uccle JSON,
and example 07 has a centred-IG JSON. Figure titles are absent by default and
can be enabled in JSON. The exact-inference kernels and archive schema are
unchanged. The release includes ten transparent analysis scripts:

1. the complete Uccle record from 1892 and the evolution of TXx;
2. matched GEV shape and scale simulations;
3. six explicit structural simulations;
4. selection recovery with Laplace state updates;
5. the same recovery experiment with Laplace-initialized PGAS;
6. Laplace analysis of TXx, TXn, TNx, and TNn;
7. PGAS analysis of the same four series.
8. a centered random-walk GEV mixing benchmark with a fast Laplace default,
   conjugate inverse-gamma process-variance updates, and exact validation
   through Laplace-MH or PGAS.
9. exact Laplace-MH fits for the same six structural simulations as item 4;
10. exact Laplace-MH fits of Uccle TXx, TXn, TNx, and TNn under the same
    model and calibrated priors as item 6.

There is no presentation workflow layer. Every script constructs its models
with `bx.Model`, `bx.LocalLinearTrend`, `bx.DummySeasonal`, and `bx.GEV`, then
calls `bx.simulate`, `bx.fit`, `FitResult` summaries, and plotting methods
directly.

## Installation

```bash
python -m pip install ".[plot,test]"
python -c "import bucex; print(bucex.__version__)"
```

## Model

For a monthly block extreme,

\[
Y_t\mid\eta_t,\sigma,\xi\sim
\operatorname{GEV}(\eta_t,\sigma,\xi),
\qquad \eta_t=\mu_t+\gamma_t,
\]

subject to \(1+\xi(Y_t-\eta_t)/\sigma>0\). A local-linear component is

\[
\mu_{t+1}=\mu_t+\beta_t+s_\mu z^\mu_{t+1},\qquad
\beta_{t+1}=\beta_t+s_\beta z^\beta_{t+1},
\]

and the dummy seasonal component may have innovation coefficient \(s_\gamma\).
The Fruehwirth--Schnatter representation estimates signed coefficients and
standard-normal non-centred states, keeping the important neighbourhood near
zero accessible without an inverse-gamma process-variance prior.

For a centered Gaussian state equation, an `InverseGammaVariance` process
prior is conjugate conditional on the complete state trajectory. bucex detects
this combination automatically and draws the process variance by Gibbs. No
extra API flag is required. An inverse-gamma prior on the GEV observation scale
is not conjugate and therefore retains its Metropolis--Hastings update.

Componentwise SSVS assigns:

- level: `fixed` or `dynamic`;
- slope: `zero`, `fixed`, or `dynamic`;
- seasonal cycle: `zero`, `fixed`, or `dynamic`.

`sigma` and `xi` are inferred but constant through time in this release.

## Direct API

```python
import numpy as np
import bucex as bx

y = bx.load_uccle_series("TXx", start="1892-01-01")

model = bx.Model(
    bx.GEV(xi_bounds=(-0.5, 0.5)),
    (
        bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
        bx.DummySeasonal(period=12, mode="dynamic"),
    ),
)

priors = bx.ssvs_gev_priors(
    period=12,
    alpha_mean=float(np.median(y)),
    beta_mean=0.2 / 120.0,
    beta_sd=0.004,
    seasonal_initial_sd=2.25,
    innovation_slab_sd={"level": 0.03, "trend": 0.00010, "season": 0.05},
    level_dynamic_probability=0.5,
    trend_probabilities=(0.0, 0.5, 0.5),
    season_probabilities=(0.0, 0.5, 0.5),
)

laplace = bx.fit(
    y,
    model=model,
    priors=priors,
    engine="laplace",
    parameterization="fruehwirth_schnatter",
    mcmc=bx.MCMC(draws=1_000, warmup=1_000, chains=4, seed=26001),
)

pgas = bx.fit(
    y,
    model=model,
    priors=laplace.priors,
    engine="pgas",
    parameterization="fruehwirth_schnatter",
    init=laplace,
    mcmc=bx.MCMC(draws=2_000, warmup=2_000, chains=4, seed=26002),
    particles=bx.Particles(n=512, proposal="guided"),
)

laplace_mh = bx.fit(
    y,
    model=model,
    priors=priors,
    engine="laplace_mh",
    parameterization="fruehwirth_schnatter",
    mcmc=bx.MCMC(draws=2_000, warmup=2_000, chains=4, seed=26003),
    laplace=bx.Laplace(mh_steps=1),
)

print(laplace_mh.diagnostics()["engine"])

print(pgas.component_probabilities())
print(pgas.diagnostics()["engine"])
pgas.plot("level")
pgas.plot("level", show_observed=False)
pgas.plot(
    "slope",
    scale="decade",
    unit="slope / °C per decade",
    condition_on="dynamic",
    show_fixed=True,
)
pgas.plot("season")

replicated = pgas.posterior_predictive(draws=1_000, seed=26003)
replicated.plot(observed=pgas.observed)

forecast = pgas.forecast(120, draws=1_000, seed=26004)
forecast.plot(
    history=pgas.observed,
    history_dates=pgas.dates,
    history_points=360,
)

# July alone: no rapid month-to-month seasonal oscillation.
forecast.plot(
    phase=7,
    phase_label="July",
    history=pgas.observed,
    history_dates=pgas.dates,
    history_points=360,
)

# The latent level excludes seasonality and observation noise.
forecast.plot(
    target="level",
    history=np.median(pgas.state_original("level"), axis=0),
    history_dates=pgas.dates,
    history_points=360,
)
```

The standalone `level` and `slope` plots show the two trend components on their
proper scales. `show_observed=False` removes seasonally adjusted observations
from the level figure. Raw observations are never drawn on the slope axis;
the slope plot can report change per decade and overlay the conditional fixed-
slope posterior as dashed lines. The `season` plot overlays the seasonal
effect \(\gamma_{ij}\), without the level, for every phase of the cycle.

`FitResult.posterior_predictive()` generates replicated observations at the
fitted time points. `FitResult.forecast()` propagates both latent states and
observation uncertainty beyond the data. Their common plotting API supports
observed checks, recent history, predictive intervals, and optional latent-
predictor medians. `phase=` is one-based and filters both the forecast and
supplied contiguous history. `target="level"` returns the latent level alone;
the same choices are available in `Forecast.summary()`. Fitted trajectories
accept `fit.plot("predictor", phase=7)` and `fit.plot("level", phase=7)`.

## Laplace, Laplace-MH, and PGAS

For GEV observations, `engine="laplace"` samples states from an iterated local
pseudo-Gaussian approximation. It is useful for screening, debugging, and
initialization, but is not labelled exact posterior inference.

`engine="laplace_mh"` builds the same mode-matched Gaussian smoother
deterministically from the observations and current static parameters, draws a
complete trajectory by FFBS, and corrects it with independence
Metropolis--Hastings. If (L) is the exact observation likelihood and
\(\widetilde L\) is the Gaussian pseudo-likelihood, the log acceptance ratio is

\[
\bigl[\log L(x')-\log\widetilde L(x')\bigr]
-\bigl[\log L(x)-\log\widetilde L(x)\bigr].
\]

The exact and proposal state laws are the same, so their possibly singular
transition densities cancel. Integrated-slope and dummy-seasonal lag
coordinates are projected onto their exact affine recursion after Gaussian
simulation. If the zero FS trajectory is outside finite GEV support, the mode
search starts from a deterministic feasible trajectory that does not depend on
the current chain state. GEV endpoint violations by Gaussian proposal draws are
ordinary MH rejections; proposals are not truncated and never fall back to an
atom at the mode. The returned plan is therefore labelled exact-invariant.
Inspect state acceptance, support repairs and rejections, Laplace convergence,
and ESS per second. `Laplace(mh_steps=...)` controls repeated full-path
proposals at fixed parameters.

`engine="pgas"` uses conditional sequential Monte Carlo with ancestor
sampling and the exact GEV observation density. Fixed or excluded components
make transitions singular; bucex evaluates transition feasibility and
ancestor weights on the corresponding affine support.

`init=laplace` transfers one coherent posterior draw: static parameters,
structural indicators, signed innovation coefficients, and the complete state
path. It changes initialization, not the PGAS invariant distribution. Inspect
particle ESS, unique ancestors, path-update rates, reference-ancestor changes,
structural switching, and GEV support diagnostics before trusting a run.

The explicit classical benchmark in
`examples/07_centered_ig.py` instead uses a random-walk level,
`parameterization="centered"`, `asis=False`, and `InverseGammaVariance`
priors. It uses approximate `laplace` by default for a fast mixing diagnostic
and always exports traces, ACFs, ESS/R-hat, and process-variance summaries.
Set `inference.engine` in `examples/config/centered_ig.json` to `laplace_mh`
for exact Laplace-MH validation or to `pgas` for the exact particle benchmark.
Its purpose is diagnostic
comparison with the FS/non-centred specification, not to replace the latter as
the default structural-selection model.

## Ten standalone examples

Run the examples from the repository root. Edit their JSON files or pass a
copy with `--config`; there are no hidden scientific overrides:

```bash
python examples/00_uccle_record.py --config examples/config/record.json
python examples/01_tail_simulations.py --config examples/config/tail.json
python examples/02_structural_simulations.py --config examples/config/simulation.json
python examples/03_simulation_laplace.py --config examples/config/simulation.json
python examples/04_simulation_pgas.py --config examples/config/simulation.json
python examples/05_uccle_laplace.py --config examples/config/uccle.json
python examples/06_uccle_pgas.py --config examples/config/uccle.json
python examples/07_centered_ig.py --config examples/config/centered_ig.json
python examples/08_simulation_laplace_mh.py --config examples/config/simulation.json
python examples/09_uccle_laplace_mh.py --config examples/config/uccle.json
```

Outputs are written below
`results/<script>/<run-id>__<automatic-settings-signature>/`. The run ID is a
timestamp unless `output.run_id` is set in JSON. Every run contains
`run_config.json`; `output.results_root` and `output.overwrite` control its
location and collision policy.

Simulation fitting runs use `simulations/`, `fits/<scenario>/`,
`tables/<scenario>/`, and `figures/<scenario>/`. Uccle fitting runs use
`fits/<series>/`, `tables/<series>/`, and `figures/<series>/`. The Laplace,
PGAS, and Laplace-MH examples share the same scientific settings and artifact
names; only method-specific diagnostics and warm-start fits differ.

Scientific settings live in `examples/config/simulation.json`,
`examples/config/uccle.json`, and `examples/config/centered_ig.json`. Use a
copy without editing Python:

```bash
python examples/03_simulation_laplace.py --config my_simulation.json
```

PBS jobs use the same file through `qsub -v CONFIG=path/to/config.json`. To run
a pilot, copy the JSON and reduce `mcmc.draws`, `mcmc.warmup`, or particle
counts there. The copied file is the complete reproducible specification.

The simulations use period 4 and 1,000 observations for structural recovery.
Every simulated time series has its own figure. Only each scenario's level,
slope, and seasonal decomposition is shown as a three-panel figure. Shape and
scale comparisons use a common vertical range within each group.

The Uccle descriptive figures begin in 1892 and use robust local-linear LOESS
rather than a rolling median. The analysis figures include the complete
posterior predictor, paired latent-level figures with and without adjusted
observations, slope trajectories in degrees per decade, structural
probabilities, prior-to-posterior process SDs, GEV parameters, finite endpoints
where applicable, posterior seasonality, posterior predictive checks, and
ten-year forecasts by default. Each fit also saves a July-only trajectory and
forecast plus a seasonally adjusted latent-level forecast. Change the calendar
month with `figures.focus_month` in `examples/config/uccle.json`.

## HPC

Each Python example has a normal Bash runner and a matching PBS submission
file. The PBS file handles resources and logging and calls the runner with a
JSON path:

```bash
qsub job_scripts/submit_00_uccle_record.pbs

qsub -v CONFIG=examples/config/uccle_final.json job_scripts/submit_06_uccle_pgas.pbs
```

The ten numbered `run_*.sh` files also run directly and accept only
`[CONFIG] [MAX_WORKERS]`. For MCMC examples, `hpc/run_example.py` reads
`mcmc.chains`, launches one process per chain up to the allocated worker count,
and combines the saved fits before producing final figures:

```bash
bash bash_scripts/run_03_simulation_laplace.sh examples/config/simulation.json 4
```
Set `runtime.progress` to `true` in JSON. A one-chain local run prints progress
to the terminal; parallel chains write separate files under `logs/`, which can
be followed with `tail -f`. See `docs/HPC.md` and `docs/HPC_RUNNERS.md`.

## Risk summaries

A fitted GEV result retains upper/lower-tail orientation and provides
time-varying endpoint, exceedance-probability, return-period, and return-level
draws:

```python
levels = pgas.return_level_draws(20)
periods, years = pgas.return_period_draws(35.0, annual=True)
endpoint = pgas.endpoint_draws(original_scale=True)
```

For monthly data, annual exceedance probabilities compose the twelve fitted
monthly probabilities. These remain model-based, time-indexed posterior risk
summaries, not stationary return levels.

## Validation

```bash
python -m pytest
python validation/run_release_validation.py
python -m build
```

See `docs/INFERENCE_MATRIX.md`, `docs/UCCLE.md`, and `docs/VALIDATION.md` for
the detailed contracts.
