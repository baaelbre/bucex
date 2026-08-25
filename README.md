# bucex 1.3.2

`bucex` fits Bayesian unobserved-components models to Gaussian and generalized
extreme-value observations. The package combines a declarative structural
model API, componentwise SSVS, approximate Laplace state updates, exact
Laplace independence-MH updates, and exact-density PGAS state updates.

Version 1.3.2 makes the JSON/PBS workflow complete and collision-safe. Six
simulation presets live under `examples/config/simulations/`; four calibrated
single-series Uccle presets live under `examples/config/uccle/`. Every numbered
example exposes `DEFAULT_CONFIG_FILE`, accepts `--config PATH`, and reads the
file directly with `bx.load_config`. The PBS files allocate resources only;
draws, warmup, chains, seeds, priors, model choices, and figure settings remain
in JSON. Concurrent jobs receive IDs containing the configuration name and PBS
job ID, so series or scenarios starting in the same second cannot share output
directories. The release includes ten transparent analysis scripts:

1. the complete Uccle record from 1892 and the evolution of TXx;
2. matched GEV shape and scale simulations;
3. six explicit structural simulations;
4. selection recovery with Laplace state updates;
5. the same recovery experiment with Laplace-initialized PGAS;
6. Laplace analysis of TXx, TXn, TNx, and TNn;
7. PGAS analysis of the same four series;
8. a centered random-walk GEV mixing benchmark with a fast Laplace default,
   conjugate inverse-gamma process-variance updates, and exact validation
   through Laplace-MH or PGAS;
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
    alpha_sd=3.2,
    beta_mean=0.0,
    beta_sd=0.0025,
    seasonal_initial_sd=2.25,
    sigma2_prior=bx.InverseGammaPrior(2.0, 2.0),
    xi_prior=bx.UniformPrior(-0.5, 0.5),
    xi_max_abs=0.5,
    innovation_slab_sd={"level": 0.02, "trend": 0.00005, "season": 0.02},
    level_dynamic_probability=0.5,
    trend_probabilities=(0.2, 0.4, 0.4),
    season_probabilities=(0.0, 0.5, 0.5),
)

laplace = bx.fit(
    y,
    model=model,
    priors=priors,
    engine="laplace",
    parameterization="fruehwirth_schnatter",
    asis=False,
    mcmc=bx.MCMC(draws=1_000, warmup=1_000, chains=4, seed=56_000),
)

pgas = bx.fit(
    y,
    model=model,
    priors=laplace.priors,
    engine="pgas",
    parameterization="fruehwirth_schnatter",
    asis=False,
    init=laplace,
    mcmc=bx.MCMC(draws=1_000, warmup=1_000, chains=4, seed=56_000),
    particles=bx.Particles(n=128, proposal="guided"),
)

laplace_mh = bx.fit(
    y,
    model=model,
    priors=priors,
    engine="laplace_mh",
    parameterization="fruehwirth_schnatter",
    asis=False,
    mcmc=bx.MCMC(draws=1_000, warmup=1_000, chains=4, seed=56_000),
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
timestamp for direct Python execution. Bash/PBS execution adds the selected
configuration name and PBS job ID (or local process ID), preventing concurrent
jobs from colliding. Set `output.run_id` only when a manually chosen ID is
unique. Every run contains `run_config.json`; `output.results_root` and
`output.overwrite` control its location and collision policy.

Simulation fitting runs use `simulations/`, `fits/<scenario>/`,
`tables/<scenario>/`, and `figures/<scenario>/`. Uccle fitting runs use
`fits/<series>/`, `tables/<series>/`, and `figures/<series>/`. The Laplace,
PGAS, and Laplace-MH examples share the same scientific settings and artifact
names; only method-specific diagnostics and warm-start fits differ.

Complete settings live in `examples/config/`, including the six files under
`simulations/` and four files under `uccle/`. Use a selected file or an edited
copy without changing Python:

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
month with `figures.focus_month` in the selected Uccle JSON.

## HPC

Each example has a Bash runner and a matching PBS file. The PBS file requests
resources and receives one JSON path; it does not redefine chains, draws,
warmup, priors, or model settings:

```bash
qsub -N bx_txx_lap \
  -v CONFIG=examples/config/uccle/01_txx.json \
  job_scripts/submit_05_uccle_laplace.pbs
```

`job_scripts/run_parallel_chains.py` reads `mcmc.chains`, launches one process
per chain up to the allocated worker count, and combines successful fits before
creating final figures. Thus four Uccle PBS jobs can each run four parallel
chains, for up to 16 concurrent processes when the scheduler grants all jobs.
The ten numbered Bash files use the same logic interactively:

```bash
bash bash_scripts/run_05_uccle_laplace.sh examples/config/uccle/01_txx.json 4
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
