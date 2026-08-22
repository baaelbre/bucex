# bucex 1.1.0

`bucex` fits Bayesian unobserved-components models to Gaussian and generalized
extreme-value observations. The package combines a declarative structural
model API, componentwise SSVS, approximate Laplace state updates, exact
Laplace independence-MH updates, and exact-density PGAS state updates.

Version 1.1.0 adds a full-trajectory `laplace_mh` engine to the stable 1.0.1
codebase. The interface now includes ten transparent analysis scripts:

1. the complete Uccle record from 1892 and the evolution of TXx;
2. matched GEV shape and scale simulations;
3. six explicit structural simulations;
4. selection recovery with Laplace state updates;
5. the same recovery experiment with Laplace-initialized PGAS;
6. Laplace analysis of TXx, TXn, TNx, and TNn;
7. PGAS analysis of the same four series.
8. an exact-PGAS random-walk GEV benchmark with centered states and explicit
   inverse-gamma priors on the process and observation variances.
9. exact Laplace-MH fits for representative structural simulations;
10. exact Laplace-MH fits of Uccle TXx, TXn, TNx, and TNn.

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
    beta_mean=0.0,
    beta_sd=0.0015,
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
predictor medians.

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
simulation. GEV endpoint violations are ordinary MH rejections; proposals are
not truncated and never fall back to an atom at the mode. The returned plan is
therefore labelled exact-invariant. Inspect state acceptance, support
rejections, Laplace convergence, and ESS per second. `Laplace(mh_steps=...)`
controls repeated full-path proposals at fixed parameters.

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
`examples/07_centered_ig_random_walk_gev.py` instead uses a random-walk level,
`parameterization="centered"`, `asis=False`, and `InverseGammaVariance`
priors. It fits with exact-density PGAS and always exports traces, ACFs,
ESS/R-hat, process-variance summaries, and particle diagnostics. Its purpose is
diagnostic comparison with the FS/non-centred specification, not to replace
the latter as the default structural-selection model.

## Ten standalone examples

Set one run ID to give all ten script-specific result directories the same
timestamp prefix:

```bash
export BUCEX_RUN_ID=$(date +%Y%m%d_%H%M%S)
python examples/00_uccle_record.py
python examples/01_tail_simulations.py
python examples/02_structural_simulations.py
python examples/03_simulation_laplace.py
python examples/04_simulation_pgas.py
python examples/05_uccle_laplace.py
python examples/06_uccle_pgas.py
python examples/07_centered_ig_random_walk_gev.py
python examples/08_simulation_laplace_mh.py
python examples/09_uccle_laplace_mh.py
```

Outputs are written below
`results/<script>/<BUCEX_RUN_ID>__<automatic-settings-signature>/`. Every run
contains `run_config.json`. Change the root with `BUCEX_RESULTS_ROOT`, or set
`BUCEX_OVERWRITE=1` to deliberately regenerate an existing identifier.

All scientific settings remain near the top of each script. MCMC controls can
also be overridden with `BUCEX_DRAWS`, `BUCEX_WARMUP`, `BUCEX_CHAINS`,
`BUCEX_PARTICLES`, and `BUCEX_SEED`.

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
ten-year forecasts by default.

## HPC

Each Python example has a normal Bash runner and a matching PBS submission
file. The PBS file handles resources and logging and then calls the runner:

```bash
qsub job_scripts/submit_00_uccle_record.pbs

qsub -v DRAWS=2000,WARMUP=2000,CHAINS=4,PARTICLES=512 \
  job_scripts/submit_06_uccle_pgas.pbs
```

The ten numbered `run_*.sh` files under `bash_scripts/` also run directly with
positional arguments. There is no hidden scientific settings layer; every
pair is readable by itself. See `docs/HPC_RUNNERS.md` for the exact argument
order and adapt the resource directives to the local cluster. `docs/HPC.md`
gives complete pilot and final submission commands, while
`docs/LAPLACE_SENSITIVITY_HPC.md` documents the seed/slab array study.

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
