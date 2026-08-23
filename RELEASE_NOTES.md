# bucex 1.1.4 release notes

Version 1.1.4 is the HPC orchestration release for the exact Laplace-MH
simulation study. It keeps the public modelling and inference APIs, exact
kernel, and archive schema unchanged.

## Scenario-chain PBS array

The former example-08 PBS runner parallelized four chains, but every chain
still fitted all six scenarios sequentially. Version 1.1.4 maps the Cartesian
product directly onto the scheduler:

```text
6 scenarios x 4 chains = 24 independent one-core tasks
```

Array IDs 1--4 fit the stationary scenario, 5--8 the linear trend, 9--12 the
random walk, 13--16 the local linear trend, 17--20 dynamic seasonality, and
21--24 the local linear trend with fixed seasonality. A dependent finalizer
runs only after every task succeeds.

Submit the complete workflow from the repository root with:

```bash
bash bash_scripts/qsub_08_simulation_laplace_mh.sh
```

The final HPC defaults are `N_TIME=1000`, `DRAWS=1000`, `WARMUP=1000`,
`CHAINS=4`, `MH_STEPS=1`, and `MAX_CONCURRENT=24`. Each fit task requests one
core, 12 GB, and six hours; the finalizer requests one core, 16 GB, and one
hour. These resource limits target a roughly four-to-five-hour slowest fit on
the measured workload, excluding scheduler waiting time.

## One shared, collision-free result

All workers and the finalizer use
`results/08_simulation_laplace_mh/<RUN_ID>__<signature>/`. Workers write only
their unique paths below `tasks/chainXX/`; no two tasks write the same fit,
manifest, log, simulation, table, or figure. Once every requested fit and
manifest exists, the finalizer combines compatible chains and creates the
usual `fits/`, `simulations/`, `tables/`, and `figures/` trees in that same
directory. A failed task blocks finalization through the PBS dependency.

The internal worker imports example 08's scenario definitions, simulation
function, SSVS prior factory, and exact fit function. There is therefore one
scientific specification for local, interactive, and PBS-array execution.
`BUCEX_SCENARIO_KEYS` supports colon-separated subsets without changing the
global scenario-specific seed offsets.

## Inherited 1.1.3 numerical-safety patch

Version 1.1.3 made exact FS Laplace-MH robust for finite-endpoint GEV
likelihoods. All of those guarantees remain in 1.1.4.

## Finite-endpoint Laplace-MH initialization

For negative shape, a valid GEV trajectory must satisfy

```text
1 + xi * (y_t - eta_t) / sigma > 0.
```

The 1.1.0--1.1.2 independence proposal began its deterministic mode search at
the zero non-centred trajectory. A current full trajectory could be valid while
that zero baseline crossed the endpoint, causing proposal construction to fail
before an MH draw was made. Repeating the same deterministic calculation did
not resolve the failure.

Version 1.1.3 constructs a deterministic support-feasible FS path whenever the
zero path is invalid. It uses a contemporaneous level or seasonal innovation
when available and otherwise repairs an integrated stochastic slope one step
ahead. The initializer depends only on observations and static parameters, not
on the current MCMC trajectory, so the existing exact independence-MH
correction remains valid. Singular transition coordinates are still projected
onto their exact affine recursions without jitter.

## Exact-sampler failure contract

- Exact FS Laplace-MH and PGAS iterations are attempted once.
- An unrecoverable numerical exception stops the fit before a restored or
  duplicate posterior draw can be recorded.
- The former 25-attempt loop remains only for the explicitly approximate
  Laplace path where its historical recovery behavior is part of that method.
- Laplace diagnostics now report `initial_support_repair_rate`; the underlying
  per-draw indicator is retained in the fit archive.
- Regression tests reproduce the negative-shape integrated-slope support case
  and verify the fail-fast contract.

## 1.1.2 example-consistency patch

Version 1.1.2 made the paired Laplace and Laplace-MH examples scientifically
and operationally consistent. Its standalone layout remains unchanged;
1.1.4 adds the optional scenario-chain array around the same example.

## Controlled Laplace versus Laplace-MH examples

- `08_simulation_laplace_mh.py` now uses exactly the six structural truths,
  record length, GEV parameters, fitted model, calibrated SSVS prior, seeds,
  MCMC defaults, summaries, and figure suite of `03_simulation_laplace.py`.
- `09_uccle_laplace_mh.py` now uses exactly the data window, model, calibrated
  prior, structural odds, MCMC defaults, summaries, and figure suite of
  `05_uccle_laplace.py`.
- The only inferential differences in each pair are `engine="laplace_mh"`,
  `Laplace(mh_steps=...)`, exactness metadata, and MH acceptance/support
  diagnostics.
- Simulation fits now consistently write `simulations/`, `fits/`, `tables/`,
  and `figures/`; Uccle fits consistently write `fits/`, `tables/`, and
  `figures/`. Both PNG and PDF figures are retained.
- The Laplace-MH Bash/PBS launchers now run independent chains concurrently
  and combine compatible fits exactly like their Laplace counterparts.
- Temporary local-debug overrides in example 03 were removed.
- New parity tests prevent future drift in settings, model declarations,
  priors, output contracts, and runner defaults.

## 1.1.1 centered-inference patch

Version 1.1.1 added automatic centered inverse-gamma process-variance updates.
It did not change or optimize the Laplace-MH implementation introduced in
1.1.0.

### Conjugate centered process-variance updates

- A process component with `InverseGammaVariance` is now updated from its
  exact inverse-gamma full conditional whenever
  `parameterization="centered"`.
- The dispatch is automatic from the existing prior object; no sampler flag or
  new public prior type is required.
- The conjugate update applies to Gaussian state-transition variances
  conditional on the complete latent path. An inverse-gamma prior on the GEV
  observation scale remains nonconjugate and retains its log-scale MH update.
- Disturbance/non-centred scale updates remain MH. With ASIS, bucex records the
  method used by each centered and non-centred sweep separately.
- `FitResult` diagnostics now report parameter update methods. Gibbs and fixed
  updates are omitted from MH acceptance summaries instead of receiving a
  misleading acceptance rate of one or zero.

### Centered inverse-gamma benchmark

- `07_centered_ig_random_walk_gev.py` now uses `engine="laplace"` by default,
  matching its purpose as a fast qualitative demonstration of centered/IG
  mixing.
- Set `BUCEX_ENGINE=laplace_mh` for exact Laplace-MH validation or
  `BUCEX_ENGINE=pgas` for the exact particle benchmark. The model, priors,
  diagnostics, and saved outputs remain the same across engines.
- Example, Bash, PBS, and run metadata include the selected engine. Laplace-MH
  performance code is unchanged in this release.

## 1.1.0 Laplace-MH release

Version 1.1.0 adds exact-invariant, full-trajectory Laplace independence-MH
inference for univariate non-Gaussian state-space models.

### New `laplace_mh` engine

- `bx.fit(..., engine="laplace_mh")` is available for univariate GEV models
  under the FS, centered, and disturbance parameterizations.
- The iterated-Laplace smoother is only a proposal. The exact GEV likelihood
  corrects every full-path draw by independence Metropolis--Hastings, and the
  returned `InferencePlan` is labelled exact-invariant.
- Proposal construction is deterministic in observations and static
  parameters. It cannot accidentally depend on the current state through a
  finite-iteration mode warm start.
- The exact and Gaussian proposal state measures are identical, so singular
  transition densities cancel from the acceptance ratio.
- Integrated-slope and dummy-seasonal lag coordinates are projected onto their
  exact affine recursions after numerical Gaussian simulation. No artificial
  process noise is introduced.
- Endpoint-invalid GEV proposals are normal MH rejections. The exact kernel
  does not rejection-sample a truncated proposal and never falls back to an
  atom at the Laplace mode.
- `bx.Laplace(mh_steps=...)` controls repeated whole-trajectory proposals while
  reusing a cached Gaussian forward filter in the FS implementation.

### Diagnostics and API

- `fit.diagnostics()["engine"]` reports state acceptance, proposal support
  rejections, Laplace convergence, iteration count, and relative mode change.
- The raw per-draw diagnostics and per-chain `state_laplace_mh` acceptance are
  retained in `FitResult` archives.
- Low-level proposal construction, drawing, correction-weight evaluation, and
  the general `laplace_mh` kernel are exported for research use.
- Exact SSVS parameter/model updates are used with `laplace_mh`; the older
  pseudo-observation SSVS update remains confined to approximate `laplace`.
- Hierarchical `MultiSeriesModel` fitting rejects `laplace_mh` in this release
  until the shared hierarchy receives a matching exact correction.

### Examples and validation

- Added `08_simulation_laplace_mh.py` and `09_uccle_laplace_mh.py`, plus Bash
  runners and PBS submission files.
- Added a quadratic-observation identity test: Laplace-MH acceptance is one and
  correction weights are constant up to floating-point precision.
- Added exact deterministic-support, negative-shape GEV, public API, SSVS, and
  diagnostics regression tests.
- Package version, distributions, source archive, documentation, and example
  inventories are now consistently `1.1.0`.

## 1.0.1 baseline

Version 1.0.1 is the first refinement of the clean stable `bucex` release. It
continues the former 2.6.2 research code, keeps its modelling and inference
contracts, and makes the fitted-component and predictive output easier to use
in the Uccle analysis and presentation.

## Eight direct-API examples

- Seven readable analysis scripts and one centered/inverse-gamma diagnostic
  benchmark live directly under `examples/`.
- Each script declares its scientific settings near the top and constructs
  models explicitly with the public `bucex` API.
- Results use compact, Windows-safe directories under
  `results/<script>/<run-id>__<settings-signature>/` and always retain the run
  signature and complete `run_config.json` metadata.
- Uccle descriptive figures begin in 1892 and use robust local-linear LOESS.
- Tail comparisons plot the actual theoretical GEV densities relative to
  location, including finite endpoints for negative shape.
- Shape and scale comparison groups share plotting ranges, and structural
  simulations remain one time series per figure.
- The eighth script fits the random-walk GEV with exact-density PGAS,
  `parameterization="centered"`, `asis=False`, and explicit inverse-gamma
  priors on the level-innovation and observation variances. It always exports
  parameter traces, process-variance traces, ACFs, ESS/R-hat, and particle
  diagnostics.

## Component inference and figures

- `fit.plot("level")` now produces a dedicated latent-level plot, using
  seasonally adjusted observations when a seasonal state is present.
- `fit.plot("level", show_observed=False)` produces the same posterior level
  without adjusted observations; every fitting example saves both versions.
- `fit.plot("slope")` now produces a dedicated latent-slope plot on the proper
  change-per-interval scale; raw temperature observations are not drawn on
  this axis.
- Uccle slope figures report degrees per decade. They can condition on the
  dynamic-slope class and add the conditional fixed-slope posterior and its
  uncertainty as dashed lines.
- All four fitting examples save separate `level` and `slope` figures in
  addition to the complete predictor, selection, process-SD, GEV, seasonal,
  endpoint, and optional diagnostic figures.
- `fit.plot("season")` overlays the phase-specific seasonal effects
  \(\gamma_{ij}\), without adding the level, so changes in the seasonal pattern
  are visible directly.
- Process-SD and structural-selection labels have been shortened for cleaner
  manuscript and presentation figures.

## Predictive output

- `fit.posterior_predictive()` samples replicated observations at the fitted
  time points using joint retained state and parameter draws.
- `fit.forecast()` propagates structural states and observation uncertainty
  beyond the end of the record.
- Both return a common public object with tidy tables and plotting methods.
  The four fitting examples save posterior predictive checks, forecasts, and
  their corresponding CSV files.

## PBS/Torque execution

- Normal runners are in `bash_scripts/run_*.sh`; scheduler files are in
  `job_scripts/submit_*.pbs`.
- Every numbered example has its own Bash/PBS pair.
- The fitting runners execute independent chains concurrently, constrain BLAS
  threads, wait for all chains, and combine compatible fits before producing
  final tables and figures.
- Runners use the configured virtual-environment interpreter directly and
  verify `bucex` and Matplotlib before starting.
- A separate PBS array workflow spans simulation seeds, prior slab profiles,
  and chains; its grid is editable in `config/laplace_sensitivity_grid.sh`.
- `docs/HPC.md`, `docs/HPC_RUNNERS.md`, and
  `docs/LAPLACE_SENSITIVITY_HPC.md` contain setup, argument, submission,
  dependency, monitoring, and final-run commands.

## Compatibility and repository hygiene

- The import name and distribution name are both `bucex`; the package version
  is `1.0.1`.
- Existing safe-fit files remain readable. The persistence schema stays at
  2.6.2 deliberately, because package semantic versioning and archive schema
  compatibility are separate contracts.
- Generated results, logs, PDFs, fit archives, caches, environments, and build
  products are excluded by the root `.gitignore` so a fresh repository cannot
  accidentally acquire large posterior objects.
- Source distributions include the examples, Bash runners, PBS files,
  sensitivity configuration, validation scripts, and documentation.
