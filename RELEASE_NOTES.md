# bucex 1.0.1 release notes

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
