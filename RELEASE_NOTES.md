# bucex 1.0.0 release notes

Version 1.0.0 is the first stable release from the clean `bucex` repository.
It is the direct successor of the former 2.6.2 package: the modelling and
inference APIs remain compatible, while the repository, examples, plots, and
HPC entry points have been consolidated for a reproducible full release.

## Seven direct-API analyses

- Exactly seven readable Python scripts live directly under `examples/`.
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

## Component inference and figures

- `fit.plot("level")` now produces a dedicated latent-level plot, using
  seasonally adjusted observations when a seasonal state is present.
- `fit.plot("slope")` now produces a dedicated latent-slope plot on the proper
  change-per-interval scale; raw temperature observations are not drawn on
  this axis.
- All four fitting examples save separate `level` and `slope` figures in
  addition to the complete predictor, selection, process-SD, GEV, seasonal,
  endpoint, and optional diagnostic figures.
- `fit.plot("season")` overlays the phase-specific trajectories
  \(\mu_{ij}+\gamma_{ij}\) so changes in the seasonal pattern are visible.

## PBS/Torque execution

- Normal runners are in `bash_scripts/run_*.sh`; scheduler files are in
  `job_scripts/submit_*.pbs`.
- Every one of the seven analyses has its own Bash/PBS pair.
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
  is `1.0.0`.
- Existing safe-fit files remain readable. The persistence schema stays at
  2.6.2 deliberately, because package semantic versioning and archive schema
  compatibility are separate contracts.
- Generated results, logs, PDFs, fit archives, caches, environments, and build
  products are excluded by the root `.gitignore` so a fresh repository cannot
  accidentally acquire large posterior objects.
- Source distributions include the examples, Bash runners, PBS files,
  sensitivity configuration, validation scripts, and documentation.
