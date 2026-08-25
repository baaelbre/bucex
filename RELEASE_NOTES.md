# bucex 1.3.0 release notes

Version 1.3.0 is a configuration and figure-cleanup release. It does not
change the state equations, Laplace-MH target, PGAS target, public fitting API,
or archive schema.

## Authoritative JSON settings

The numbered examples remain sequential, but duplicated blocks of scientific
settings have moved to three human-readable files:

- `examples/config/simulation.json` is shared by examples 02, 03, 04, and 08;
- `examples/config/uccle.json` is shared by examples 05, 06, and 09;
- `examples/config/centered_ig.json` controls example 07.

This fixes the configuration drift that had developed between the nominally
matched inference scripts. The canonical simulation prior uses the broad
baseline slabs retained after the sensitivity exercise: 0.1 for level,
0.0008 for slope, and 0.07 for seasonality. The Uccle configuration uses the
documented monthly calibration 0.03 / 0.0001 / 0.05 and equal prior odds for
fixed versus dynamic slope and season.

Pass a custom file without editing Python:

```bash
python examples/03_simulation_laplace.py --config my_simulation.json
```

PBS files receive the same path through `qsub -v CONFIG=...`. Scientific and
sampling values are never silently overridden by environment variables. A
small generic HPC runner fans independent chains across allocated cores and
combines them through the examples' existing public fit-combination API. Each
output manifest records the selected settings file and resolved values.

## Cleaner examples and figures

- Removed the complete Laplace-sensitivity runner, reducer, grid, PBS jobs,
  and dedicated documentation from the release.
- Removed obsolete launchers for the former long example-07 filename.
- Removed parenthetical inference-engine labels from scientific figure titles.
- Prior-to-posterior process-SD figures no longer say "analytic half-normal".
- Level figures are saved with and without seasonally adjusted observations.
- Seasonal plots contain only the seasonal effect; fixed-slope overlays are
  purple median lines without an uncertainty ribbon.
- Plot labels use mathematical notation such as $y_t$, $\hat{\mu}_t$,
  $\hat{\alpha}_t$, $\hat{\beta}_t$, and $\hat{\beta}_0$.
- Posterior predictive checks and forecasts are generated through the public
  `FitResult` API.
- Added regression coverage for shared JSON settings, absent sensitivity
  artifacts, engine-neutral titles, and title-free process-SD panels.

## Inherited guarantees

Version 1.3.0 retains phase-specific and seasonally adjusted trajectories,
automatic conjugate centred inverse-gamma process-variance updates,
deterministic finite-endpoint repair for Laplace-MH proposals, singular affine
state-transition handling, conditional PGAS ancestor sampling, and the
JSON-driven parallel-chain PBS workflow for all fitting examples.
