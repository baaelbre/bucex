# bucex 1.2.1 release notes

Version 1.2.1 is a configuration and figure-cleanup release. It does not
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

`BUCEX_CONFIG=/path/to/file.json` is equivalent. Existing `BUCEX_*`
environment overrides are applied after loading JSON, preserving the PBS
pilot and multi-chain workflows. Each output manifest records the selected
settings file and the resolved values.

## Cleaner examples and figures

- Removed the complete Laplace-sensitivity runner, reducer, grid, PBS jobs,
  and dedicated documentation from the release.
- Removed obsolete launchers for the former long example-07 filename.
- Removed parenthetical inference-engine labels from scientific figure titles.
- Prior-to-posterior process-SD figures now use labelled horizontal axes and
  no panel or figure titles.
- Added regression coverage for shared JSON settings, absent sensitivity
  artifacts, engine-neutral titles, and title-free process-SD panels.

## Inherited guarantees

Version 1.2.1 retains phase-specific and seasonally adjusted trajectories,
automatic conjugate centred inverse-gamma process-variance updates,
deterministic finite-endpoint repair for Laplace-MH proposals, singular affine
state-transition handling, conditional PGAS ancestor sampling, and the
24-task scenario-chain PBS workflow for example 08.
