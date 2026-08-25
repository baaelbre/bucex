# bucex 1.3.1 release notes

Version 1.3.1 is a simulation-configuration release. It does not change the
state equations, Laplace, Laplace-MH, or PGAS targets, the public fitting API,
or the result-archive schema.

## Six one-scenario configurations

The release adds one complete schema-1 JSON input for each structural truth:

- `examples/config/simulations/01_stationary.json`;
- `examples/config/simulations/02_linear_trend.json`;
- `examples/config/simulations/03_random_walk.json`;
- `examples/config/simulations/04_local_linear_trend.json`;
- `examples/config/simulations/05_changing_seasonality.json`;
- `examples/config/simulations/06_llt_fixed_seasonality.json`.

Each file activates exactly one of the stable scenario keys used by examples
02, 03, 04, and 08. The six files are ordinary inputs selected with
`--config`; they are not output `run_config.json` manifests. Their
`output.run_id` values are `null`, so local runs automatically receive
separate timestamped directories.

Every numbered example exposes a `DEFAULT_CONFIG_FILE` next to its imports and
calls `bx.load_config` directly. The path can be changed in the script for an
IDE/notebook run, while `--config PATH` overrides it at launch. The former
`examples/_example_config.py` wrapper module has been removed.

The successful supplied manifests were converted to the current input schema.
The recorded scenario identity is authoritative: the supplied file named
`run_config_stationary.json` contains a random-walk-only retry and therefore
provides the random-walk settings. The later joint stationary/linear run
provides those two presets, while the all-six baseline provides the local
linear trend and seasonal presets.

Run the cases separately from the repository root:

```bash
python examples/03_simulation_laplace.py --config examples/config/simulations/01_stationary.json
python examples/03_simulation_laplace.py --config examples/config/simulations/02_linear_trend.json
python examples/03_simulation_laplace.py --config examples/config/simulations/03_random_walk.json
python examples/03_simulation_laplace.py --config examples/config/simulations/04_local_linear_trend.json
python examples/03_simulation_laplace.py --config examples/config/simulations/05_changing_seasonality.json
python examples/03_simulation_laplace.py --config examples/config/simulations/06_llt_fixed_seasonality.json
```

The same files can be passed to example 02 for simulation-only figures,
example 04 for PGAS, or example 08 for exact Laplace-MH.

## Backward compatibility

The existing `examples/config/simulation.json` remains the shared all-scenario
default. Version 1.3.1 retains phase-specific and seasonally adjusted
trajectories, automatic conjugate centred inverse-gamma process-variance
updates, deterministic finite-endpoint repair for Laplace-MH proposals,
singular affine state-transition handling, conditional PGAS ancestor sampling,
and the JSON-driven parallel-chain PBS workflow for all fitting examples.
