# bucex 1.3.2 release notes

Version 1.3.2 completes the JSON-driven local and PBS workflow. It does not
change the state equations, Laplace, Laplace-MH, or PGAS targets, the public
fitting API, or the result-archive schema.

## Four calibrated Uccle configurations

The release adds one complete input file per temperature-extreme series:

- `examples/config/uccle/01_txx.json`;
- `examples/config/uccle/02_txn.json`;
- `examples/config/uccle/03_tnx.json`;
- `examples/config/uccle/04_tnn.json`.

The files differ only by selected series. Each contains the location-varying
GEV structure, data-derived initial level centre, calibrated priors, inference
controls, prediction and figure settings, and output policy. The primary slabs
are `(0.02, 0.00005, 0.02)` for level, slope, and season; the fixed-slope prior
is centred at zero with SD `0.0025` degrees Celsius per month; structural odds
are `(0.20, 0.40, 0.40)` for absent/fixed/dynamic slope and
`(0.00, 0.50, 0.50)` for absent/fixed/dynamic seasonality.

The all-series `examples/config/uccle.json` carries the same calibrated
settings. Examples 05, 06, and 09 now construct the supported model modes and
initial-level rule directly from the selected JSON.

## Production MCMC defaults

The four Uccle presets and all six simulation presets now use 1,000 warmup
iterations, 1,000 retained draws, and four independent chains. Scenario-
specific simulation and prior values from the proven runs are unchanged.

## Collision-safe parallel jobs

The generic runner now lives at `job_scripts/run_parallel_chains.py`; the
separate `hpc/` directory has been removed. With `output.run_id: null`, every
Bash/PBS invocation receives a readable ID containing the timestamp,
configuration filename, and PBS job ID or local process ID. Simultaneous
series or scenario jobs therefore cannot share outputs or per-chain logs.

Temporary chain JSONs retain draws and warmup, set only one chain, offset the
seed, and select chain-only output. They also record the original selected
JSON, so `run_config.json` no longer points to a temporary configuration that
disappears after the job. The source JSON remains untouched.

The four Uccle series may be submitted as four separate four-core PBS jobs.
Each job runs its four chains concurrently; when the scheduler grants every
job, all 16 series-chain fits run in parallel and are combined within their own
series job.

## Documentation and compatibility

The main README and the configuration, Uccle, architecture, validation, and
HPC guides now describe the same JSON-only setting contract and include the
complete simulation and Uccle submission commands. Existing custom schema-1
JSON files remain valid for non-Uccle examples. Older Uccle JSON files need the
new explicit `model` section and `priors.alpha_mean` field when used with
examples 05, 06, or 09.
