# Example settings

The numbered fitting examples use three authoritative JSON files:

- `simulation.json`: examples 02, 03, 04, and 08;
- `uccle.json`: examples 05, 06, and 09;
- `centered_ig.json`: example 07.

The `simulations/` directory contains six complete one-scenario presets for
example 03 (and the matching examples 02, 04, and 08):

- `01_stationary.json`;
- `02_linear_trend.json`;
- `03_random_walk.json`;
- `04_local_linear_trend.json`;
- `05_changing_seasonality.json`;
- `06_llt_fixed_seasonality.json`.

Each is an input configuration, not a historical output `run_config.json`.
Only one scenario key is active, and `output.run_id` is `null`, so separate
timestamped result directories are created automatically.

Each numbered Python example exposes its default path as
`DEFAULT_CONFIG_FILE` and then calls `bx.load_config` directly. You can edit
that one path or override it with `--config PATH`; no configuration helper
module or environment-variable layer sits between the JSON and the example.

Edit a copy and pass it directly:

```bash
python examples/03_simulation_laplace.py --config my_simulation.json
```

For example:

```bash
python examples/03_simulation_laplace.py \
  --config examples/config/simulations/01_stationary.json
```

For PBS, pass the same file as `CONFIG`:

```bash
qsub -v CONFIG=examples/config/my_simulation.json \
  job_scripts/submit_03_simulation_laplace.pbs
```

There are no environment-variable overrides of scientific settings. Copy a
JSON file for a pilot or sensitivity run and edit that copy. Every output
`run_config.json` records the selected file and all effective values.
