# Example settings

The numbered fitting examples use three authoritative JSON files:

- `simulation.json`: examples 02, 03, 04, and 08;
- `uccle.json`: examples 05, 06, and 09;
- `centered_ig.json`: example 07.

Edit a copy and pass it directly:

```bash
python examples/03_simulation_laplace.py --config my_simulation.json
```

`BUCEX_CONFIG=/path/to/file.json` is equivalent and is convenient in PBS
jobs. Existing `BUCEX_*` variables are applied after the JSON file, so short
pilot runs can still override such fields as `BUCEX_N_TIME`, `BUCEX_DRAWS`,
`BUCEX_WARMUP`, `BUCEX_CHAINS`, and `BUCEX_PARTICLES` without changing the
scientific baseline. Every output `run_config.json` records the resolved
settings file and all effective values.
