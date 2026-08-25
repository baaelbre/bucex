# Example settings

The numbered fitting examples use three authoritative JSON files:

- `simulation.json`: examples 02, 03, 04, and 08;
- `uccle.json`: examples 05, 06, and 09;
- `centered_ig.json`: example 07.

Edit a copy and pass it directly:

```bash
python examples/03_simulation_laplace.py --config my_simulation.json
```

For PBS, pass the same file as `CONFIG`:

```bash
qsub -v CONFIG=examples/config/my_simulation.json \
  job_scripts/submit_03_simulation_laplace.pbs
```

There are no environment-variable overrides of scientific settings. Copy a
JSON file for a pilot or sensitivity run and edit that copy. Every output
`run_config.json` records the selected file and all effective values.
