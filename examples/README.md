# bucex 1.3.1 examples

The ten numbered files are direct, sequential uses of the public `bucex` API:

1. `00_uccle_record.py` — Uccle record from 1892 and robust LOESS summaries.
2. `01_tail_simulations.py` — matched theoretical GEV shape and scale comparisons.
3. `02_structural_simulations.py` — six explicit structural data-generating models.
4. `03_simulation_laplace.py` — selection recovery with Laplace state updates.
5. `04_simulation_pgas.py` — the same study with Laplace-initialized PGAS.
6. `05_uccle_laplace.py` — Laplace fits of TXx, TXn, TNx, and TNn.
7. `06_uccle_pgas.py` — PGAS fits of the same four series.
8. `07_centered_ig.py` — centered random-walk GEV with inverse-gamma variances.
9. `08_simulation_laplace_mh.py` — exact Laplace-MH simulation fits.
10. `09_uccle_laplace_mh.py` — exact Laplace-MH Uccle fits.

All choices are visible in JSON:

- `config/record.json`: descriptive Uccle figures;
- `config/tail.json`: tail/scale simulations;
- `config/simulation.json`: examples 02, 03, 04, and 08;
- `config/simulations/*.json`: one ready-to-run file for each of the six
  structural simulation truths;
- `config/uccle.json`: examples 05, 06, and 09;
- `config/centered_ig.json`: example 07.

Every script has a visible `DEFAULT_CONFIG_FILE` near the top and calls
`bx.load_config` directly. Change that path for an IDE/notebook run, or leave
the file untouched and select a JSON with `--config PATH` at launch. The
command-line choice takes precedence.

Run a default or edited copy from the repository root:

```bash
python examples/03_simulation_laplace.py
python examples/03_simulation_laplace.py --config examples/config/my_pilot.json
```

To run the six Laplace recovery analyses separately and in presentation order:

```bash
python examples/03_simulation_laplace.py --config examples/config/simulations/01_stationary.json
python examples/03_simulation_laplace.py --config examples/config/simulations/02_linear_trend.json
python examples/03_simulation_laplace.py --config examples/config/simulations/03_random_walk.json
python examples/03_simulation_laplace.py --config examples/config/simulations/04_local_linear_trend.json
python examples/03_simulation_laplace.py --config examples/config/simulations/05_changing_seasonality.json
python examples/03_simulation_laplace.py --config examples/config/simulations/06_llt_fixed_seasonality.json
```

There are no hidden `BUCEX_*` scientific overrides. The run directory is
`output.results_root/<script>/<run-id>__<short-signature>/`; a timestamp is
used when `output.run_id` is `null`. `run_config.json` records the complete
effective specification.

Fit examples save predictor, level, observation-free level, slope, structural
selection, process-SD, GEV-parameter, seasonality, posterior-predictive, and
forecast outputs. Titles are `null` by default. Legends use $y_t$,
$\hat{\mu}_t$, $\hat{\alpha}_t$, $\hat{\beta}_t$, and $\hat{\beta}_0$.
Uccle slopes are in degrees per decade; the fixed-slope median is purple and
has no ribbon.

For PBS execution, see [`../docs/HPC.md`](../docs/HPC.md).
