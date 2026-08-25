# bucex 1.3.2 examples

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

Every script exposes `DEFAULT_CONFIG_FILE`, accepts `--config PATH`, and calls
`bx.load_config` directly. JSON contains the scientific, MCMC, figure, runtime,
and output settings; there are no hidden scientific `BUCEX_*` overrides.

The complete configuration sets are:

- `config/simulations/*.json`: one file for each of six simulation truths;
- `config/uccle/*.json`: one file for each of TXx, TXn, TNx, and TNn;
- `config/simulation.json` and `config/uccle.json`: all-case defaults;
- `config/record.json`, `config/tail.json`, and `config/centered_ig.json`:
  settings for examples 00, 01, and 07.

Run the six Laplace simulation analyses separately:

```bash
python examples/03_simulation_laplace.py --config examples/config/simulations/01_stationary.json
python examples/03_simulation_laplace.py --config examples/config/simulations/02_linear_trend.json
python examples/03_simulation_laplace.py --config examples/config/simulations/03_random_walk.json
python examples/03_simulation_laplace.py --config examples/config/simulations/04_local_linear_trend.json
python examples/03_simulation_laplace.py --config examples/config/simulations/05_changing_seasonality.json
python examples/03_simulation_laplace.py --config examples/config/simulations/06_llt_fixed_seasonality.json
```

Run the four Uccle Laplace analyses separately:

```bash
python examples/05_uccle_laplace.py --config examples/config/uccle/01_txx.json
python examples/05_uccle_laplace.py --config examples/config/uccle/02_txn.json
python examples/05_uccle_laplace.py --config examples/config/uccle/03_tnx.json
python examples/05_uccle_laplace.py --config examples/config/uccle/04_tnn.json
```

Direct Python commands execute the selected example normally. To run four
chains as four concurrent processes, use the matching Bash runner with
`MAX_WORKERS=4`, or submit its PBS file as described in `docs/HPC.md`.

Outputs are stored under
`results/<script>/<run-id>__<settings-signature>/`. Fitting examples save
independent fits, combined fits, tables, and figures. Titles are absent by
default; Uccle slopes are shown in degrees Celsius per decade, and lower-tail
series are returned to their original temperature orientation.
