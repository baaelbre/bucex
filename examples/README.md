# bucex 1.1.0 examples

The first seven standalone files reproduce the complete COMPSTAT analysis. An
eighth diagnostic benchmark makes the classical centered/inverse-gamma
alternative explicit, and two new scripts demonstrate exact Laplace-MH
inference. All ten use the public `bucex` API directly and are
intended to be read as well as run:

1. `00_uccle_record.py` — record from 1892, TXx evolution, and robust LOESS.
2. `01_tail_simulations.py` — matched shape and scale experiments.
3. `02_structural_simulations.py` — six explicit structural models.
4. `03_simulation_laplace.py` — Laplace fits and selection recovery.
5. `04_simulation_pgas.py` — PGAS fits initialized from Laplace.
6. `05_uccle_laplace.py` — Laplace analysis of TXx, TXn, TNx, and TNn.
7. `06_uccle_pgas.py` — PGAS analysis and engine comparison.
8. `07_centered_ig_random_walk_gev.py` — exact-PGAS random-walk GEV with
   centered states, inverse-gamma variance priors, and mandatory mixing
   diagnostics.
9. `08_simulation_laplace_mh.py` — exact Laplace-MH structural simulations.
10. `09_uccle_laplace_mh.py` — exact Laplace-MH analysis of TXx, TXn, TNx,
    and TNn, including acceptance and endpoint-support diagnostics.

The four component-selection fitting examples also save both forms of the level plot, a dedicated
slope plot, posterior seasonality, a posterior predictive check, and an
out-of-sample forecast. The Uccle slope figures use degrees per decade and can
show the conditional fixed-slope posterior beside the stochastic one.
The diagnostic example has no slope or seasonal component; it saves both level
figures, parameter and process-variance traces, ACFs, ESS/R-hat tables, particle
diagnostics, a posterior predictive check, and a forecast.

Run them from the package root. Each script writes to its own directory and
automatically combines a timestamp with the settings that identify that run.
Set one timestamp before a linked local or HPC run so the directories
share the same prefix:

```bash
export BUCEX_RUN_ID=$(date +%Y%m%d_%H%M%S)
python examples/00_uccle_record.py
python examples/01_tail_simulations.py
python examples/02_structural_simulations.py
python examples/03_simulation_laplace.py
python examples/04_simulation_pgas.py
python examples/05_uccle_laplace.py
python examples/06_uccle_pgas.py
python examples/07_centered_ig_random_walk_gev.py
python examples/08_simulation_laplace_mh.py
python examples/09_uccle_laplace_mh.py
```

The resulting layout is
`results/<script>/<BUCEX_RUN_ID>__<automatic-settings-signature>/`.
There is no switch for disabling either part of the run name. Set
`BUCEX_RESULTS_ROOT` to change the root. Every run also contains a complete
`run_config.json`; set `BUCEX_OVERWRITE=1` only when deliberately rerunning an
existing identifier.

Run signatures and repeated figure/table names are deliberately compact. This
keeps the complete path below the legacy Windows directory limit even when the
repository itself is inside a long OneDrive path. The readable identifiers
needed for browsing stay in the folder name; all omitted settings, seeds, and
the mapping from short scenario directories such as `llt` to their full names
are recorded in `run_config.json`.

Every scientific setting is near the top of the relevant script. For HPC use,
each example has a positional-argument runner in `bash_scripts/` and a matching
resource-and-logging submission file in `job_scripts/`. Their direct Bash and
`qsub -v` argument conventions are documented in
[`../docs/HPC_RUNNERS.md`](../docs/HPC_RUNNERS.md). A complete copy-and-paste
submission sequence, including the parallel-chain setup, is in
[`../docs/HPC.md`](../docs/HPC.md).
