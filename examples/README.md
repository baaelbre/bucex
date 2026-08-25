# bucex 1.2.1 examples

The first seven numbered files reproduce the complete COMPSTAT analysis. An
eighth diagnostic benchmark makes the classical centered/inverse-gamma
alternative explicit, and the final two scripts demonstrate exact Laplace-MH
inference. All ten use the public `bucex` API directly. Their settings are in
three short JSON files under `examples/config/`, so the numbered Python files
stay sequential and readable without duplicating scientific defaults:

1. `00_uccle_record.py` — record from 1892, TXx evolution, and robust LOESS.
2. `01_tail_simulations.py` — matched shape and scale experiments.
3. `02_structural_simulations.py` — six explicit structural models.
4. `03_simulation_laplace.py` — Laplace fits and selection recovery.
5. `04_simulation_pgas.py` — PGAS fits initialized from Laplace.
6. `05_uccle_laplace.py` — Laplace analysis of TXx, TXn, TNx, and TNn.
7. `06_uccle_pgas.py` — PGAS analysis and engine comparison.
8. `07_centered_ig.py` — centered random-walk GEV with
   conjugate inverse-gamma process-variance updates, a fast Laplace default,
   optional exact Laplace-MH/PGAS validation, and mandatory mixing diagnostics.
9. `08_simulation_laplace_mh.py` — exact Laplace-MH analysis of the same six
   simulations, priors, tables, and figures as example 03. On PBS,
   `_08_simulation_laplace_mh_task.py` is the internal one-scenario/one-chain
   worker used by the 24-task array; it is not a separate scientific example.
10. `09_uccle_laplace_mh.py` — exact Laplace-MH analysis of TXx, TXn, TNx,
    and TNn under the same model, calibrated priors, tables, and figures as
    example 05, with acceptance and endpoint-support diagnostics.

The six component-selection fitting examples save both forms of the level
plot, a dedicated slope plot, posterior seasonality, a posterior predictive
check, and an out-of-sample forecast. They also save one phase-specific
trajectory/forecast and a seasonally adjusted latent-level forecast. The
simulation phase defaults to 1; the Uccle calendar month defaults to July.
The Uccle slope figures use degrees per decade and can show the conditional
fixed-slope posterior beside the stochastic one.
The diagnostic example has no slope or seasonal component; it saves both level
figures, parameter and process-variance traces, ACFs, ESS/R-hat tables,
engine diagnostics, a posterior predictive check, and a forecast.

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
python examples/07_centered_ig.py
python examples/08_simulation_laplace_mh.py
python examples/09_uccle_laplace_mh.py
```

The resulting layout is
`results/<script>/<BUCEX_RUN_ID>__<automatic-settings-signature>/`.
There is no switch for disabling either part of the run name. Set
`BUCEX_RESULTS_ROOT` to change the root. Every run also contains a complete
`run_config.json`; set `BUCEX_OVERWRITE=1` only when deliberately rerunning an
existing identifier.

Simulation-fitting examples store artifacts under `simulations/`,
`fits/<scenario>/`, `tables/<scenario>/`, and `figures/<scenario>/`. Uccle
fitting examples use `fits/<series>/`, `tables/<series>/`, and
`figures/<series>/`. The Laplace, PGAS, and Laplace-MH scripts intentionally
share those layouts and all scientific defaults. The shared JSON source makes
that equality structural, and release tests compare the resolved contracts.

Run signatures and repeated figure/table names are deliberately compact. This
keeps the complete path below the legacy Windows directory limit even when the
repository itself is inside a long OneDrive path. The readable identifiers
needed for browsing stay in the folder name; all omitted settings, seeds, and
the mapping from short scenario directories such as `llt` to their full names
are recorded in `run_config.json`.

The authoritative settings are `config/simulation.json`, `config/uccle.json`,
and `config/centered_ig.json`. Pass a custom file with `--config PATH` or
`BUCEX_CONFIG=PATH`. For HPC use, each example has a positional-argument runner
in `bash_scripts/` and a matching resource-and-logging submission file in
`job_scripts/`. Their direct Bash and `qsub -v` argument conventions are documented in
[`../docs/HPC_RUNNERS.md`](../docs/HPC_RUNNERS.md). A complete copy-and-paste
submission sequence, including the parallel-chain setup, is in
[`../docs/HPC.md`](../docs/HPC.md).

For the final exact simulation run, submit the scenario-chain array and its
dependent finalizer with one command:

```bash
bash bash_scripts/qsub_08_simulation_laplace_mh.sh
```

The array workers and finalizer share one output directory. Workers only write
unique `tasks/chainXX/fits/<scenario>/` and `tasks/chainXX/manifests/` paths;
the finalizer writes the combined `fits/`, `tables/`, `figures/`, and
`simulations/` artifacts after all tasks succeed.
