# Running bucex 1.3.2 with PBS

## What controls a run

The selected JSON file is the complete scientific and computational
specification. The PBS file controls only requested resources, logging, and
which Bash runner is launched.

| Concern | Controlled by |
|---|---|
| data, scenario, model, priors | selected JSON |
| draws, warmup, chains, seed | selected JSON |
| particles and Laplace-MH steps | selected JSON |
| figures, predictions, output policy | selected JSON |
| walltime, cores, memory | PBS directives |
| concurrent chain processes and final combination | `job_scripts/run_parallel_chains.py` |

The PBS files do not overwrite draws, warmup, chains, priors, or model
settings. For a four-chain fit, the runner temporarily partitions the JSON
into four one-chain copies. It changes only `mcmc.chains` from 4 to 1 in each
copy, offsets the four seeds, enables `runtime.chain_only`, and assigns separate
output IDs. Draws and warmup remain unchanged. The source JSON is never edited.

## One-time setup

From the repository root on the cluster:

```bash
module load Python/3.12.3-GCCcore-13.3.0
python -m venv "$HOME/venvs/bucex_env"
source "$HOME/venvs/bucex_env/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e ".[plot]"
python -c "import bucex, matplotlib; print(bucex.__version__)"
```

The printed version should be `1.3.2`. If the environment is elsewhere, set
its location before submitting jobs:

```bash
export BUCEX_VENV_DIR=/absolute/path/to/bucex_env
```

## Six simulation jobs

The six production presets use 1,000 warmup iterations, 1,000 retained draws,
and four chains. Submit the Laplace analyses independently:

```bash
qsub -N bx_s01_lap -v CONFIG=examples/config/simulations/01_stationary.json job_scripts/submit_03_simulation_laplace.pbs
qsub -N bx_s02_lap -v CONFIG=examples/config/simulations/02_linear_trend.json job_scripts/submit_03_simulation_laplace.pbs
qsub -N bx_s03_lap -v CONFIG=examples/config/simulations/03_random_walk.json job_scripts/submit_03_simulation_laplace.pbs
qsub -N bx_s04_lap -v CONFIG=examples/config/simulations/04_local_linear_trend.json job_scripts/submit_03_simulation_laplace.pbs
qsub -N bx_s05_lap -v CONFIG=examples/config/simulations/05_changing_seasonality.json job_scripts/submit_03_simulation_laplace.pbs
qsub -N bx_s06_lap -v CONFIG=examples/config/simulations/06_llt_fixed_seasonality.json job_scripts/submit_03_simulation_laplace.pbs
```

For exact Laplace-MH fits, use the same six JSON paths with
`job_scripts/submit_08_simulation_laplace_mh.pbs`. For PGAS, use
`job_scripts/submit_04_simulation_pgas.pbs`. This changes the inference script,
not the selected simulation, prior, or MCMC settings.

## Four Uccle jobs

Each Uccle JSON selects one series and requests four chains. Submit all four
commands immediately; PBS may run the jobs simultaneously or queue them
according to your allocation.

Fast Laplace tuning fits:

```bash
qsub -N bx_txx_lap -v CONFIG=examples/config/uccle/01_txx.json job_scripts/submit_05_uccle_laplace.pbs
qsub -N bx_txn_lap -v CONFIG=examples/config/uccle/02_txn.json job_scripts/submit_05_uccle_laplace.pbs
qsub -N bx_tnx_lap -v CONFIG=examples/config/uccle/03_tnx.json job_scripts/submit_05_uccle_laplace.pbs
qsub -N bx_tnn_lap -v CONFIG=examples/config/uccle/04_tnn.json job_scripts/submit_05_uccle_laplace.pbs
```

Final exact Laplace-MH fits:

```bash
qsub -N bx_txx_lmh -v CONFIG=examples/config/uccle/01_txx.json job_scripts/submit_09_uccle_laplace_mh.pbs
qsub -N bx_txn_lmh -v CONFIG=examples/config/uccle/02_txn.json job_scripts/submit_09_uccle_laplace_mh.pbs
qsub -N bx_tnx_lmh -v CONFIG=examples/config/uccle/03_tnx.json job_scripts/submit_09_uccle_laplace_mh.pbs
qsub -N bx_tnn_lmh -v CONFIG=examples/config/uccle/04_tnn.json job_scripts/submit_09_uccle_laplace_mh.pbs
```

PGAS uses the same four JSON files with
`job_scripts/submit_06_uccle_pgas.pbs`.

Each Uccle job requests four cores and starts its four one-chain Python
processes concurrently. If PBS runs all four series jobs together, TXx, TXn,
TNx, TNn, and all 16 chains run concurrently. This is supported by the code;
the scheduler still decides whether your account may occupy all 16 cores at
once.

## Run IDs and output safety

Leave `output.run_id` as JSON `null` for routine submissions. The runner then
builds a readable unique ID from:

```text
timestamp + configuration filename + PBS job ID
```

This prevents jobs starting in the same second from sharing output or log
paths. If you set `output.run_id` manually, it must be unique for that
submission.

Final combined results are written below:

```text
results/<example>/<automatic-id>_combined__<settings-signature>/
```

The sibling `_chain01` through `_chain04` directories retain the independent
one-chain fits used in the combination.

## Monitor jobs

```bash
qstat -u "$USER"
tail -f "$(ls -t logs/05_uccle_laplace_*.log | head -1)"
tail -f "$(ls -t logs/05_uccle_laplace_*_chain01.log | head -1)"
```

The main PBS log records launch and combine status. MCMC progress is in the
four per-chain logs when `runtime.progress` is `true` in JSON.

## Interactive use of the same runner

The Bash runners use exactly the same JSON and parallel-chain logic outside
PBS. The final argument is the maximum number of concurrent workers:

```bash
bash bash_scripts/run_05_uccle_laplace.sh examples/config/uccle/01_txx.json 4
```
