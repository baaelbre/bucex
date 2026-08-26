# Complete PBS submission commands

Run commands from the bucex 1.4.0 repository root. Each command submits one
independent PBS job. The selected JSON owns draws, warmup, chain count, seed,
priors, model, inference tuning, figures, and output policy.

## Log-scale simulations

```bash
qsub -v CONFIG=examples/config/phi/simulation_stationary.json job_scripts/submit_10_simulation_phi.pbs
qsub -v CONFIG=examples/config/phi/simulation_linear.json job_scripts/submit_10_simulation_phi.pbs
qsub -v CONFIG=examples/config/phi/simulation_rw.json job_scripts/submit_10_simulation_phi.pbs
qsub -v CONFIG=examples/config/phi/simulation_ssvs.json job_scripts/submit_10_simulation_phi.pbs
```

## Uccle: stationary scale

```bash
qsub -v CONFIG=examples/config/phi/uccle/01_txx_stationary.json job_scripts/submit_11_uccle_phi.pbs
qsub -v CONFIG=examples/config/phi/uccle/02_txn_stationary.json job_scripts/submit_11_uccle_phi.pbs
qsub -v CONFIG=examples/config/phi/uccle/03_tnx_stationary.json job_scripts/submit_11_uccle_phi.pbs
qsub -v CONFIG=examples/config/phi/uccle/04_tnn_stationary.json job_scripts/submit_11_uccle_phi.pbs
```

## Uccle: linear log scale

```bash
qsub -v CONFIG=examples/config/phi/uccle/01_txx_linear.json job_scripts/submit_11_uccle_phi.pbs
qsub -v CONFIG=examples/config/phi/uccle/02_txn_linear.json job_scripts/submit_11_uccle_phi.pbs
qsub -v CONFIG=examples/config/phi/uccle/03_tnx_linear.json job_scripts/submit_11_uccle_phi.pbs
qsub -v CONFIG=examples/config/phi/uccle/04_tnn_linear.json job_scripts/submit_11_uccle_phi.pbs
```

## Uccle: random-walk log scale

```bash
qsub -v CONFIG=examples/config/phi/uccle/01_txx_rw.json job_scripts/submit_11_uccle_phi.pbs
qsub -v CONFIG=examples/config/phi/uccle/02_txn_rw.json job_scripts/submit_11_uccle_phi.pbs
qsub -v CONFIG=examples/config/phi/uccle/03_tnx_rw.json job_scripts/submit_11_uccle_phi.pbs
qsub -v CONFIG=examples/config/phi/uccle/04_tnn_rw.json job_scripts/submit_11_uccle_phi.pbs
```

## Uccle: scale-model SSVS

```bash
qsub -v CONFIG=examples/config/phi/uccle/01_txx_ssvs.json job_scripts/submit_11_uccle_phi.pbs
qsub -v CONFIG=examples/config/phi/uccle/02_txn_ssvs.json job_scripts/submit_11_uccle_phi.pbs
qsub -v CONFIG=examples/config/phi/uccle/03_tnx_ssvs.json job_scripts/submit_11_uccle_phi.pbs
qsub -v CONFIG=examples/config/phi/uccle/04_tnn_ssvs.json job_scripts/submit_11_uccle_phi.pbs
```

The 20 commands above may be submitted together. Every supplied JSON requests
four chains, so full simultaneous execution would use 80 one-core chain
processes: 16 for the four simulations and 64 for the Uccle fits. The code
supports this; PBS quotas decide actual concurrency.

## Compact equivalent

```bash
for config in examples/config/phi/simulation_*.json; do
  qsub -v CONFIG="$config" job_scripts/submit_10_simulation_phi.pbs
done

for config in examples/config/phi/uccle/*.json; do
  qsub -v CONFIG="$config" job_scripts/submit_11_uccle_phi.pbs
done
```

## Earlier simulation engines

The six original structural configurations can still be submitted under all
three state engines:

```bash
for config in examples/config/simulations/*.json; do
  qsub -v CONFIG="$config" job_scripts/submit_03_simulation_laplace.pbs
  qsub -v CONFIG="$config" job_scripts/submit_04_simulation_pgas.pbs
  qsub -v CONFIG="$config" job_scripts/submit_08_simulation_laplace_mh.pbs
done
```

## Earlier Uccle engines and slab settings

This loop includes each primary and `*_narrow.json` configuration under
Laplace, PGAS, and Laplace-MH:

```bash
for config in examples/config/uccle/*.json; do
  qsub -v CONFIG="$config" job_scripts/submit_05_uccle_laplace.pbs
  qsub -v CONFIG="$config" job_scripts/submit_06_uccle_pgas.pbs
  qsub -v CONFIG="$config" job_scripts/submit_09_uccle_laplace_mh.pbs
done
```

## Custom virtual environment

If the environment is not `$HOME/venvs/bucex_env`, pass it with the config:

```bash
qsub \
  -v CONFIG=examples/config/phi/uccle/01_txx_ssvs.json,BUCEX_VENV_DIR=/absolute/path/to/bucex_env \
  job_scripts/submit_11_uccle_phi.pbs
```

## Monitor and cancel

```bash
qstat -u "$USER"
```

Only to cancel an actual submitted job, use its ID from `qstat`:

```bash
qdel JOB_ID
```
