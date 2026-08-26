# Complete Uccle PBS command catalog

Run these commands from the bucex 1.5.2 repository root after installing the
package environment. JSON owns every scientific and computational setting.

## Recommended primary analysis: all six series

```bash
qsub -v CONFIG=examples/config/uccle_gaussian/01_txm.json job_scripts/submit_12_uccle_gaussian.pbs
qsub -v CONFIG=examples/config/uccle_gaussian/02_tnm.json job_scripts/submit_12_uccle_gaussian.pbs
qsub -v CONFIG=examples/config/uccle/01_txx.json job_scripts/submit_09_uccle_laplace_mh.pbs
qsub -v CONFIG=examples/config/uccle/02_txn.json job_scripts/submit_09_uccle_laplace_mh.pbs
qsub -v CONFIG=examples/config/uccle/03_tnx.json job_scripts/submit_09_uccle_laplace_mh.pbs
qsub -v CONFIG=examples/config/uccle/04_tnn.json job_scripts/submit_09_uccle_laplace_mh.pbs
```

Submit these together if the queue permits. TXm/TNm use exact Gaussian FFBS;
the extremes use exact stationary-scale GEV Laplace-MH.

## Optional descriptive record figures

```bash
qsub -v CONFIG=examples/config/record.json job_scripts/submit_00_uccle_record.pbs
```

This is descriptive and has no MCMC chains.

## Optional GEV scale-model SSVS: all four extremes

```bash
qsub -v CONFIG=examples/config/phi/uccle/01_txx_ssvs.json job_scripts/submit_11_uccle_phi.pbs
qsub -v CONFIG=examples/config/phi/uccle/02_txn_ssvs.json job_scripts/submit_11_uccle_phi.pbs
qsub -v CONFIG=examples/config/phi/uccle/03_tnx_ssvs.json job_scripts/submit_11_uccle_phi.pbs
qsub -v CONFIG=examples/config/phi/uccle/04_tnn_ssvs.json job_scripts/submit_11_uccle_phi.pbs
```

## Optional forced linear log scale

```bash
qsub -v CONFIG=examples/config/phi/uccle/01_txx_linear.json job_scripts/submit_11_uccle_phi.pbs
qsub -v CONFIG=examples/config/phi/uccle/02_txn_linear.json job_scripts/submit_11_uccle_phi.pbs
qsub -v CONFIG=examples/config/phi/uccle/03_tnx_linear.json job_scripts/submit_11_uccle_phi.pbs
qsub -v CONFIG=examples/config/phi/uccle/04_tnn_linear.json job_scripts/submit_11_uccle_phi.pbs
```

## Optional forced random-walk log scale

```bash
qsub -v CONFIG=examples/config/phi/uccle/01_txx_rw.json job_scripts/submit_11_uccle_phi.pbs
qsub -v CONFIG=examples/config/phi/uccle/02_txn_rw.json job_scripts/submit_11_uccle_phi.pbs
qsub -v CONFIG=examples/config/phi/uccle/03_tnx_rw.json job_scripts/submit_11_uccle_phi.pbs
qsub -v CONFIG=examples/config/phi/uccle/04_tnn_rw.json job_scripts/submit_11_uccle_phi.pbs
```

## Optional explicit stationary-scale sensitivity replicas

These go through the phi-aware example and are useful only when a directly
matched scale-model comparison is required. They duplicate the primary scale
assumption and are not additional primary analyses.

```bash
qsub -v CONFIG=examples/config/phi/uccle/01_txx_stationary.json job_scripts/submit_11_uccle_phi.pbs
qsub -v CONFIG=examples/config/phi/uccle/02_txn_stationary.json job_scripts/submit_11_uccle_phi.pbs
qsub -v CONFIG=examples/config/phi/uccle/03_tnx_stationary.json job_scripts/submit_11_uccle_phi.pbs
qsub -v CONFIG=examples/config/phi/uccle/04_tnn_stationary.json job_scripts/submit_11_uccle_phi.pbs
```

## Optional approximate/particle engine comparisons

The same four primary GEV JSONs can be passed to example 05 (approximate
Laplace) or example 06 (PGAS). These are method comparisons; do not submit them
for the recommended primary result unless needed.

```bash
for config in \
  examples/config/uccle/01_txx.json \
  examples/config/uccle/02_txn.json \
  examples/config/uccle/03_tnx.json \
  examples/config/uccle/04_tnn.json; do
  qsub -v CONFIG="$config" job_scripts/submit_05_uccle_laplace.pbs
  qsub -v CONFIG="$config" job_scripts/submit_06_uccle_pgas.pbs
done
```

## Optional narrower location-slab sensitivity

```bash
for config in examples/config/uccle/*_narrow.json; do
  qsub -v CONFIG="$config" job_scripts/submit_09_uccle_laplace_mh.pbs
done
```

## Custom virtual environment

```bash
qsub -v CONFIG=examples/config/uccle_gaussian/01_txm.json,BUCEX_VENV_DIR=/absolute/path/to/bucex_env job_scripts/submit_12_uccle_gaussian.pbs
```

## Monitor and cancel

```bash
qstat -u "$USER"
```

Cancel only a specific ID obtained from `qstat`:

```bash
qdel JOB_ID
```
