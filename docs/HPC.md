# Running bucex 1.3.1 with PBS

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

If the environment is elsewhere, export its location before `qsub`:

```bash
export BUCEX_VENV_DIR=/absolute/path/to/bucex_env
```

## Prepare a run

Copy, edit, and keep the JSON with the results:

```bash
cp examples/config/simulation.json examples/config/simulation_final.json
cp examples/config/uccle.json examples/config/uccle_final.json
```

Important fields are `mcmc.draws`, `mcmc.warmup`, `mcmc.chains`, the process
slab scales under `priors.innovation_slab_sd`, particle count under
`inference.pgas_particles`, `runtime.progress`, and `output.run_id`.

For a four-chain job, request at least four PBS cores. Each chain is a separate
single-threaded Python process, so this is real process-level parallelism.

## Submit

```bash
qsub job_scripts/submit_00_uccle_record.pbs
qsub job_scripts/submit_01_tail_simulations.pbs
qsub job_scripts/submit_02_structural_simulations.pbs

qsub -v CONFIG=examples/config/simulation_final.json \
  job_scripts/submit_03_simulation_laplace.pbs
qsub -v CONFIG=examples/config/simulation_final.json \
  job_scripts/submit_04_simulation_pgas.pbs
qsub -v CONFIG=examples/config/uccle_final.json \
  job_scripts/submit_05_uccle_laplace.pbs
qsub -v CONFIG=examples/config/uccle_final.json \
  job_scripts/submit_06_uccle_pgas.pbs
qsub job_scripts/submit_07_centered_ig.pbs
qsub -v CONFIG=examples/config/simulation_final.json \
  job_scripts/submit_08_simulation_laplace_mh.pbs
qsub -v CONFIG=examples/config/uccle_final.json \
  job_scripts/submit_09_uccle_laplace_mh.pbs
```

The Bash files can also be run interactively:

```bash
bash bash_scripts/run_03_simulation_laplace.sh \
  examples/config/simulation_final.json 4
```

## Monitor

```bash
qstat -u "$USER"
tail -f "$(ls -t logs/03_simulation_laplace_*.log | head -1)"
tail -f "$(ls -t logs/03_simulation_laplace_*_chain01.log | head -1)"
```

With one chain, progress appears in the main log. With multiple chains, the
per-chain files contain the MCMC progress controlled by
`runtime.progress: true`. All PDF and PNG figures made locally are also made
on the cluster because the identical numbered Python script performs the
final combine and plotting step.
