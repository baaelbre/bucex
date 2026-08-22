# Running the ten BUCEX examples with PBS

Run every command below from the clean `bucex` repository root. The five fitting
jobs reserve four cores and start one independent one-chain Python process per
core.  After all chains finish, the runner combines them and writes the final
tables and figures once.  BLAS threads are fixed at one, so four chains use four
cores without oversubscription.

## One-time setup

```bash
cd /path/to/bucex
source "$HOME/venvs/bucex_env/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[plot]"
python -c "import sys, matplotlib, bucex; print(sys.executable); print(matplotlib.__version__); print(bucex.__version__)"
mkdir -p logs results
```

The runners use `$HOME/venvs/bucex_env/bin/python` directly and stop if it is
missing; they never silently fall back to another Python. If the environment
is elsewhere, include its location in the submission, for example:

```bash
qsub -v BUCEX_VENV_DIR="$HOME/path/to/bucex_env",DRAWS=400,WARMUP=100,CHAINS=4 \
  job_scripts/submit_03_simulation_laplace.pbs
```

Set `BUCEX_PROGRESS=1` in `qsub -v` to write periodic MCMC progress lines to
the per-chain logs. The default is `0` for quieter batch logs.

## Submit all ten pilot jobs

```bash
STAMP="$(date +%Y%m%d_%H%M%S)"

qsub -v START=1892-01-01,END=latest,DATA_DIR=data,RESULTS_ROOT=results,RUN_ID="${STAMP}_record" \
  job_scripts/submit_00_uccle_record.pbs

qsub -v N_TIME=800,PERIOD=4,RESULTS_ROOT=results,RUN_ID="${STAMP}_tails" \
  job_scripts/submit_01_tail_simulations.pbs

qsub -v N_TIME=1000,PERIOD=4,SIMULATION_SEED=13081997,RESULTS_ROOT=results,RUN_ID="${STAMP}_structures" \
  job_scripts/submit_02_structural_simulations.pbs

qsub -v N_TIME=1000,PERIOD=4,SIMULATION_SEED=13081997,DRAWS=400,WARMUP=100,CHAINS=4,MCMC_SEED=13081997,RESULTS_ROOT=results,RUN_ID="${STAMP}_sim_lap" \
  job_scripts/submit_03_simulation_laplace.pbs

qsub -v N_TIME=1000,PERIOD=4,SIMULATION_SEED=13081997,DRAWS=400,WARMUP=100,CHAINS=4,PARTICLES=128,MCMC_SEED=13081997,RESULTS_ROOT=results,RUN_ID="${STAMP}_sim_pgas" \
  job_scripts/submit_04_simulation_pgas.pbs

qsub -v START=1892-01-01,END=latest,DRAWS=500,WARMUP=500,CHAINS=4,MCMC_SEED=56000,DATA_DIR=data,RESULTS_ROOT=results,RUN_ID="${STAMP}_uccle_lap" \
  job_scripts/submit_05_uccle_laplace.pbs

qsub -v START=1892-01-01,END=latest,DRAWS=500,WARMUP=500,CHAINS=4,PARTICLES=128,MCMC_SEED=56000,DATA_DIR=data,RESULTS_ROOT=results,RUN_ID="${STAMP}_uccle_pgas" \
  job_scripts/submit_06_uccle_pgas.pbs

qsub -v N_TIME=1000,SIMULATION_SEED=13081997,RANDOM_WALK_SD=0.05,LEVEL_IG_A=2.0,LEVEL_IG_B=0.0025,DRAWS=500,WARMUP=500,CHAINS=4,PARTICLES=128,MCMC_SEED=13081997,RESULTS_ROOT=results,RUN_ID="${STAMP}_centered_ig" \
  job_scripts/submit_07_centered_ig_random_walk_gev.pbs

qsub -v N_TIME=240,DRAWS=400,WARMUP=400,CHAINS=2,MH_STEPS=1,RESULTS_ROOT=results,RUN_ID="${STAMP}_sim_laplace_mh" \
  job_scripts/submit_08_simulation_laplace_mh.pbs

qsub -v START=1892-01-01,END=latest,SERIES=TXx,DRAWS=400,WARMUP=400,CHAINS=2,MH_STEPS=1,DATA_DIR=data,RESULTS_ROOT=results,RUN_ID="${STAMP}_uccle_laplace_mh" \
  job_scripts/submit_09_uccle_laplace_mh.pbs
```

These jobs are independent and may be submitted together.  The scheduler will
start each when its requested resources are available.

## Final fitting jobs

After the pilot results and diagnostics are satisfactory, submit the four
fitting jobs with 2,000 warm-up iterations, 2,000 retained draws, and four
chains.  The PGAS examples below use 512 particles.

```bash
STAMP="$(date +%Y%m%d_%H%M%S)"

qsub -v N_TIME=1000,PERIOD=4,SIMULATION_SEED=13081997,DRAWS=2000,WARMUP=2000,CHAINS=4,MCMC_SEED=13081997,RESULTS_ROOT=results,RUN_ID="${STAMP}_sim_lap_final" \
  job_scripts/submit_03_simulation_laplace.pbs

qsub -v N_TIME=1000,PERIOD=4,SIMULATION_SEED=13081997,DRAWS=2000,WARMUP=2000,CHAINS=4,PARTICLES=512,MCMC_SEED=13081997,RESULTS_ROOT=results,RUN_ID="${STAMP}_sim_pgas_final" \
  job_scripts/submit_04_simulation_pgas.pbs

qsub -v START=1892-01-01,END=latest,DRAWS=2000,WARMUP=2000,CHAINS=4,MCMC_SEED=56000,DATA_DIR=data,RESULTS_ROOT=results,RUN_ID="${STAMP}_uccle_lap_final" \
  job_scripts/submit_05_uccle_laplace.pbs

qsub -v START=1892-01-01,END=latest,DRAWS=2000,WARMUP=2000,CHAINS=4,PARTICLES=512,MCMC_SEED=56000,DATA_DIR=data,RESULTS_ROOT=results,RUN_ID="${STAMP}_uccle_pgas_final" \
  job_scripts/submit_06_uccle_pgas.pbs

qsub -v N_TIME=1000,SIMULATION_SEED=13081997,RANDOM_WALK_SD=0.05,LEVEL_IG_A=2.0,LEVEL_IG_B=0.0025,DRAWS=2000,WARMUP=2000,CHAINS=4,PARTICLES=512,MCMC_SEED=13081997,RESULTS_ROOT=results,RUN_ID="${STAMP}_centered_ig_final" \
  job_scripts/submit_07_centered_ig_random_walk_gev.pbs

qsub -v N_TIME=1000,DRAWS=2000,WARMUP=2000,CHAINS=4,MH_STEPS=1,RESULTS_ROOT=results,RUN_ID="${STAMP}_sim_laplace_mh_final" \
  job_scripts/submit_08_simulation_laplace_mh.pbs

qsub -v START=1892-01-01,END=latest,DRAWS=2000,WARMUP=2000,CHAINS=4,MH_STEPS=1,DATA_DIR=data,RESULTS_ROOT=results,RUN_ID="${STAMP}_uccle_laplace_mh_final" \
  job_scripts/submit_09_uccle_laplace_mh.pbs
```

## Monitoring and outputs

```bash
qstat -u "$USER"
qstat -f JOB_ID
qstat -n JOB_ID
ls -lt logs | head
tail -f "$(ls -t logs/05_uccle_laplace_*.log | head -1)"
qdel JOB_ID
```

For a four-chain fitting job, the runner keeps four one-chain directories such
as `RUN_ID_chain01__...c1...` and writes the result to use for inference under
`RUN_ID_combined__...c4...`.  It also keeps one log per chain.  If a chain
fails, the runner exits nonzero and does not create a misleading combined fit.

The requested resources are defined at the top of each `.pbs` file.  Current
defaults are `ppn=4, mem=48gb` for Laplace and `ppn=4, mem=64gb` for PGAS.  Do
not set `CHAINS` above `ppn`; if eight chains are ever needed, request eight
cores and scale memory as well.

## Running a bash runner directly

This is useful on an interactive allocation.  Argument order is documented at
the top of each runner.  For example:

```bash
bash bash_scripts/run_05_uccle_laplace.sh \
  1892-01-01 latest 400 100 4 56000 data results manual_laplace 0

bash bash_scripts/run_06_uccle_pgas.sh \
  1892-01-01 latest 400 100 4 128 56000 data results manual_pgas 0

bash bash_scripts/run_07_centered_ig_random_walk_gev.sh \
  1000 13081997 400 100 4 128 13081997 results manual_centered_ig 0

bash bash_scripts/run_08_simulation_laplace_mh.sh \
  240 400 400 2 11001 results manual_sim_laplace_mh 1

bash bash_scripts/run_09_uccle_laplace_mh.sh \
  1892-01-01 latest TXx 400 400 2 11010 data results manual_uccle_laplace_mh 1
```

For example 07, the process truth and inverse-gamma sensitivity settings may
be supplied by name without editing either HPC file:

```bash
qsub -v RANDOM_WALK_SD=0.03,LEVEL_IG_A=2.0,LEVEL_IG_B=0.0009,DRAWS=2000,WARMUP=2000,CHAINS=4,PARTICLES=512 \
  job_scripts/submit_07_centered_ig_random_walk_gev.pbs
```

Here `LEVEL_IG_B` is the scale of the prior on the innovation variance, not on
its standard deviation. The full settings are retained in `run_config.json`.
