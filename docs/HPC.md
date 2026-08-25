# Running bucex 1.2.1 with PBS

Run every command below from the clean `bucex` repository root. Most fitting
jobs reserve four cores and start one independent one-chain process per core.
Example 08 has an additional high-throughput workflow: every scenario-chain
pair is a separate one-core PBS array task, followed by a dependent finalizer.
BLAS threads are fixed at one throughout.

## One-time setup

```bash
cd /kyukon/data/gent/vo/000/gvo00048/vsc42619/GitHub/bucex
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
qsub -v BUCEX_VENV_DIR="$HOME/path/to/bucex_env",DRAWS=1000,WARMUP=1000,CHAINS=4 \
  job_scripts/submit_03_simulation_laplace.pbs
```

Set `BUCEX_PROGRESS=1` in `qsub -v` to write periodic MCMC progress lines to
the per-chain logs. The default is `0` for quieter batch logs.

## Final Laplace-MH simulations in one command

This is the recommended example-08 workflow. The defaults submit six scenarios
times four chains as 24 independent tasks, with all 24 eligible to run at once:

```bash
cd /path/to/bucex-1.2.1
export BUCEX_VENV_DIR="$HOME/venvs/bucex_env"
bash bash_scripts/qsub_08_simulation_laplace_mh.sh
```

The final profile is:

```text
N_TIME=1000  PERIOD=4  DRAWS=1000  WARMUP=1000
CHAINS=4     MH_STEPS=1  MAX_CONCURRENT=24
```

Each fit task requests one core, 12 GB, and at most six hours. The dependent
finalizer requests one core, 16 GB, and one hour. Based on the measured local
runtime, the intended compute time is about four to five hours for the slowest
fit task plus finalization; scheduler waiting time is not included or
guaranteed by bucex.

Override settings directly on the submission command:

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)_laplace_mh_final" \
DRAWS=1000 WARMUP=1000 CHAINS=4 MAX_CONCURRENT=24 \
bash bash_scripts/qsub_08_simulation_laplace_mh.sh
```

The helper prints the fit-array ID, finalizer ID, and exact shared result
directory. The completely explicit equivalent is:

```bash
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${STAMP}_sim_laplace_mh_final"
VARS="N_TIME=1000,PERIOD=4,SIMULATION_SEED=13081997,DRAWS=1000,WARMUP=1000,CHAINS=4,MCMC_SEED=13081997,RESULTS_ROOT=results,RUN_ID=${RUN_ID},OVERWRITE=0,MH_STEPS=1,SCENARIO_KEYS=stationary:linear:random_walk:llt:dynamic_season:llt_season,BUCEX_PROGRESS=1"

FIT_ID=$(qsub -t 1-24%24 -v "${VARS}" \
  job_scripts/submit_08_simulation_laplace_mh_array.pbs)
FINAL_ID=$(qsub -W "depend=afterok:${FIT_ID}" -v "${VARS}" \
  job_scripts/submit_08_simulation_laplace_mh_finalize.pbs)

echo "fit array: ${FIT_ID}"
echo "finalizer: ${FINAL_ID}"
```

Some PBS installations call the array dependency `afterokarray`; on such a
cluster set `BUCEX_PBS_DEPENDENCY=afterokarray` before invoking the helper.
Scenario keys are colon-separated when selecting a subset, for example
`SCENARIO_KEYS=stationary:random_walk:llt`.

## Submit all ten pilot jobs

```bash
STAMP="$(date +%Y%m%d_%H%M%S)"

qsub -v START=1892-01-01,END=latest,DATA_DIR=data,RESULTS_ROOT=results,RUN_ID="${STAMP}_record" \
  job_scripts/submit_00_uccle_record.pbs

qsub -v N_TIME=800,PERIOD=4,RESULTS_ROOT=results,RUN_ID="${STAMP}_tails" \
  job_scripts/submit_01_tail_simulations.pbs

qsub -v N_TIME=1000,PERIOD=4,SIMULATION_SEED=13081997,RESULTS_ROOT=results,RUN_ID="${STAMP}_structures" \
  job_scripts/submit_02_structural_simulations.pbs

qsub -v N_TIME=1000,PERIOD=4,SIMULATION_SEED=13081997,DRAWS=1000,WARMUP=1000,CHAINS=4,MCMC_SEED=13081997,RESULTS_ROOT=results,RUN_ID="${STAMP}_sim_lap" \
  job_scripts/submit_03_simulation_laplace.pbs

qsub -v N_TIME=1000,PERIOD=4,SIMULATION_SEED=13081997,DRAWS=1000,WARMUP=1000,CHAINS=4,PARTICLES=128,MCMC_SEED=13081997,RESULTS_ROOT=results,RUN_ID="${STAMP}_sim_pgas" \
  job_scripts/submit_04_simulation_pgas.pbs

qsub -v START=1892-01-01,END=latest,DRAWS=1000,WARMUP=1000,CHAINS=4,MCMC_SEED=56000,DATA_DIR=data,RESULTS_ROOT=results,RUN_ID="${STAMP}_uccle_lap" \
  job_scripts/submit_05_uccle_laplace.pbs

qsub -v START=1892-01-01,END=latest,DRAWS=1000,WARMUP=1000,CHAINS=4,PARTICLES=128,MCMC_SEED=56000,DATA_DIR=data,RESULTS_ROOT=results,RUN_ID="${STAMP}_uccle_pgas" \
  job_scripts/submit_06_uccle_pgas.pbs

qsub -v N_TIME=1000,SIMULATION_SEED=13081997,RANDOM_WALK_SD=0.05,LEVEL_IG_A=2.0,LEVEL_IG_B=0.0025,DRAWS=1000,WARMUP=1000,CHAINS=4,PARTICLES=128,MCMC_SEED=13081997,RESULTS_ROOT=results,RUN_ID="${STAMP}_centered_ig" \
  job_scripts/submit_07_centered_ig.pbs

RUN_ID="${STAMP}_sim_laplace_mh" DRAWS=1000 WARMUP=1000 CHAINS=4 \
  bash bash_scripts/qsub_08_simulation_laplace_mh.sh

qsub -v START=1892-01-01,END=latest,DRAWS=1000,WARMUP=1000,CHAINS=4,MCMC_SEED=56000,MH_STEPS=1,DATA_DIR=data,RESULTS_ROOT=results,RUN_ID="${STAMP}_uccle_laplace_mh" \
  job_scripts/submit_09_uccle_laplace_mh.pbs
```

These jobs are independent and may be submitted together.  The scheduler will
start each when its requested resources are available.

## Final fitting jobs

After the pilot results and diagnostics are satisfactory, submit the ordinary
fits with 2,000 warm-up iterations, 2,000 retained draws, and four chains. The
PGAS examples below use 512 particles. The exact Laplace-MH simulation array
instead uses 1,000 warm-up iterations and 1,000 retained draws per chain so
each independent task fits the six-hour target.

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
  job_scripts/submit_07_centered_ig.pbs

RUN_ID="${STAMP}_sim_laplace_mh_final" \
  DRAWS=1000 WARMUP=1000 CHAINS=4 MAX_CONCURRENT=24 \
  bash bash_scripts/qsub_08_simulation_laplace_mh.sh

qsub -v START=1892-01-01,END=latest,DRAWS=2000,WARMUP=2000,CHAINS=4,MCMC_SEED=56000,MH_STEPS=1,DATA_DIR=data,RESULTS_ROOT=results,RUN_ID="${STAMP}_uccle_laplace_mh_final" \
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

The example-08 array uses one shared directory instead:

```text
results/08_simulation_laplace_mh/<RUN_ID>__n1000p4_d1000w1000c4m1/
  tasks/chain01/fits/stationary/combined.bucex
  tasks/chain01/manifests/stationary.json
  ...
  tasks/chain04/fits/llt_season/combined.bucex
  fits/<scenario>/combined.bucex
  simulations/
  tables/
  figures/
  logs/<scenario>/
```

Array tasks never share a writable fit or manifest path. The final `fits/`,
`tables/`, `figures/`, and `simulations/` trees are produced only after every
task succeeds. If one task fails, the `afterok` dependency prevents a partial
result from being presented as final.

The requested resources are defined at the top of each `.pbs` file. Current
defaults are `ppn=4, mem=48gb` for Laplace and `ppn=4, mem=64gb` for PGAS.  Do
not set `CHAINS` above `ppn`; if eight chains are ever needed, request eight
cores and scale memory as well.

That `CHAINS <= ppn` constraint applies to the older single-job runners. In the
example-08 array, every task requests `ppn=1`; `CHAINS` determines the number
of array elements instead.

## Running a bash runner directly

This is useful on an interactive allocation.  Argument order is documented at
the top of each runner.  For example:

```bash
bash bash_scripts/run_05_uccle_laplace.sh \
  1892-01-01 latest 1000 1000 4 56000 data results manual_laplace 0

bash bash_scripts/run_06_uccle_pgas.sh \
  1892-01-01 latest 1000 1000 4 128 56000 data results manual_pgas 0

bash bash_scripts/run_07_centered_ig.sh \
  1000 13081997 1000 1000 4 128 13081997 results manual_centered_ig 0

bash bash_scripts/run_08_simulation_laplace_mh.sh \
  1000 4 13081997 1000 1000 4 13081997 results manual_sim_laplace_mh 0 1

bash bash_scripts/run_09_uccle_laplace_mh.sh \
  1892-01-01 latest 1000 1000 4 56000 data results manual_uccle_laplace_mh 0 1
```

For example 07, the process truth and inverse-gamma sensitivity settings may
be supplied by name without editing either HPC file:

```bash
qsub -v ENGINE=laplace,RANDOM_WALK_SD=0.03,LEVEL_IG_A=2.0,LEVEL_IG_B=0.0009,DRAWS=2000,WARMUP=2000,CHAINS=4 \
  job_scripts/submit_07_centered_ig.pbs
```

Here `LEVEL_IG_B` is the scale of the prior on the innovation variance, not on
its standard deviation. The full settings are retained in `run_config.json`.
