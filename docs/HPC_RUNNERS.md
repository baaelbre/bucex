# PBS runners and submission files

Every analysis now has two files, following the usual PBS pattern:

- `bash_scripts/run_*.sh` is the executable runner. It accepts
  positional arguments, uses the bucex virtual-environment interpreter,
  exports the matching `BUCEX_*` variables, and calls the Python example.
- `job_scripts/submit_*.pbs` contains resource requests, logging, and
  named PBS settings. It calls the corresponding runner.

For the ordinary runners in examples 03--09, `CHAINS=4` means four independent
one-chain processes run concurrently on four requested PBS cores. Example 08
also has a high-throughput alternative that maps all six scenarios and all
four chains onto a 24-element PBS array. A dependent one-core job then combines
the fits and produces the final tables and figures. Examples 00--02 remain
single-process jobs.

For example:

```text
submit_03_simulation_laplace.pbs
    -> run_03_simulation_laplace.sh
        -> examples/03_simulation_laplace.py
```

Submit from the package root. `PBS_O_WORKDIR` will then point to the correct
directory:

```bash
qsub job_scripts/submit_03_simulation_laplace.pbs
```

For the recommended final Laplace-MH simulation run, use the submission helper:

```bash
bash bash_scripts/qsub_08_simulation_laplace_mh.sh
```

It calls `qsub` twice: a scenario-chain fit array followed by an `afterok`
finalizer. Defaults are 24 concurrent one-core tasks, 1,000 warm-up iterations,
and 1,000 retained draws. All task and final artifacts use one shared run
directory without concurrent writes to the same file.

## Environment

The runners look for the virtual environment in
`${HOME}/venvs/bucex_env` by default. They use its `bin/python` directly and
stop if it is absent; there is no fallback to an unrelated system Python.
Create and verify the environment from the package root with:

```bash
cd /kyukon/data/gent/vo/000/gvo00048/vsc42619/GitHub/bucex
python -m venv "${HOME}/venvs/bucex_env"
source "${HOME}/venvs/bucex_env/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[plot]"
python -c 'import sys, matplotlib, bucex; print(sys.executable); print(matplotlib.__version__); print(bucex.__version__)'
```

To use another environment, either edit the `VENV_DIR` default in the runner
or pass its directory when submitting:

```bash
qsub -v BUCEX_VENV_DIR=/absolute/path/to/bucex_env \
  job_scripts/submit_03_simulation_laplace.pbs
```

You can instead provide an explicit interpreter:

```bash
qsub -v BUCEX_PYTHON=/absolute/path/to/bucex_env/bin/python \
  job_scripts/submit_03_simulation_laplace.pbs
```

## Run a shell script directly

The runners use ordinary positional arguments, just like any Bash script. The
order is documented at the top of every file. For example, the Laplace
simulation runner expects:

```text
N_TIME PERIOD SIMULATION_SEED DRAWS WARMUP CHAINS MCMC_SEED
RESULTS_ROOT RUN_ID OVERWRITE
```

Thus a final run can be started interactively as:

```bash
bash bash_scripts/run_03_simulation_laplace.sh \
  1000 4 13081997 2000 2000 4 13081997 results manual_test 0
```

Arguments may be omitted from the right. For example, this changes only
`N_TIME` and `PERIOD` and keeps all later defaults:

```bash
bash bash_scripts/run_02_structural_simulations.sh 1500 4
```

Use the literal value `latest` for an open-ended Uccle end date:

```bash
bash bash_scripts/run_05_uccle_laplace.sh \
  1892-01-01 latest 2000 2000 4 56000 data results uccle_final 0
```

## Pass arguments through qsub

For a queued job, pass named variables with `qsub -v`. The PBS file reads
these variables and places them in the correct positional order for the
runner:

```bash
qsub -v N_TIME=1000,PERIOD=4,SIMULATION_SEED=13081997,DRAWS=2000,WARMUP=2000,CHAINS=4,MCMC_SEED=13081997 \
  job_scripts/submit_03_simulation_laplace.pbs
```

For PGAS, add the particle count:

```bash
qsub -v N_TIME=1000,PERIOD=4,DRAWS=2000,WARMUP=2000,CHAINS=4,PARTICLES=512 \
  job_scripts/submit_04_simulation_pgas.pbs
```

For the Uccle analysis:

```bash
qsub -v START=1892-01-01,END=2023-12-31,DRAWS=2000,WARMUP=2000,CHAINS=4,DATA_DIR=data \
  job_scripts/submit_05_uccle_laplace.pbs
```

There must be commas between variables and no spaces around the commas.
`qsub` variables override the defaults in the PBS file; the PBS file then
passes the resulting values to the `.sh` runner.

Add `BUCEX_PROGRESS=1` to the `qsub -v` list to write periodic MCMC progress
messages to the per-chain log files. It is `0` by default.

## Arguments by pair

| pair | runner positional arguments / PBS variable names |
|---|---|
| `00_uccle_record` | `START END DATA_DIR RESULTS_ROOT RUN_ID OVERWRITE` |
| `01_tail_simulations` | `N_TIME PERIOD RESULTS_ROOT RUN_ID OVERWRITE` |
| `02_structural_simulations` | `N_TIME PERIOD SIMULATION_SEED RESULTS_ROOT RUN_ID OVERWRITE` |
| `03_simulation_laplace` | `N_TIME PERIOD SIMULATION_SEED DRAWS WARMUP CHAINS MCMC_SEED RESULTS_ROOT RUN_ID OVERWRITE` |
| `04_simulation_pgas` | `N_TIME PERIOD SIMULATION_SEED DRAWS WARMUP CHAINS PARTICLES MCMC_SEED RESULTS_ROOT RUN_ID OVERWRITE` |
| `05_uccle_laplace` | `START END DRAWS WARMUP CHAINS MCMC_SEED DATA_DIR RESULTS_ROOT RUN_ID OVERWRITE` |
| `06_uccle_pgas` | `START END DRAWS WARMUP CHAINS PARTICLES MCMC_SEED DATA_DIR RESULTS_ROOT RUN_ID OVERWRITE` |
| `07_centered_ig` | `N_TIME SIMULATION_SEED DRAWS WARMUP CHAINS PARTICLES MCMC_SEED RESULTS_ROOT RUN_ID OVERWRITE` |
| `08_simulation_laplace_mh` | `N_TIME PERIOD SIMULATION_SEED DRAWS WARMUP CHAINS MCMC_SEED RESULTS_ROOT RUN_ID OVERWRITE MH_STEPS` |
| `09_uccle_laplace_mh` | `START END DRAWS WARMUP CHAINS MCMC_SEED DATA_DIR RESULTS_ROOT RUN_ID OVERWRITE MH_STEPS` |

The example-08 array additionally accepts environment variables
`SCENARIO_KEYS` (colon-separated), `MAX_CONCURRENT`, and
`BUCEX_PBS_DEPENDENCY`. Its internal files are:

```text
bash_scripts/qsub_08_simulation_laplace_mh.sh
bash_scripts/run_08_simulation_laplace_mh_task.sh
bash_scripts/run_08_simulation_laplace_mh_finalize.sh
job_scripts/submit_08_simulation_laplace_mh_array.pbs
job_scripts/submit_08_simulation_laplace_mh_finalize.pbs
examples/_08_simulation_laplace_mh_task.py
```

The main scientific settings—GEV scale and shape, process-noise truths, SSVS
probabilities, and prior scales—remain visible at the top of the Python files.
Edit those there for scientific sensitivity analyses.

For example 07, the PBS file also accepts `ENGINE` (`laplace` by default,
`laplace_mh`, or `pgas`), `RANDOM_WALK_SD`, `LEVEL_IG_A`,
`LEVEL_IG_B`, `SIGMA_IG_A`, `SIGMA_IG_B`, `XI_PRIOR_MEAN`, and `XI_PRIOR_SD`.
The `LEVEL_IG_*` pair parameterizes the inverse-gamma prior on
`q_level = sd.level**2` in shape/scale form.

## Submit the complete workflow

All ten examples are self-contained, so they may be submitted together; no
PBS dependency is required. Use one timestamp prefix and descriptive suffixes:

```bash
STAMP="$(date +%Y%m%d_%H%M%S)"
qsub -v RUN_ID="${STAMP}_structures",N_TIME=1000,PERIOD=4 \
  job_scripts/submit_02_structural_simulations.pbs
qsub -v RUN_ID="${STAMP}_sim_lap",N_TIME=1000,PERIOD=4,DRAWS=1000,WARMUP=1000,CHAINS=4 \
  job_scripts/submit_03_simulation_laplace.pbs
qsub -v RUN_ID="${STAMP}_sim_pgas",N_TIME=1000,PERIOD=4,DRAWS=1000,WARMUP=1000,CHAINS=4,PARTICLES=128 \
  job_scripts/submit_04_simulation_pgas.pbs
qsub -v RUN_ID="${STAMP}_centered_ig",ENGINE=laplace,N_TIME=1000,DRAWS=1000,WARMUP=1000,CHAINS=4,LEVEL_IG_A=2.0,LEVEL_IG_B=0.0025 \
  job_scripts/submit_07_centered_ig.pbs
RUN_ID="${STAMP}_sim_laplace_mh" DRAWS=1000 WARMUP=1000 CHAINS=4 \
  bash bash_scripts/qsub_08_simulation_laplace_mh.sh
qsub -v RUN_ID="${STAMP}_uccle_laplace_mh",START=1892-01-01,DRAWS=1000,WARMUP=1000,CHAINS=4,MCMC_SEED=56000,MH_STEPS=1 \
  job_scripts/submit_09_uccle_laplace_mh.pbs
```

See [`HPC.md`](HPC.md) for copy-and-paste pilot and final commands for all
ten jobs, plus monitoring commands.

For an ordinary four-chain job, use the result directory containing
`RUN_ID_combined__...c4...`. The `RUN_ID_chain01__...c1...` through
`RUN_ID_chain04__...c1...` directories and their separate logs are retained
for chain-level debugging. `CHAINS` must not exceed the `ppn` request in the
PBS file.

For the example-08 array, use the single directory printed by the submission
helper. One-chain task fits remain under `tasks/chainXX/fits/<scenario>/`; the
combined fit and presentation artifacts are under `fits/`, `tables/`, and
`figures/` in that same directory.

Monitor jobs with `qstat -u "$USER"` and cancel one with `qdel JOB_ID`.
Adjust the `#PBS` walltime, memory, CPU, project, and queue directives in each
submission file to match the local cluster.
