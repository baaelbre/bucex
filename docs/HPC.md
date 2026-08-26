# Running bucex 1.4.0 with PBS

## One source of settings

The selected JSON is the complete scientific and computational
specification. The PBS file requests resources and starts a runner; it does
not replace model or MCMC values.

| Concern | Controlled by |
|---|---|
| data, scale model, structural model, priors | selected JSON |
| draws, warmup, thinning, chains, seed | selected JSON |
| particles, Laplace and log-scale proposal controls | selected JSON |
| figures, result root, overwrite policy | selected JSON |
| walltime, cores, memory, scheduler log | PBS file |
| concurrent chain processes and combination | generic chain runner |

For a four-chain JSON, the runner makes four temporary one-chain copies. It
keeps draws, warmup, thinning, priors, and all model settings unchanged. It
sets `chains` to one per process, offsets the base seed, gives every chain a
unique run ID, and combines the four compatible fits after they all succeed.
The original JSON is never modified and is recorded in result metadata.

## One-time cluster setup

From the repository root:

```bash
module load Python/3.12.3-GCCcore-13.3.0
python -m venv "$HOME/venvs/bucex_env"
source "$HOME/venvs/bucex_env/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e ".[plot]"
python -c "import bucex, matplotlib; print(bucex.__version__)"
```

The version should be `1.4.0`. The Bash runner defaults to
`$HOME/venvs/bucex_env`. If the environment is elsewhere:

```bash
export BUCEX_VENV_DIR=/absolute/path/to/bucex_env
```

## Submit one log-scale job

Run from the repository root and leave `output.run_id` as JSON `null`:

```bash
qsub -v CONFIG=examples/config/phi/simulation_rw.json \
  job_scripts/submit_10_simulation_phi.pbs

qsub -v CONFIG=examples/config/phi/uccle/01_txx_ssvs.json \
  job_scripts/submit_11_uccle_phi.pbs
```

The first command fits the RW scale model to the simulation. The second fits
TXx with scale-model SSVS. Each selected JSON requests 1,000 warmup iterations,
1,000 retained draws, and four independent chains.

## Submit all four simulation scale models

```bash
qsub -v CONFIG=examples/config/phi/simulation_stationary.json job_scripts/submit_10_simulation_phi.pbs
qsub -v CONFIG=examples/config/phi/simulation_linear.json job_scripts/submit_10_simulation_phi.pbs
qsub -v CONFIG=examples/config/phi/simulation_rw.json job_scripts/submit_10_simulation_phi.pbs
qsub -v CONFIG=examples/config/phi/simulation_ssvs.json job_scripts/submit_10_simulation_phi.pbs
```

Equivalent shell loop:

```bash
for config in examples/config/phi/simulation_*.json; do
  qsub -v CONFIG="$config" job_scripts/submit_10_simulation_phi.pbs
done
```

## Submit all 16 Uccle scale jobs

There is one JSON for every combination of TXx, TXn, TNx, TNn and stationary,
linear, RW, SSVS scale. Submit all of them with:

```bash
for config in examples/config/phi/uccle/*.json; do
  qsub -v CONFIG="$config" job_scripts/submit_11_uccle_phi.pbs
done
```

To submit one model for all four series, use one suffix:

```bash
for config in examples/config/phi/uccle/*_rw.json; do
  qsub -v CONFIG="$config" job_scripts/submit_11_uccle_phi.pbs
done
```

Replace `_rw.json` by `_stationary.json`, `_linear.json`, or `_ssvs.json`.
The explicit 20-command inventory is in `HPC_ALL_COMMANDS.md`.

## Allowed parallelism

Yes: series, scale models, and chains are independent at the job boundary.
The current code supports all of the following:

- four chains inside one job, using four one-core processes;
- TXx, TXn, TNx, and TNn as four jobs at the same time;
- stationary, linear, RW, and SSVS fits as separate jobs at the same time;
- all 16 Uccle series-by-model jobs at the same time.

One model across four series can therefore use up to 16 cores. All 16 Uccle
jobs can use up to 64 cores. This is allowed by bucex; PBS account limits and
queue policy determine how many jobs actually start. Do not make
`mcmc.chains` larger than the PBS `ppn` allocation unless you also increase
the allocation.

The Uccle example intentionally fits separate univariate models. Dynamic
`phi` is not currently available for `MultiSeriesModel`, so no hidden shared
state couples the four jobs.

## Local use of the identical runner

The final argument is the maximum number of chain processes:

```bash
bash bash_scripts/run_10_simulation_phi.sh examples/config/phi/simulation_rw.json 4
bash bash_scripts/run_11_uccle_phi.sh examples/config/phi/uccle/01_txx_ssvs.json 4
```

Direct Python also works, but `bx.fit` executes its chains within that Python
run rather than through the process fan-out wrapper:

```bash
python examples/11_uccle_phi.py --config examples/config/phi/uccle/01_txx_ssvs.json
```

## Run IDs and the former output collision

With `output.run_id: null`, the runner creates:

```text
YYYYMMDD_HHMMSS_<config-name>_<PBS-job-id>
```

Each chain adds `_chain01` through `_chain04`; the combined run adds
`_combined`. This prevents simultaneous settings files from writing the same
`run_config.json`. A manually specified `output.run_id` must be unique for the
submission.

Results are written below:

```text
results/10_simulation_phi/<automatic-id>_combined__<signature>/
results/11_uccle_phi/<automatic-id>_combined__<signature>/
```

The one-chain directories remain available for diagnostics and recovery.

## Monitor jobs

```bash
qstat -u "$USER"
tail -f "$(ls -t logs/11_uccle_phi_*.log | head -1)"
tail -f "$(ls -t logs/11_uccle_phi_*_chain01.log | head -1)"
```

The main log reports fan-out and combination. Per-chain MCMC progress appears
in chain logs when `runtime.progress` is `true` in JSON.

## Why there is no separate `hpc/` directory

HPC execution is a thin deployment layer, not a second analysis. PBS files
and the generic chain runner live in `job_scripts/`; local wrappers live in
`bash_scripts/`; examples and authoritative JSONs remain in `examples/`.
This keeps the same Python and settings file in use locally and on PBS and
avoids duplicated HPC-only configuration.

The older examples 00–09 retain their matching PBS files. Their complete
submission inventory remains in `HPC_ALL_COMMANDS.md`.
