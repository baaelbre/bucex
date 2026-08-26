# Running bucex 1.5.2 with PBS

## One source of settings

The selected JSON owns the analysis. PBS requests resources and launches it.

| Concern | Controlled by |
|---|---|
| data, family, structure, priors | JSON |
| draws, warmup, thinning, chains, seed | JSON |
| inference and proposal controls | JSON |
| figures, predictions, result root, overwrite | JSON |
| walltime, cores, memory, scheduler log | PBS wrapper |
| process fan-out and fit combination | generic runner |

For a four-chain JSON, the runner creates four temporary effective configs
with one chain and distinct seeds/run IDs, executes them concurrently, and
combines the compatible fits. Draws and warmup remain 1,000 **in every chain**.
The selected source JSON is neither edited nor overwritten.

## One-time setup

From the `bucex-1.5.2` repository root:

```bash
module load Python/3.12.3-GCCcore-13.3.0
python -m venv "$HOME/venvs/bucex_env"
source "$HOME/venvs/bucex_env/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e ".[plot]"
python -c "import bucex, matplotlib; print(bucex.__version__)"
mkdir -p logs
```

The version must print `1.5.2`. If the environment is elsewhere, export once
before submission:

```bash
export BUCEX_VENV_DIR=/absolute/path/to/bucex_env
```

## Submit the recommended six Uccle jobs

Monthly means, exact Gaussian FFBS:

```bash
qsub -v CONFIG=examples/config/uccle_gaussian/01_txm.json job_scripts/submit_12_uccle_gaussian.pbs
qsub -v CONFIG=examples/config/uccle_gaussian/02_tnm.json job_scripts/submit_12_uccle_gaussian.pbs
```

Monthly extremes, exact stationary-scale GEV Laplace-MH:

```bash
qsub -v CONFIG=examples/config/uccle/01_txx.json job_scripts/submit_09_uccle_laplace_mh.pbs
qsub -v CONFIG=examples/config/uccle/02_txn.json job_scripts/submit_09_uccle_laplace_mh.pbs
qsub -v CONFIG=examples/config/uccle/03_tnx.json job_scripts/submit_09_uccle_laplace_mh.pbs
qsub -v CONFIG=examples/config/uccle/04_tnn.json job_scripts/submit_09_uccle_laplace_mh.pbs
```

All six may be submitted together. Each asks for one four-core job and runs
four independent chains concurrently, so the maximum is 24 one-core chain
processes if the scheduler starts everything at once. The code permits this;
the account allocation and queue policy decide actual concurrency.

## Optional scale-model SSVS sensitivity

For a single sensitivity fit:

```bash
qsub -v CONFIG=examples/config/phi/uccle/01_txx_ssvs.json job_scripts/submit_11_uccle_phi.pbs
```

For all four extreme series:

```bash
for config in examples/config/phi/uccle/*_ssvs.json; do
  qsub -v CONFIG="$config" job_scripts/submit_11_uccle_phi.pbs
done
```

Use `*_linear.json` or `*_rw.json` for forced-model sensitivity. The scale
files use the same primary location slabs as the stationary GEV analysis.
`HPC_ALL_COMMANDS.md` lists every explicit Uccle submission, including
descriptive and alternative-inference examples.

## Local use of the identical chain runner

```bash
bash bash_scripts/run_12_uccle_gaussian.sh examples/config/uccle_gaussian/01_txm.json 4
bash bash_scripts/run_09_uccle_laplace_mh.sh examples/config/uccle/01_txx.json 4
bash bash_scripts/run_11_uccle_phi.sh examples/config/phi/uccle/01_txx_ssvs.json 4
```

The final number is the maximum simultaneous chain processes. Direct Python
also works but performs all chains inside that process:

```bash
python examples/12_uccle_gaussian.py --config examples/config/uccle_gaussian/01_txm.json
```

## Run IDs and concurrent jobs

Leave `output.run_id` as JSON `null`. The runner generates:

```text
YYYYMMDD_HHMMSS_<config-name>_<PBS-job-id>
```

Chain runs add `_chain01` through `_chain04`; the report run adds `_combined`.
This prevents simultaneous configs from sharing a `run_config.json`. A manual
run ID is safe only if it is unique to that submission.

## Monitor

```bash
qstat -u "$USER"
tail -f "$(ls -t logs/12_uccle_gaussian_*.log | head -1)"
tail -f "$(ls -t logs/09_uccle_laplace_mh_*.log | head -1)"
```

Per-chain logs contain MCMC progress when `runtime.progress` is true. The main
job log records the fan-out and combination.

## Why there is no `hpc/` directory

HPC execution is a thin deployment layer, not a second analysis. PBS wrappers
and the generic chain runner live in `job_scripts/`; interactive wrappers live
in `bash_scripts/`; Python examples and authoritative JSONs remain in
`examples/`. Local and PBS execution therefore use exactly the same analysis
code and settings.
