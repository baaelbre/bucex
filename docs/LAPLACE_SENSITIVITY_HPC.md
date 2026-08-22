# Laplace slab/seed sensitivity runs on PBS

This bundle runs a reproducible grid of simulation seeds and prior-slab
profiles. Every PBS fit task runs **one chain** for all six structural
scenarios. A dependent PBS array then combines the four chains and generates
the usual tables and figures.

## File convention

```text
bash_scripts/
  run_laplace_sensitivity_all.sh
  run_laplace_sensitivity_chain.sh
  run_laplace_sensitivity_combine.sh
job_scripts/
  submit_laplace_sensitivity_fits.pbs
  submit_laplace_sensitivity_combine.pbs
config/
  laplace_sensitivity_grid.sh
```

All executable runners are `run_*.sh`. All files submitted directly with
`qsub` are `submit_*.pbs`. The shared sensitivity settings are kept separately
under `config/`.

## 1. Included files

The full workflow is already part of this release. The Python example exposes
the simulation and prior settings through environment variables and defaults
to `LOCAL_LEVEL_SD=0.02`.

## 2. Create the environment once on the login node

Use the Python module available on your VSC cluster. For example:

```bash
cd /path/to/bucex

# Only if your cluster uses environment modules:
module avail Python
module load <the-python-module-you-will-also-use-in-the-jobs>

python -m venv "$HOME/venvs/bucex_env"
source "$HOME/venvs/bucex_env/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install matplotlib pandas numpy scipy

python -c "import bucex, matplotlib; print(bucex.__version__, matplotlib.__version__)"
```

Do not run `pip install` with a different Python from the one in the activated
environment. Check with `which python` and `python -m pip --version`.

## 3. Edit the grid

Edit only `config/laplace_sensitivity_grid.sh`.

The supplied grid contains five seeds and five profiles:

| profile | level slab | trend slab | season slab | initial-season SD |
|---|---:|---:|---:|---:|
| `base` | 0.03 | 0.0008 | 0.05 | 0.50 |
| `l02` | 0.02 | 0.0008 | 0.05 | 0.50 |
| `t05` | 0.03 | 0.0005 | 0.05 | 0.50 |
| `t12` | 0.03 | 0.0012 | 0.05 | 0.35 |
| `s035` | 0.03 | 0.0008 | 0.035 | 0.35 |

This gives `5 seeds × 5 profiles × 4 chains = 100` fit tasks, followed by
`25` combination tasks. Exact settings are also stored in each run's
`run_config.json`.

## 4. Submit everything

If no Python module is needed:

```bash
cd /path/to/bucex
export BUCEX_VENV_DIR="$HOME/venvs/bucex_env"
bash bash_scripts/run_laplace_sensitivity_all.sh
```

If the virtual environment requires a Python module:

```bash
export BUCEX_VENV_DIR="$HOME/venvs/bucex_env"
export BUCEX_PYTHON_MODULE='<the-same-module-used-to-create-the-venv>'
bash bash_scripts/run_laplace_sensitivity_all.sh
```

The helper prints both PBS job IDs. The second array depends on successful
completion of the first.

Manual submission is also possible:

```bash
source config/laplace_sensitivity_grid.sh

FIT_ID=$(qsub \
  -v BUCEX_VENV_DIR="$HOME/venvs/bucex_env" \
  -t "1-${SENS_N_FIT_TASKS}%${SENS_MAX_CONCURRENT_FITS}" \
  job_scripts/submit_laplace_sensitivity_fits.pbs)

qsub \
  -v BUCEX_VENV_DIR="$HOME/venvs/bucex_env" \
  -W "depend=afterok:${FIT_ID}" \
  -t "1-${SENS_N_COMBINE_TASKS}%${SENS_MAX_CONCURRENT_COMBINES}" \
  job_scripts/submit_laplace_sensitivity_combine.pbs
```

If the local PBS variant rejects an array-parent `afterok` dependency, wait
until `qstat` shows that the fit array is complete and submit the combine array
without `-W`, or replace `afterok` by the site's documented `afterokarray`.

## 5. Monitor progress

```bash
qstat -u "$USER"
tail -f logs/laplace_sensitivity/fit_s13081997_base_c1_1.log
```

Count completed one-chain fits:

```bash
find results/laplace_sensitivity/03_simulation_laplace \
  -path '*/fits/*/combined.bucex' | wc -l
```

Combined results, tables, and figures are placed under directories such as:

```text
results/laplace_sensitivity/03_simulation_laplace/
  s13081997_base_combined__n1000p4_d1000w1000c4/
```

The one-chain directories are retained so mixing can be inspected separately.

## Run one configuration interactively

The runner takes positional arguments in this order:

```text
simulation_seed profile level_slab trend_slab season_slab initial_season_sd
chain draws warmup n_time period results_root
```

For example:

```bash
bash bash_scripts/run_laplace_sensitivity_chain.sh \
  13081997 l02 0.02 0.0008 0.05 0.50 \
  1 1000 1000 1000 4 results/laplace_sensitivity
```

That command is useful for validating the environment before submitting the
full array.
