# BUCEX 1.9.8 — broader half-normal calibration and matched private fits

Use the new source directory and new results roots. Do not copy these settings over a running 1.9.6/1.9.6.1/1.9.7 job. Saved fits and prior simulations from those releases are not reused. Existing Python dependencies can be reused; the launchers put the current source first on `PYTHONPATH`.

## Settings

For each structural component, the signed innovation coefficient has prior
`s_c,j | tau_c ~ Normal(0, tau_c^2)` and `tau_c ~ HalfNormal(A_c)` in the shared model. In private analyses each response has its own `tau_c,j ~ HalfNormal(A_c)`. Thus the one-response marginal priors match exactly. Private means no pooling, not no shrinkage. Initial rates stay independent with SD 0.01 °C per seasonal step.

| Setting | A_alpha | A_beta | A_gamma |
|---|---:|---:|---:|
| Grid 1 | 0.05 | 0.001 | 0.1 |
| Grid 2 | 0.05 | 0.002 | 0.1 |
| Grid 3 | 0.05 | 0.004 | 0.1 |
| Grid 4 | 0.1 | 0.001 | 0.1 |
| Reference | 0.1 | 0.002 | 0.1 |
| Grid 6 | 0.1 | 0.004 | 0.1 |
| Grid 7 | 0.2 | 0.001 | 0.1 |
| Grid 8 | 0.2 | 0.002 | 0.1 |
| Grid 9 | 0.2 | 0.004 | 0.1 |
| Seasonal lower | 0.1 | 0.002 | 0.05 |
| Seasonal upper | 0.1 | 0.002 | 0.2 |

This is a 3×3 level–slope grid plus two central seasonal checks: 11 calibrations, not a 27-cell factorial. The seasonal checks assess local sensitivity at the reference; they do not assess every interaction with the level–slope settings. All 11 settings have matched private full-record fits.

Other settings remain fixed: dynamic level, slope and seasonality; half-normal hyperpriors; exact GIG updates of their scales (introduced in 1.9.7); identity copula; separately calibrated initial rates; initial level SD 10 °C; initial seasonal orthonormal contrast SD 10 °C; Normal GEV-shape prior with SD 0.3; seasonal observation log-scale contrast SD 0.3; observation variance prior IG(2,2). The initial seasonal SD is a different parameter from A_gamma. This release changes A_gamma to 0.1, not the initial seasonal SD.

Screen: two parallel chains, 1,000 warmup + 2,000 retained iterations per chain, R-hat threshold 1.05 and minimum ESS 200. Each fit retains its posterior archive, pointwise 95% intervals, trajectory and allocation diagnostics, and forecast outputs. Validation includes 90%, 95% and 99% central coverage, misses above/below, interval widths, CRPS and PIT. Full-record forecasts extend 120 seasonal steps (30 years), with 50,000 predictive draws. Validation uses 10,000 predictive draws and the same 120-step requested horizon, truncated by the observed record.

The reference is a calibration candidate, not a selected optimum. Prior simulations run for all 11 shared calibrations (1,000 joint replications per setting on screen). Shared/private matching is checked analytically in the configuration gate; the shared simulation does not represent private cross-response dependence. No shrinkage model is selected using a desired warming rate or the smallest individual CRPS.

## How the work is divided

| Work | Gallade | BIOBOT |
|---|---:|---:|
| Shared full-record fits | 11 | 0 |
| Private full-record fits (11 × 6 responses) | 66 | 0 |
| Shared hindcasts (11 settings per origin) | 22 | 33 |
| Private reference hindcasts (6 responses per origin) | 12 | 18 |
| **Total individual fits** | **111** | **51** |
| Experiment groups | 46 | 36 |

Gallade receives the 1990-11 and 2000-11 origins. BIOBOT receives 2010-11, 2015-11 and 2020-11. The available held-out lengths are respectively 120, 103, 63, 43 and 23 seasonal blocks. The folds overlap in calendar time and are calibration evidence, not independent replications or an untouched final test set.

A pooled fit uses 2 CPUs. Each Gallade private bundle runs six response fits × two chains = 12 CPUs. There are 33 pooled array elements and 13 private array elements on Gallade. BIOBOT schedules individual private fits, each using two CPUs, within one CPU/memory budget.

## Gallade screen

Place `bucex-1.9.8.zip` in your home directory, then extract into a separate source directory:

```bash
cd -P /kyukon/data/gent/vo/000/gvo00048/vsc42619/GitHub
unzip "$HOME/bucex-1.9.8.zip"
cd -P bucex-1.9.8

export BUCEX_RESULTS_ROOT="/kyukon/data/gent/vo/000/gvo00048/vsc42619/bucex198_results"
export BUCEX_SCHEDULER=slurm
export VSC_CLUSTER=gallade
export VSC_PROJECT=gvo00048
unset VSC_PARTITION
export VSC_ARRAY_LIMIT=16
export VSC_SEPARATE_ARRAY_LIMIT=4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Reuse the Gallade environment from your recent runs.
export BUCEX_VENV="/kyukon/scratch/gent/426/vsc42619/bucex_env_gallade_py311_196_20260929_170136"
export BUCEX_PYTHON="$BUCEX_VENV/bin/python"
export BUCEX_ENV_SETUP="$BUCEX_VENV/environment.sh"

bash RUN_SWEETSPOT_HPC.sh --dry-run &&
bash RUN_SWEETSPOT_HPC.sh
```

If that environment has been removed, create a fresh one on Gallade before submitting. Do not activate the architecture-specific environment or run its Python on the login node:

```bash
export BUCEX_VENV="/kyukon/scratch/gent/426/vsc42619/bucex_env_gallade_py311_198_$(date +%Y%m%d_%H%M%S)"
export BUCEX_PYTHON_MODULE=Python/3.11.3-GCCcore-12.3.0
bash SETUP_HPC_ENV.sh &&
export BUCEX_PYTHON="$BUCEX_VENV/bin/python" &&
export BUCEX_ENV_SETUP="$BUCEX_VENV/environment.sh" &&
bash RUN_SWEETSPOT_HPC.sh --dry-run &&
bash RUN_SWEETSPOT_HPC.sh
```

Expected dry-run: a `bx198_probe`, a `1-33%16` pooled array with 2 CPUs / 12 GiB per element, a `1-13%4` private array with 12 CPUs / 48 GiB per element, and automatic collection. Both arrays depend on a successful compute-node probe. The probe runs short pooled and private Gaussian/GEV fits and prior simulations. It checks startup, not convergence. Arrays have a six-hour per-element limit; runtime on Gallade is not measured by this release.

The two array caps are separate: at the defaults, active fits can use up to 16×2 + 4×12 = 80 CPUs across Gallade. Raising `VSC_ARRAY_LIMIT` changes the pooled cap; raise `VSC_SEPARATE_ARRAY_LIMIT` explicitly if you also want more private bundles. Scheduler/account limits still apply. `gallade` is the cluster; do not use it as a partition.

`RUN_SCREEN_EXPERIMENTS.sh` also defaults to this same HPC batch. Do not submit both launchers for the same results root.

Check the scheduler and logs:

```bash
squeue -M gallade -u "$USER"
sacct -M gallade --starttime=today --user="$USER" \
  --format=JobID%24,JobName%28,State%20,ExitCode,Elapsed
ls -lt "$BUCEX_RESULTS_ROOT/scheduler_logs"
ls -lt "$BUCEX_RESULTS_ROOT/screen/logs"
```

Scheduler logs: `$BUCEX_RESULTS_ROOT/scheduler_logs/`. Individual task logs: `$BUCEX_RESULTS_ROOT/screen/logs/`. Probe detail: `$BUCEX_RESULTS_ROOT/screen/startup_probes/`. Full results: `$BUCEX_RESULTS_ROOT/screen/<task-id>/`; `task.json` records completion and numerical status. Private results add the response directory beneath `report/`.

Automatic collection runs after both arrays finish, even if some jobs fail. Manual partial collection:

```bash
bash COLLECT_HPC_RESULTS.sh screen sweetspot_hpc --figures
```

## BIOBOT screen

Extract the archive into a separate directory on BIOBOT. Activate the Python environment that worked for your recent runs, then:

```bash
cd -P ~/bucex-1.9.8
export BUCEX_PYTHON="$(command -v python)"
export BUCEX_RESULTS_ROOT="$PWD/results/serra_198_biobot"
export BUCEX_CPUS=48
export BUCEX_MAX_JOBS=24
export BUCEX_MEMORY_GB=150
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

bash RUN_SWEETSPOT_BIOBOT.sh --dry-run
```

The dry-run must show 51 pending fits on an empty root. Python must be at least 3.10. The source needs NumPy, SciPy, pandas, threadpoolctl and matplotlib; if your environment lacks dependencies, install them in that environment with `python -m pip install -e '.[plot,test]'`.

Run inside a persistent terminal session, for example:

```bash
screen -S bx198
bash RUN_SWEETSPOT_BIOBOT.sh
```

Detach with **Ctrl-A, then D**; return with `screen -r bx198`. The screen session inherits the exports above. The queue probes the environment, simulates priors, schedules up to 24 fits / 48 chain workers within the 150 GiB reservation, then collects results. These are planning reservations; the queue also checks available memory. Use a smaller budget if other work is running. The actual posterior jobs run in parallel; the startup probe is intentionally small.

Status from another terminal with the same results-root export:

```bash
bash RUN_SWEETSPOT_BIOBOT.sh --status
ls -lt "$BUCEX_RESULTS_ROOT/screen/logs"
```

Task results are in `$BUCEX_RESULTS_ROOT/screen/<task-id>/`. Queue progress is printed in the `screen` session; individual logs are in `screen/logs/` under the results root. The HTML is `screen/collected/sweetspot_validation/sweetspot_review.html`. Since BIOBOT runs held-out fits only, its report has validation results; the combined report adds the full-record trajectories from Gallade.

If you choose to run everything on BIOBOT instead of splitting across hosts, use a separate root and `bash RUN_SWEETSPOT_BIOBOT.sh --batch sweetspot`. That schedules all 162 fits. Do not run this alongside the split queues into the same root.

## Collect and combine

Each host creates compact evidence ZIPs under `<results-root>/exports/`, named `bucex198_screen_<batch>_<timestamp>.zip`. They contain reports and compact diagnostics; the large `fit.bucex` posterior archives stay in their task directories. Upload the evidence ZIPs from both hosts for review. Preserve the source release that produced them: provenance checks cover source, settings and daily data.

For a local combined HTML, bring both result trees to the same machine (compact exports may be extracted into separate roots) and run from the unchanged 1.9.8 source:

```bash
python -m research.seasonal.sweetspot_report \
  --root /path/to/gallade_results \
  --other-root /path/to/biobot_results \
  --tier screen --batch sweetspot \
  --output /path/to/bucex198_combined_review
```

Replace those three paths with the actual local directories. The report checks provenance and rejects duplicated tasks across roots. It preserves all six private responses and treats partial private bundles as incomplete.

Outputs include:

- Title-free level and slope plots with pointwise 95% intervals, separate 3×3 grid heatmaps for pooled/private fits, central seasonal comparisons and a matched pooling comparison.
- `matched_pooling_comparisons.csv`, `seasonal_comparisons.csv`, `scale_learning.csv`, `allocation_correlations.csv`, `stability_regions.csv` and all pairwise stability differences.
- Per-origin CRPS, paired CRPS on identical held-out cases, 90/95/99% coverage, widths and PIT exports.
- Paired draws, local derivatives with respect to log hyperprior scales, level/slope variance allocation, 10-/30-year forecasts and parameter/trajectory numerical checks in each fit's report.

The comparison gates use the same declared physical tolerances as 1.9.6.1: level median/interval limits 0.1/0.2 °C, slope median/interval limits 0.05/0.1 °C per decade, forecast-level 0.2/0.4 °C, forecast-observation 0.2/0.5 °C, risk probability 0.02 and slope-variance fraction 0.1. Every pair in a neighbouring four-cell rectangle must pass for all six responses. Seasonal checks are separate from these rectangles. Passing is a descriptive screening criterion, not proof that the prior no longer matters.

Completed but unconverged tasks are marked `needs_review`; they are not silently rerun or presented as confirmed results. `--retry-failed` retries recorded failures only. A scientific setting or source change requires a fresh root. Use the unchanged code to resume. The paper tier remains available (`bash RUN_SWEETSPOT_HPC.sh paper`, and `bash RUN_SWEETSPOT_BIOBOT.sh --tier paper`), but longer confirmation should follow review of this screen, rather than automatically selecting the most favourable slope curve.
