# BUCEX 1.9.8.1 — fixed-prior comparison and screening commands

## What is compared

Every full-record setting is fitted with three constructions:

| Construction | Prior on signed innovation coefficient | What is learned |
|---|---|---|
| Pooled HN | `s_c,j | tau_c ~ Normal(0,tau_c^2)`, `tau_c ~ HalfNormal(A_c)` | One scale per component, shared across all six responses |
| Separate HN | `s_c,j | tau_c,j ~ Normal(0,tau_c,j^2)`, `tau_c,j ~ HalfNormal(A_c)` | Separate scales for each response |
| Separate fixed Normal | `s_c,j ~ Normal(0,a_c^2)`, with `a_c=A_c` fixed | The innovation coefficients and process variances; **no hyperpriors** |

Fixing the Normal prior SD does not fix the process variance: `q_c,j=s_c,j^2` is inferred in every construction. Equal calibration values match `E[s_c,j^2]=A_c^2`, but fixed Normal and normal–half-normal marginal distributions differ. The two HN constructions have exactly matching one-response marginal priors.

The requested **3×4 grid** is:

| Level scale | Slope scales | Seasonal scale |
|---:|---|---:|
| 0.05 | 0.001, 0.002, 0.005, 0.01 | 0.1 |
| 0.1 | 0.001, 0.002, 0.005, 0.01 | 0.1 |
| 0.5 | 0.001, 0.002, 0.005, 0.01 | 0.1 |

The reference remains `(0.1, 0.002, 0.1)`. Two additional seasonal checks use 0.05 and 0.2 at level 0.1/slope 0.002: **14 settings per construction**. In fixed-prior fits these numbers are prior SDs `a`, rather than HN hyperprior scales `A`.

Other settings: dynamic level, slope and dummy seasonality; initial level SD 10 °C; initial zero-sum seasonal contrast SD 10 °C; independent initial slope SD 0.01 °C per season; GEV shape prior SD 0.3; observation variance IG(2,2); seasonal log-scale contrast SD 0.3; no residual copula. HN hyperparameters retain exact GIG updates.

**Screen budget:** two parallel chains, 1,000 warmup and 2,000 retained iterations per chain. Full-record forecasts extend 120 seasons (30 years), with 50,000 predictive draws. Validation uses 10,000 predictive draws. Trajectory intervals are 95%; validation reports central 90%, 95% and 99% coverage, widths, tail misses, CRPS and PITs. Short-chain diagnostics are screening evidence, not confirmation of convergence.

## Host split

| Work | Gallade | BIOBOT |
|---|---:|---:|
| Pooled full-record fits | 14 | 0 |
| Separate HN full-record fits | 84 | 0 |
| Separate fixed-Normal full-record fits | 84 | 0 |
| Pooled hindcasts | 28 | 42 |
| Separate HN reference hindcasts | 12 | 18 |
| Separate fixed-Normal hindcasts | 168 | 252 |
| **Individual fits** | **390** | **312** |

Gallade runs full-record fits and origins **1990-11 and 2000-11**. BIOBOT runs origins **2010-11, 2015-11 and 2020-11**. Pooled and fixed-prior validation covers every setting at all five origins. Separate HN validation covers the reference only. Held-out lengths are 120, 103, 63, 43 and 23 seasons, respectively; each forecast is truncated where observed data end.

On Gallade, six separate response fits are bundled into one array element, running concurrently with two chain workers each. The arrays have **42 pooled elements** (2 CPUs/12 GiB each) and **58 separate elements** (12 CPUs/48 GiB each). Thus 100 array elements execute the 390 fits, plus a startup probe and collection job. On BIOBOT, the queue schedules the 312 individual fits within its CPU and memory budgets. The two host batches have no duplicate tasks.

## BIOBOT

Place the release ZIP in your home directory. Use the Python environment that worked for your recent fits:

```bash
cd ~
unzip bucex-1.9.8.1.zip
cd -P ~/bucex-1.9.8.1
conda activate bastiaan

export BUCEX_PYTHON="$(command -v python)"
export BUCEX_RESULTS_ROOT="$PWD/results/serra_1981_biobot"
export BUCEX_CPUS=48
export BUCEX_MAX_JOBS=24
export BUCEX_MEMORY_GB=150
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$BUCEX_RESULTS_ROOT"

bash RUN_SWEETSPOT_BIOBOT.sh --dry-run
```

An empty result root should show **312 selected fits**. Then launch the screening queue in the background:

```bash
nohup bash RUN_SWEETSPOT_BIOBOT.sh \
  > "$BUCEX_RESULTS_ROOT/queue.log" 2>&1 < /dev/null &
echo "$!" > "$BUCEX_RESULTS_ROOT/queue.pid"
tail -f "$BUCEX_RESULTS_ROOT/queue.log"
```

Ctrl-C stops `tail`, not the background queue. Alternatively, run `bash RUN_SWEETSPOT_BIOBOT.sh` inside your usual tmux/screen session; use only one launcher for this result root.

Status and task logs:

```bash
bash RUN_SWEETSPOT_BIOBOT.sh --status
ls -lt "$BUCEX_RESULTS_ROOT/screen/logs"
```

The queue can run up to 24 fits / 48 chain workers, subject to available memory. Use lower limits if other work is running. Python must be at least 3.10, with NumPy, SciPy, pandas, threadpoolctl and matplotlib. If dependencies are missing, install them in the active environment with `python -m pip install -e '.[plot]'`.

## Gallade

Place the ZIP in your home directory and extract into a fresh source directory on the physical data path:

```bash
cd -P /kyukon/data/gent/vo/000/gvo00048/vsc42619/GitHub
unzip "$HOME/bucex-1.9.8.1.zip"
cd -P bucex-1.9.8.1

export BUCEX_RESULTS_ROOT="/kyukon/data/gent/vo/000/gvo00048/vsc42619/bucex1981_results"
export BUCEX_SCHEDULER=slurm
export VSC_CLUSTER=gallade
export VSC_PROJECT=gvo00048
unset VSC_PARTITION
export VSC_ARRAY_LIMIT=16
export VSC_SEPARATE_ARRAY_LIMIT=8
```

Reuse the environment that previously worked on Gallade:

```bash
export BUCEX_VENV="/kyukon/scratch/gent/426/vsc42619/bucex_env_gallade_py311_196_20260929_170136"
export BUCEX_PYTHON="$BUCEX_VENV/bin/python"
export BUCEX_ENV_SETUP="$BUCEX_VENV/environment.sh"

bash RUN_SWEETSPOT_HPC.sh --dry-run &&
bash RUN_SWEETSPOT_HPC.sh
```

The environment supplies dependencies; runtime scripts put the new source directory on `PYTHONPATH`. The compute-node probe checks the model and version before arrays start. Do not run this architecture-specific Python on the login node.

If that older environment is missing, use this block **instead of** the reuse block:

```bash
export BUCEX_VENV="$VSC_SCRATCH/bucex_env_gallade_py311_1981_$(date +%Y%m%d_%H%M%S)"
export BUCEX_PYTHON_MODULE=Python/3.11.3-GCCcore-12.3.0
bash SETUP_HPC_ENV.sh &&
export BUCEX_PYTHON="$BUCEX_VENV/bin/python" &&
export BUCEX_ENV_SETUP="$BUCEX_VENV/environment.sh" &&
bash RUN_SWEETSPOT_HPC.sh --dry-run &&
bash RUN_SWEETSPOT_HPC.sh
```

Expected dry-run: `bx1981_probe`, pooled array **`1-42%16`**, separate array **`1-58%8`**, and an automatic collection job. Arrays start only after the compute probe succeeds. Collection starts after both arrays finish, including when some tasks fail. The existing six-hour limit is per array element, not a deadline for the entire screen.

At these caps, active array work can reserve up to **128 CPUs** (16×2 + 8×12) across the cluster. Caps can be raised if your allocation permits, but queue availability determines when resources start. `gallade` is a cluster, not a partition. `RUN_SCREEN_EXPERIMENTS.sh` defaults to the same batch; do not submit both launchers into one root.

```bash
squeue -M gallade -u "$USER"
sacct -M gallade --starttime=today --user="$USER" \
  --format=JobID%24,JobName%28,State%20,ExitCode,Elapsed
ls -lt "$BUCEX_RESULTS_ROOT/scheduler_logs"
ls -lt "$BUCEX_RESULTS_ROOT/screen/logs"
```

Scheduler logs are in `scheduler_logs/`; fit logs are in `screen/logs/`. Task directories contain `task.json`, `resolved_config.json`, and reports. Fixed fits have names such as `posterior_fixed_reference_TXm`; six separate fits are never mistaken for one pooled fit.

## Results and review

Both launchers automatically collect results and create compact evidence ZIPs under `<results-root>/exports/`, named `bucex1981_screen_<batch>_<timestamp>.zip`. Posterior archives stay in their task directories; the compact exports contain the evidence needed for the grid report.

- Gallade HTML: `screen/collected/sweetspot_hpc/sweetspot_review.html`.
- BIOBOT HTML: `screen/collected/sweetspot_validation/sweetspot_review.html`.
- Main new tables: `matched_fixed_prior_comparisons.csv` and `matched_prior_crps.csv`.
- Other tables: `matched_pooling_comparisons.csv`, `seasonal_comparisons.csv`, `stability_regions.csv`, `all_pairwise_stability.csv`, `scale_learning.csv`, `allocation_correlations.csv`, and validation coverage/PIT/CRPS exports.
- Each construction gets its own 3×4 heatmaps and level/rate plots with 95% intervals. A reference overlay compares all three constructions.
- Fixed-prior fits export `fixed_prior_settings.csv`, coefficient/trajectory draws, allocation and local-sensitivity diagnostics. They have no hyperparameter posterior to plot.
- Prior simulations cover all 14 pooled HN and 14 fixed-Normal settings. Separate HN has the same one-response marginal prior as pooled HN, although its cross-response prior dependence differs.

A matched CRPS difference is separate-fit CRPS minus pooled CRPS at the **same calibration, response, origin, date and horizon**. The older `paired_crps.csv` instead compares every variant to the pooled reference calibration; both are labelled separately.

Bring the two compact ZIPs to the same machine, extract them into separate directories, then generate the combined report from unchanged 1.9.8.1 source:

```bash
python -m research.seasonal.sweetspot_report \
  --root /path/to/extracted_gallade \
  --other-root /path/to/extracted_biobot \
  --tier screen --batch sweetspot \
  --output /path/to/bucex1981_combined_review
```

Each input directory must contain its extracted `screen/` subdirectory. Replace the three example paths with actual local directories. Missing fits are reported; incomplete separate bundles and numerical flags cannot pass stability checks.

Manual partial collection on Gallade:

```bash
bash COLLECT_HPC_RESULTS.sh screen sweetspot_hpc --figures
```

This is a substantially larger screen than 1.9.8 because fixed-prior validation now spans the full grid. No three-hour completion time is guaranteed: pooled and single-response fit times differ, and scheduler waits and forecast/report costs remain. Use the first completed jobs to assess actual throughput.

To run the whole study on BIOBOT alone, choose a different output root and run `bash RUN_SWEETSPOT_BIOBOT.sh --batch sweetspot` (702 fits). The optional `--batch sweetspot_fixed` runs only fixed-prior fits (504), and overlaps the complete study. Do not launch overlapping batches into the same result root.

The paper tier remains available through `bash RUN_SWEETSPOT_HPC.sh paper` and `bash RUN_SWEETSPOT_BIOBOT.sh --tier paper`. The commands above submit only the screen. Completed but poorly mixed fits are marked `needs_review`; `--retry-failed` retries execution failures, not scientifically unconverged completed fits. Preserve unchanged source when resuming, and use a fresh root after any source or scientific setting change.
