# Legacy 1.9.6 suite, retained in 1.9.7

For the new joint calibration study, use **BUCEX-1.9.7-commands.md** instead.
This document describes the separately available legacy `all` suite.

The release uses the pooled half-normal reference with
`A_level=0.01`, **`A_slope=0.0002`**, and `A_season=0.01`.
Initial slopes remain separate `Normal(0, 0.01²)` priors.
Initial levels have SD 10 °C; initial seasonal **orthonormal contrast coordinates** have SD 10 °C.
The latter induces an exchangeable zero-sum cycle, with marginal SD 8.66 °C per season.
It is implemented as the mathematically equivalent correlated prior on the sampler's lag coordinates.
The observation-variance prior remains IG(2,2), seasonal log-scale contrasts have SD 0.30,
and the GEV shape prior has SD 0.30.

Every screen fit has **two parallel chains**, each with **1,000 warm-up + 1,000 retained** iterations.
Short chains are for screening; convergence flags remain visible. Paper settings use four chains
and the longer retained-draw budgets. None of the screen commands submits paper fits.

## What `all` runs

| Work | Fits |
|---|---:|
| Full-record pooled reference and sensitivities | 25 |
| Matched private half-normal fits, one per summary | 6 |
| Expanding-window hindcasts | 56 |
| Pre-2019 refits | 6 |
| Monthly/seasonal forecasts of matching seasonal targets | 6 |
| Full-record monthly supplement | 1 |
| **Total** | **100** |

The full-record grid contains all combinations of level scales `{0.005, 0.01, 0.02}`
and slope scales `{0.0001, 0.0002, 0.0004, 0.001}`. `reference` is the centre level scale
and slope 0.0002; `half_slope` is the former slope 0.0001; `double_slope` is now 0.0004.
Additional checks use seasonal scales 0.05 and 0.10, fixed location seasonality,
initial-slope SDs 0.005 and 0.02, shape SDs 0.15 and 0.60, log-scale contrast SDs
0.15 and 0.60, constant observation dispersion, half/double baseline observation scale,
and an initialization bridge with the old SD-20 independent lag prior and SD-20 level.
That bridge holds the new slope reference fixed.

All four central-level slope settings are validated at November 1970, 1975, …, 2020:
44 fits. Fixed location seasonality, seasonal scale 0.10, and the two shape-prior
sensitivities add 12 fits at 2010, 2015, and 2020. Forecasts extend up to **35 years**;
verification stops at the available record, JJA 2026. Later origins therefore have
shorter available horizons. Every comparison uses matching origin, response, and target dates.
Coverage uses 90%, 95%, and 99% intervals; reports also contain CRPS, forecast PIT,
threshold scores, interval widths, and directional tail misses. Horizon bands are
1–5, 6–10, 11–20, and 21–35 years. Tables count unique verification dates as well as
forecast cases: overlapping forecasts are not independent observations.

Full-record seasonal forecasts extend **30 years**, using **50,000 future paths**.
Predictive fan-chart quantiles invert the average conditional CDF, integrating out
observation noise. Validation uses 10,000 future paths for screen and 50,000 for paper.
The monthly supplement uses 10,000 / 20,000 future paths. Nested annual return-level
calculations use 5,000 / 20,000 whole paired paths to bound their cost; the exact count
is recorded in `risk_definitions.json`. Posterior bands are 95% throughout.

No lognormal, half-t, half-Cauchy, or leave-one-summary-out jobs are submitted.
Thirty further matched private fits across five level/slope settings remain available
under `deferred`; they are not part of `all` or the three-hour target.

## BIOBOT

Upload the ZIP to your home directory. Extract to its new directory, keeping existing
versions and results intact:

```bash
cd ~
unzip bucex-1.9.7.zip
cd ~/bucex-1.9.7
conda activate bastiaan
export BUCEX_PYTHON="$(command -v python)"
"$BUCEX_PYTHON" -m pip install -e '.[plot]' --no-deps
export BUCEX_RESULTS_ROOT="$PWD/results/serra_197_screen1k"
export BUCEX_CPUS=48
export BUCEX_MAX_JOBS=24
export BUCEX_MEMORY_GB=150
mkdir -p "$BUCEX_RESULTS_ROOT"

bash RUN_SCREEN_BIOBOT.sh all --dry-run
nohup bash RUN_SCREEN_BIOBOT.sh all \
  > "$BUCEX_RESULTS_ROOT/overnight.log" 2>&1 < /dev/null &
echo "$!" > "$BUCEX_RESULTS_ROOT/overnight.pid"
tail -f "$BUCEX_RESULTS_ROOT/overnight.log"
```

The queue runs at most 24 fits / 48 chain processes, with BLAS threads restricted to
one per process. Its memory checks may temporarily reduce concurrency. Monthly jobs
start early so that a slower monthly fit is not left until the final wave. Prior
simulations and startup checks run first; tables, figures, and a compact review ZIP
are produced automatically after fitting.

`Ctrl-C` exits `tail`; the detached queue keeps running. From another terminal:

```bash
cd ~/bucex-1.9.7
conda activate bastiaan
export BUCEX_PYTHON="$(command -v python)"
export BUCEX_RESULTS_ROOT="$PWD/results/serra_197_screen1k"
bash RUN_SCREEN_BIOBOT.sh all --status
tail -f "$BUCEX_RESULTS_ROOT/screen/logs/posterior_reference.log"
```

Completed matching fits are skipped on restart. To retry recorded failures after
checking their logs, run the same launcher with `--retry-failed`. Do not change source,
settings, or data inside an active release directory. New settings require a new root.

## VSC HPC — native Slurm on Gallade

Upload `bucex-1.9.7.zip` to your VSC home directory. These commands put the source on
compute-visible scratch and results on the VO data path used in this project:

```bash
cd "$VSC_SCRATCH"
unzip "$HOME/bucex-1.9.7.zip"
cd -P bucex-1.9.7
export VSC_CLUSTER=gallade
export VSC_PROJECT=gvo00048
unset VSC_PARTITION
export BUCEX_SCHEDULER=slurm
export VSC_ARRAY_LIMIT=48
export BUCEX_RESULTS_ROOT="/kyukon/data/gent/vo/000/gvo00048/$USER/bucex197_results"
mkdir -p "$BUCEX_RESULTS_ROOT"

export BUCEX_VENV="$PWD/bucex_env_gallade_py311_197"
bash SETUP_HPC_ENV.sh
export BUCEX_PYTHON="$BUCEX_VENV/bin/python"
export BUCEX_ENV_SETUP="$BUCEX_VENV/environment.sh"

bash RUN_SCREEN_ALL.sh --dry-run
bash RUN_SCREEN_ALL.sh
```

`SETUP_HPC_ENV.sh` builds the environment on a compute node and waits for that setup
job. Do not activate the architecture-specific environment on the login node. If you
already have a working Gallade environment, you can instead set `BUCEX_PYTHON` and
`BUCEX_ENV_SETUP` to its existing absolute paths and omit the setup step. The runtime
adds this release directory to `PYTHONPATH` and the probe records the loaded version.

Submission creates a startup/prior-check job, then three arrays:

- 90 shared-fit elements, at most 48 concurrent; 2 CPUs per element.
- 1 private-fit bundle, running six independent fits with two chains each; 12 CPUs.
- 4 monthly elements; 2 CPUs each.

That is **95 array elements running 100 fits**, plus startup and automatic collection
jobs. The cap is per resource class; total allocated CPUs may exceed 96 while the
monthly and private arrays also run. Queue and account limits can reduce concurrency.
All arrays wait for the startup check to pass. Collection waits for every submitted
array to finish, including failed tasks, and reports missing/failed fits explicitly.

There is **no three-hour launcher cutoff**. Slurm requests six hours per screen element
as a safety margin; that is an allocation limit, not an estimate of runtime.

Check progress:

```bash
squeue --clusters=gallade -u "$USER"
ls -lt "$BUCEX_RESULTS_ROOT/scheduler_logs" | head
ls -lt "$BUCEX_RESULTS_ROOT/screen/logs" | head
tail -f "$BUCEX_RESULTS_ROOT/screen/logs/posterior_reference.log"
```

Submission IDs and commands are saved in `$BUCEX_RESULTS_ROOT/screen/submissions/`.
For a scheduler failure, use the printed job ID:

```bash
sacct --clusters=gallade -j JOB_ID \
  --format=JobID%24,State%20,ExitCode,Elapsed
```

A failed startup probe prevents model jobs from starting. Its log names the failed
check and Python path. The compute node must be able to read the project, Python
and environment-setup paths and write the results directory.

To collect again manually after inspecting/retrying failures:

```bash
bash COLLECT_HPC_RESULTS.sh screen all --figures
```

## Results and timing

On either host:

- Per-fit progress: `<root>/screen/logs/`.
- Fits and CSVs: `<root>/screen/<task_id>/report/`; private and validation reports have channel subdirectories.
- Combined status: `<root>/screen/collected/all/status.json` and `tasks.csv`.
- Predictive comparisons: `<root>/screen/collected/all/calendar_35y/`.
- Main review figures: `<root>/screen/collected/all/diagnostic_figures_NOT_FINAL/`.
- Prior checks: `<root>/screen/prior_simulations/`.
- Automatic compact review archive: `<root>/exports/`.

`seasonal_cycle_changes` shows draw-wise changes from the first observation of the
same season, making small seasonal evolution visible separately from the starting
cycle's amplitude. The forecast uncertainty figure distinguishes latent level,
seasonal location and observation uncertainty. The code does not force bands to
widen or make their upper boundaries parallel.

At your earlier **27.2 minutes per two-chain fit**, 100 fits / 24 concurrent fits means
five waves, or **about 136 minutes before extra overhead**. Allow additional time for
monthly models, prior checks, forecasting and report collection. Around three hours
is a reasonable target on an otherwise available BIOBOT, not a promise. CPU contention
and scheduler waiting can make it longer. On VSC, more concurrent allocations can
reduce compute time, but submission time is not the start of computation.

```bash
"$BUCEX_PYTHON" -m research.seasonal.runtime_estimate --cpus 48 --minutes-per-fit 27.2
```

The 50,000-path, 120-season, six-response forecast and predictive-quantile check took
about 33 seconds locally using a short smoke posterior. This verifies the optimized
forecast path, not total MCMC runtime. More forecast simulations reduce Monte Carlo
noise; they do not improve MCMC convergence. Review diagnostics before interpreting
prior sensitivity or choosing the final paper fit.
