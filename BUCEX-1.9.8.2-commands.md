# BUCEX 1.9.8.3 — monthly reference and sensitivity screen

This release adds **six separate monthly analyses with fixed Normal innovation priors, no hyperpriors, and repeating calendar-month observation scales**. Process variances remain learned through the signed coefficients. Existing seasonal workflows remain available.

## Calibration

The reference uses the central seasonal calibration from 1.9.8.1, converted to match the **30-year contribution SDs**. Change `seasonal_calibration` in `research/monthly/config/screen_study.json` before starting a fresh run if another seasonal anchor is wanted; conversion is automatic.

| Fixed Normal prior SD | Seasonal | Monthly |
|---|---:|---:|
| Level innovation | 0.1 | 0.057735026919 |
| Slope innovation | 0.002 | 0.000383292331 |
| Seasonal innovation | 0.1 | 0.1 |
| Initial slope | 0.01 | 0.003333333333 |

For H transitions the level gain is sqrt(H), integrated-slope gain is sqrt(H(H−1)(2H−1)/6), same-phase dummy-seasonal gain is sqrt(2H/p), and initial-slope gain is H. Matching H=120,p=4 with H=360,p=12 gives the values above. The initial-rate SD remains **0.4 °C/decade**. Matching is exact for these 30-year prior SDs; it does not equate the full monthly and seasonal processes, and integrated-slope effects differ slightly at other horizons. Fixed Normal and pooled half-normal priors have different marginal shapes even when their coefficient second moments match.

Initial level SD 10 °C; orthonormal initial seasonal contrast SD 10 °C; baseline observation variance IG(2,2); monthly log-scale contrast SD 0.30; GEV shape Normal(0,0.30²). The contrast prior applies to eleven coordinates, yielding marginal calendar-month contrast SD sqrt(11/12) times the coordinate SD. Observations run **March 1892–August 2026 (1,614 monthly blocks)**, matching the complete daily blocks retained for the seasonal analysis.

## Work scheduled

| Batch | Fits | Contents |
|---|---:|---|
| `monthly_reference` | 6 | Reference only |
| `monthly_structural` | 90 | Reference + structural/initial-state sensitivities |
| `monthly_observation` | 38 | Shape and observation-scale sensitivities |
| `monthly_sensitivity` | 122 | Every sensitivity, without reference |
| `monthly_all` | 128 | Complete monthly screen |

The **22 settings** are:

- Reference plus the other eight combinations of level and slope multipliers {1/2,1,2}.
- Seasonal innovation SD halved/doubled (seasonal-equivalent 0.05/0.20).
- Initial slope SD halved/doubled.
- Fixed location seasonality.
- The earlier initialization bridge: level SD 20, initial seasonal SD 20, lag basis. This intentionally changes both starting-state width and basis.
- GEV shape SD 0.15/0.60, only for the four GEV responses; unchanged Gaussian fits are not duplicated.
- Constant observation dispersion.
- Monthly log-scale contrast SD 0.15/0.60.
- Baseline observation variance IG(2,0.5)/IG(2,8), corresponding to half/double observation-scale calibration.

No half-t, half-Cauchy, lognormal hyperpriors, pooling, cross-summary contrasts or leave-one-out jobs are added. This round fits the complete record; **held-out refits and monthly-versus-seasonal comparisons are deferred**.

**Screen:** two parallel chains, **1,000 warmup + 1,000 retained iterations per chain**. This is the established shorter monthly/block screening budget, shorter than the 1.9.8.1 seasonal screen's 2,000 retained draws. Numerical screen: R-hat<1.05, bulk/tail ESS≥200. Screen runs are not publication convergence evidence. Paper tier is available through `BUCEX_TIER=paper`: four chains, 4,000 warmup +12,000 retained, R-hat<1.01 and ESS≥400.

Every fit includes 30-year (360-month) forecasts with **50,000 simulated future paths**, monthly predictive quantiles from the averaged conditional CDF, 500 posterior-predictive replications, and 95% intervals. Forecast tables also contain central 99% intervals and one-sided 95%/99% quantiles. For the more expensive seasonal/annual aggregation tables, the screen retains 1,000 complete paired future paths (paper:5,000); these never mix parameters or months across draws. Large `fit.bucex` archives are saved for subsequent analysis.

## Recommended parallel split

Run **reference + structural sensitivities on Gallade (90 fits)** and **observation sensitivities on BIOBOT (38 fits)**. These batches do not overlap. Each command also accepts `monthly_all` if you prefer to run everything on one host. Do not run both full batches unless you intend to duplicate the work.

Each Gallade array element runs **one response / one setting**, with its two chains in parallel; default array cap 32 means at most 64 chain workers. Each screen element requests 16 GiB and 12 hours. The walltime is a per-job safety limit, not a completion-time estimate. Monthly p=12 paths are more expensive than seasonal p=4 paths; queue delay and convergence are not guaranteed.

### BIOBOT

Put the release ZIP in your home directory, then:

```bash
cd ~
unzip bucex-1.9.8.3.zip
cd -P ~/bucex-1.9.8.3
conda activate bastiaan
export BUCEX_PYTHON="$(command -v python)"
export BUCEX_RESULTS_ROOT="$PWD/results/monthly1982_biobot"
export BUCEX_CPUS=48
export BUCEX_MAX_JOBS=24
export BUCEX_MEMORY_GB=150
mkdir -p "$BUCEX_RESULTS_ROOT"

bash RUN_MONTHLY_BIOBOT.sh monthly_observation --dry-run
nohup bash RUN_MONTHLY_BIOBOT.sh monthly_observation \
  > "$BUCEX_RESULTS_ROOT/queue.log" 2>&1 < /dev/null &
echo "$!" > "$BUCEX_RESULTS_ROOT/queue.pid"
tail -f "$BUCEX_RESULTS_ROOT/queue.log"
```

The queue adjusts concurrency to both the worker budget and memory reservation; monthly forecasts and state arrays are larger than seasonal ones. It uses available CPUs, not GPUs. If older jobs are still running, reduce these budgets or wait; the queue cannot reserve resources on behalf of unrelated jobs.

Use the existing working environment. If dependencies are missing, run `python -m pip install -e '.[plot]'` in that environment. Runtime sets the extracted source directory on `PYTHONPATH`; no GitHub pull is needed for this ZIP.

```bash
bash RUN_MONTHLY_BIOBOT.sh monthly_observation --status
ls -lt "$BUCEX_RESULTS_ROOT/screen/logs"
```

Ctrl-C stops `tail`, not the background queue. Completed tasks are skipped on an identical rerun. Inspect failures before using `--retry-failed`. Changing the calibration, source or data requires a fresh results root.

### Gallade

Put the ZIP in your home directory and extract into the compute-visible physical data path:

```bash
cd -P /kyukon/data/gent/vo/000/gvo00048/vsc42619/GitHub
unzip "$HOME/bucex-1.9.8.3.zip"
cd -P bucex-1.9.8.3
export BUCEX_RESULTS_ROOT="/kyukon/data/gent/vo/000/gvo00048/vsc42619/bucex1982_monthly_results"
export BUCEX_SCHEDULER=slurm
export VSC_CLUSTER=gallade
export VSC_PROJECT=gvo00048
unset VSC_PARTITION
export VSC_ARRAY_LIMIT=32
```

Reuse the environment that worked previously, if it still exists:

```bash
export BUCEX_VENV="/kyukon/scratch/gent/426/vsc42619/bucex_env_gallade_py311_196_20260929_170136"
export BUCEX_PYTHON="$BUCEX_VENV/bin/python"
export BUCEX_ENV_SETUP="$BUCEX_VENV/environment.sh"

bash RUN_MONTHLY_HPC.sh monthly_structural --dry-run
bash RUN_MONTHLY_HPC.sh monthly_structural
```

Expect a compute-node startup probe, array **1–90%32**, and automatic collection. The probe tests monthly Gaussian, maximum-GEV, minimum-GEV, constant dispersion and fixed seasonality before the array starts. Failed probes block the array. The per-task sampler logs are in `<results-root>/screen/logs`; scheduler logs are in `<results-root>/scheduler_logs`.

If the old environment is unavailable, use this block **instead of** the reuse block:

```bash
export BUCEX_VENV="$VSC_SCRATCH/bucex_env_gallade_py311_1983_$(date +%Y%m%d_%H%M%S)"
bash SETUP_HPC_ENV.sh
export BUCEX_PYTHON="$BUCEX_VENV/bin/python"
export BUCEX_ENV_SETUP="$BUCEX_VENV/environment.sh"
bash RUN_MONTHLY_HPC.sh monthly_structural --dry-run
bash RUN_MONTHLY_HPC.sh monthly_structural
```

Setup runs on a compute node. Do not run the Gallade architecture-specific Python on the login host. Do not use login-only `/user/data` paths for source, environment, logs or results. Results are directed to the data volume here, not scratch.

```bash
squeue -u "$USER"
ls -lt "$BUCEX_RESULTS_ROOT/scheduler_logs"
ls -lt "$BUCEX_RESULTS_ROOT/screen/logs"
```

## Overview and exports

Both launchers collect automatically after fitting, even when some jobs fail. The standalone HTML is:

`<results-root>/screen/collected/<batch>/index.html`

Reference plots include six-panel levels/rates, **one combined plot of all six monthly scale patterns**, PIT/QQ, calendar-month QQ, seasonal changes, prior–posterior comparisons, traces, historical risk, and predictive/location intervals and widths. Sensitivity overlays retain 95% bands. Tables list completion, numerical problems and differences from the reference. No setting is automatically selected as “best”. The BIOBOT-only observation batch cannot show reference plots until the Gallade reference results are combined with it.

Compact review ZIPs are saved under `<results-root>/exports/`. They omit large posterior archives. Keep the `fit.bucex` files on the compute host for the later comparison.

For manual collection on BIOBOT:

```bash
bash COLLECT_MONTHLY.sh monthly_observation
```

For manual collection from the Gallade login node (submits a compute job):

```bash
bash COLLECT_HPC_RESULTS.sh screen monthly_structural
```

To combine the two compact ZIPs on a machine with the working Python environment, from the unchanged 1.9.8.3 source:

```bash
python -m research.monthly.combine \
  /path/to/bucex1982_screen_monthly_structural_TIMESTAMP.zip \
  /path/to/bucex1982_screen_monthly_observation_TIMESTAMP.zip \
  --output results/monthly1982_combined
```

Open `results/monthly1982_combined/screen/collected/monthly_all/index.html`. The merger rejects conflicting duplicate evidence, omits derived host reports and rebuilds a combined overview from the task tables. Screening results should first establish acceptable mixing and marginal adequacy; comparative held-out evaluation comes later.
