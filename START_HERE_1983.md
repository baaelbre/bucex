# BUCEX 1.9.8.3 — final manuscript runs

The reference is pooled half-normal shrinkage with `(A_alpha, A_beta, A_gamma) = (0.1, 0.002, 0.1)`. Initial level and orthonormal seasonal-coordinate SDs are 10; the separate initial-slope SD is 0.01 per season. The half-normal squared-scale update remains exact GIG. There is no copula and no pooling of initial slopes.

This release targets about **3–4 hours of compute time**, using the measured 1.9.8.x timings. The reference starts first. Other fits run alongside it. There is no three-hour termination timer; queueing, CPU contention and setup can extend elapsed time. A finished fit is not automatically a converged fit.

## Budgets

| Fit | Chains, run in parallel | Warm-up per chain | Retained per chain |
|---|---:|---:|---:|
| Full seasonal reference | 4 | 2,000 | 4,000 |
| Seasonal sensitivity, pre-2019 and validation | 2 | 1,000 | 1,000 |
| Monthly supplementary checks | 2 | 500 | 500 |

The reference yields 16,000 retained draws. Its gate requires rank-normalized split R-hat below 1.01 and bulk/tail ESS of at least 400 for monitored parameters and scientific targets. Screening fits retain their basic numerical checks (R-hat 1.05 / ESS 200); flags remain visible and cannot be interpreted as prior sensitivity without review. The monthly checks are preliminary. Additional iterations may be needed if any important conclusion rests on a flagged fit.

The entire suite lives under the `paper` output directory, but every resolved configuration and task inventory explicitly identifies `sampling_role=reference` or `screen`. The word `paper` here names the experiment suite, not a convergence certificate for every sensitivity fit.

## Everything included by the final launcher

There are **123 fits in 73 HPC experiment groups**:

* **35 seasonal posterior fits:** the pooled reference; the 3×3 local grid `A_alpha ∈ {0.05, 0.1, 0.2}`, `A_beta ∈ {0.001, 0.002, 0.004}`; seasonal anchors 0.05 and 0.2; fixed location seasonality; initial-slope SD ×0.5/×2; initial level/cycle SD 5 and 20; shape SD 0.15/0.6; constant dispersion; seasonal log-scale SD 0.15/0.6; observation-scale priors ×0.5/×2; six separate HN fits and six separate fixed-normal fits at the reference calibration.
* **59 validation fits:** pooled reference origins November 1970, 1975, …, 2020, forecasting up to 35 years (using available held-out data); slope-half/double, shape-narrow/wide, separate HN and fixed-normal comparisons from November 2010, 2015 and 2020. CRPS, PITs, coverage at 90/95/99%, lower/upper misses, and threshold counts/scores use matched forecast cases.
* **17 pre-2019 fits:** pooled reference, slope-half/double, shape-narrow/wide, and six-response separate HN/fixed-normal comparisons, all stopping in May 2019. The risk tables include 35, 36.6 and 39.7°C.
* **6 block-comparison fits:** pooled seasonal/monthly pairs from November 2010, 2015 and 2020, scored on the same seasonal outcomes. Monthly priors match the seasonal 30-year contribution SDs.
* **6 monthly full-record fits:** separate fixed-normal fits with monthly dispersion, at the equivalent reference calibration.

Each full-record fit includes 30-year forecasts, seasonal risks, return levels, annual aggregation, PIT/QQ and posterior predictive tables. Marginal forecasts use 50,000 predictive draws; validation uses 10,000. The reference includes detailed traces, main figures and the strict numerical gate. Prior calibration and complete prior simulations run automatically during collection.

Posterior parameter, trajectory and probability summaries use **means**, with pointwise equal-tailed **95% intervals**. Median columns remain in the CSVs for audit; they are not renamed. Forecast uncertainty tables additionally retain 99% bands and the 95th/99th quantiles. Predictive observation averages are finite Monte Carlo averages; no claim of a finite GEV moment is inferred from them. Quantiles and probabilities retain their usual definitions.

`release_checks/FINAL_RUNS.csv` lists every individual fit and its resolved settings. `FINAL_PLAN.json` records the runtime assumptions: approximately 133 minutes for the reference on BIOBOT or 160 on Gallade; a capacity-constrained estimate for the complete suite is about 175/211 minutes respectively, before setup/collection. These are extrapolations, not new timing measurements. BIOBOT assumes 48 available chain workers and 150 GiB; Gallade can distribute jobs across nodes.

The previous broad exploratory calibration grid, complete monthly sensitivity grid and constant-trend benchmark are not repeated automatically. Existing `sweetspot_*` and `monthly_all` commands remain available. The constant-trend script is included separately under its original name. Half-t, half-Cauchy, lognormal and leave-one-out experiments are not part of this final suite.

## BIOBOT

Extract the downloaded ZIP into a fresh directory. Activate the same Python environment used for the previous successful runs. These scripts import the release in the current directory; no GitHub update is assumed.

```bash
cd ~/bucex-1.9.8.3
python -c 'import bucex; print(bucex.__version__)'
export BUCEX_RESULTS_ROOT="$PWD/results/serra_1983_final"
export BUCEX_CPUS=48
export BUCEX_MEMORY_GB=150
export BUCEX_MAX_JOBS=24
mkdir -p "$BUCEX_RESULTS_ROOT"
bash RUN_FINAL_BIOBOT.sh --dry-run
nohup bash RUN_FINAL_BIOBOT.sh > "$BUCEX_RESULTS_ROOT/overnight.log" 2>&1 < /dev/null &
echo "$!" > "$BUCEX_RESULTS_ROOT/overnight.pid"
tail -f "$BUCEX_RESULTS_ROOT/overnight.log"
```

The reference gets four workers immediately. Long monthly checks start next so they do not become a late final wave. CPU and memory budgets bound subsequent concurrent jobs. Use available capacity; running a second large suite on the same host invalidates these timing assumptions. Ctrl-C stops `tail`, not the detached queue.

```bash
bash RUN_FINAL_BIOBOT.sh --status
tail -f "$BUCEX_RESULTS_ROOT/paper/logs/final_posterior_reference.log"
```

## Gallade / VSC HPC

Extract into a compute-visible directory and enter that directory with `cd -P`. If you already have a working Gallade environment, point `BUCEX_VENV` to it and skip setup. Otherwise:

```bash
cd -P "$VSC_SCRATCH/bucex-1.9.8.3"
export VSC_CLUSTER=gallade
export VSC_PROJECT=gvo00048
export BUCEX_VENV="$VSC_SCRATCH/bucex_env_gallade_py311_1983"
bash SETUP_HPC_ENV.sh
export BUCEX_PYTHON="$BUCEX_VENV/bin/python"
export BUCEX_ENV_SETUP="$BUCEX_VENV/environment.sh"
export BUCEX_RESULTS_ROOT="/kyukon/data/gent/vo/000/gvo00048/vsc42619/bucex1983_final"
mkdir -p "$BUCEX_RESULTS_ROOT"
bash RUN_FINAL_HPC.sh --dry-run
bash RUN_FINAL_HPC.sh
squeue -u "$USER"
```

Setup runs on a compute node. Do not run the Gallade Python on the login node. The physical `/kyukon/data/...` result path avoids the login-only `/user/data` alias and keeps large fit archives off personal scratch. Change it if your allocation uses another compute-visible data directory.

The submission creates a startup probe, then **four arrays** with 1 reference, 53 shared checks, 10 groups of six separate fits, and 9 monthly checks. That is 73 array elements, not 123 submission commands. Each separate group runs six independent two-chain fits on 12 allocated CPUs; no fit exceeds four chains. Check arrays depend on the reference **starting**, not completing. Slurm decides the actual start time and node availability. Each array element requests six hours as a scheduler allowance, not an expected duration or a forced three-hour cutoff. Collection starts after all arrays terminate, and reports failures as well as successes.

HPC logs: `$BUCEX_RESULTS_ROOT/scheduler_logs/`. Per-fit logs: `$BUCEX_RESULTS_ROOT/paper/logs/`. Submission IDs and full commands are saved in `paper/submissions/`.

## Results and recovery

The reference's tables appear at `paper/final_posterior_reference/report/`. Its figures are rendered as soon as that fit completes. They are labelled `diagnostic_figures_NOT_FINAL` unless its strict check passes.

The combined report is `paper/collected/final_paper/index.html`. A compact review ZIP appears in `exports/`; large `fit.bucex` archives stay in the fit directories. Download the compact ZIP for discussion. All results are newly generated: no previous medians or numerical results are relabelled.

To rebuild a partial or completed report without refitting:

```bash
python -m research.seasonal.final_report --root "$BUCEX_RESULTS_ROOT"
```

Run that command on BIOBOT or within a compute allocation on Gallade. It does not run the missing fits. After inspecting any failures and ensuring the earlier queue has stopped, BIOBOT can resume with `bash RUN_FINAL_BIOBOT.sh --retry-failed`; completed fits are skipped. An interrupted process on another host is deliberately not reset automatically. Use a fresh results root if code or scientific settings change.
