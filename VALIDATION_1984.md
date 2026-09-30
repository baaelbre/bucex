# BUCEX 1.9.8.4: forecasts from more origins, evaluated by horizon

This is a focused validation release based on 1.9.8.3. It does not change the
model or replace the full-record reference and sensitivity runs. Keep it in its
own directory while 1.9.8.3 jobs run. Use a new results root; provenance checks
will refuse to mix results from different code, data or configurations.

The default `reference` selection fits the pooled half-normal model with
`A_level=0.1`, `A_slope=0.002`, `A_season=0.1`, initial-slope SD `0.01` per season,
and the other 1.9.8.3 observation and initial-state priors unchanged.

## Validation design

Each fit uses the expanding record from 1892 through November of its origin.
The first prediction is the following DJF. State innovations are simulated
forward without assimilating the verification observations. One fit supplies
all lead horizons; horizons are not separate MCMC jobs.

| November origin | First predicted meteorological year | Last requested year | Available seasonal cases |
|---|---:|---:|---:|
| 1956 | 1957 | 1986 | 120 |
| 1966 | 1967 | 1996 | 120 |
| 1976 | 1977 | 2006 | 120 |
| 1986 | 1987 | 2016 | 120 |
| 1996 | 1997 | 2026 | 119 |
| 2000 | 2001 | 2030 | 103 |
| 2006 | 2007 | 2036 | 79 |
| 2010 | 2011 | 2040 | 63 |
| 2015 | 2016 | 2045 | 43 |
| 2020 | 2021 | 2050 | 23 |

Data end in August 2026, so the final observed season is JJA 2026. Unobserved
future seasons are not generated or scored by this validation-only runner.
The 1996-origin summer subset contains 30 summers; its all-season window has
119/120 seasons. All incomplete windows are labelled.

The report provides:

- Lead bands: years 1–10, 11–20, 21–30.
- Cumulative windows: years 1, 5, 10, 20, 30, with completeness and actual counts.
- Separate DJF, MAM, JJA and SON metrics and year-by-year CSVs.
- CRPS; mean error, MAE and RMSE; central 90%, 95%, 99% coverage, widths,
  and lower/upper misses; forecast PIT histograms for each origin.
- Conditional risk means and 95% intervals, expected and observed threshold
  counts, 95% predictive count intervals and the predictive count distribution.
  JJA TXx > 35 °C gets a dedicated comparison. TXx 36.6 and 39.7 °C are retained
  as additional, retrospectively chosen threshold checks.
- Location and observation forecasts plotted together, with central 95%
  intervals and the 99th predictive percentile.
- Same-calendar-window comparisons between origins: for example, 1997–2006
  at leads 21–30 from 1976 versus leads 1–10 from 1996.
- Per-fit numerical status. Poorly mixed runs remain flagged, not silently
  discarded from the comparison.

Overlapping origins and summaries do not provide independent replications.
The calibration/design was chosen after viewing later observations: these are
retrospective forecast experiments, not untouched prospective validation.
Report the denominators and inspect uncertainty/mixing before interpreting
differences. A historical underprediction pattern alone does not identify a
changepoint or establish its cause.

## Budgets and parallel work

| Tier | Chains per fit | Warmup per chain | Retained per chain | Predictive draws |
|---|---:|---:|---:|---:|
| screen | 2 | 1,000 | 1,000 | 20,000 |
| paper | 4 | 2,000 | 4,000 | 50,000 |

The default suite is **10 fit jobs**, plus an HPC startup check and collector.
Each job runs its chains concurrently. On BIOBOT, 20 available CPUs permit all
ten screening fits to run together. On Gallade the default array cap is ten;
the scheduler decides actual start times. The six responses are updated inside
each pooled chain, not as six independent fits. No chain thinning is added.

Screening targets the same fit budget as the earlier short runs. With enough
resources, elapsed time is approximately that of the slowest fit plus startup
and reporting, not ten fit times. Queue delays, hardware contention and actual
mixing prevent a guaranteed three-hour finish. The six-hour HPC wall request
is a scheduling allowance, not an instruction to run for six hours. More
predictive draws reduce simulation noise; they do not increase MCMC ESS or
force interval widths to increase.

## Gallade

Extract this release into a new directory, for example
`/kyukon/data/gent/vo/000/gvo00048/vsc42619/GitHub/bucex-1.9.8.4`, and run:

```bash
cd /kyukon/data/gent/vo/000/gvo00048/vsc42619/GitHub/bucex-1.9.8.4
export VSC_CLUSTER=gallade
export VSC_PROJECT=gvo00048
export BUCEX_VENV="$PWD/bucex_env_gallade_py311_1984"
export BUCEX_RESULTS_ROOT="/kyukon/data/gent/vo/000/gvo00048/vsc42619/bucex1984_validation"
bash SETUP_HPC_ENV.sh
bash RUN_VALIDATION_HPC.sh screen reference
```

The launcher sets `BUCEX_PYTHON` and `BUCEX_ENV_SETUP` from `BUCEX_VENV` itself.
It does not rely on a child setup script exporting variables into the caller.
This fixes the previous accidental `/environment.sh` path. If you choose a
scratch environment, export its complete path as `BUCEX_VENV` before both
commands. Setup occurs on a compute node, not the login node.

To inspect submission commands without submitting:

```bash
bash RUN_VALIDATION_HPC.sh screen reference --dry-run
```

Scheduler logs: `$BUCEX_RESULTS_ROOT/scheduler_logs/`.
Per-fit logs: `$BUCEX_RESULTS_ROOT/screen/logs/`.
An `afterany` collector builds a report even if some fits fail. Numerical or
task failures are listed in the report. Failed startup checks block the array.

## BIOBOT

Extract into `~/bucex-1.9.8.4`. Activate the existing working Python environment;
the runner puts this checkout first on `PYTHONPATH` without changing the other
checkout or installing into an environment used by running jobs.

```bash
cd ~/bucex-1.9.8.4
conda activate bastiaan
unset BUCEX_PYTHON BUCEX_ENV_SETUP
export BUCEX_RESULTS_ROOT="$HOME/bucex1984_validation"
export BUCEX_CPUS=20
export BUCEX_MAX_JOBS=10
export BUCEX_MEMORY_GB=120
mkdir -p "$BUCEX_RESULTS_ROOT"
nohup bash RUN_VALIDATION_BIOBOT.sh screen reference \
  > "$BUCEX_RESULTS_ROOT/validation.log" 2>&1 < /dev/null &
echo "$!" > "$BUCEX_RESULTS_ROOT/validation.pid"
tail -f "$BUCEX_RESULTS_ROOT/validation.log"
```

Reserve only CPUs and memory not already used by other queues. These limits
apply to this queue, not other running jobs. Use `--dry-run` or `--status` after
`screen reference` to inspect the plan or progress without launching fits.

## Optional fixed-Normal comparison

The model without hyperpriors is available separately. `fixed` launches
60 small jobs: one per response and origin. `all` launches 70 fits, including
the ten pooled ones. This does not rerun the old sensitivity suite and is not
a constant-linear-trend benchmark: the fixed-Normal model still has evolving
level, slope and seasonality.

```bash
# Gallade, optional
bash RUN_VALIDATION_HPC.sh screen all

# BIOBOT, optional; run one queue per results root
bash RUN_VALIDATION_BIOBOT.sh screen all
```

Already completed tasks with matching provenance are skipped by the workers.
Do not start overlapping live queues in the same results root. If adding this
comparison later, wait for the reference batch to finish; collect `all` for
paired pooled-versus-fixed results. `--retry-failed` retries recorded failures,
not successful runs or active attempts. Incomplete interrupted HPC jobs need
inspection before a new results root is used.

## Results and collection

The main standalone HTML is:

```
$BUCEX_RESULTS_ROOT/screen/validation_report_horizon_reference/report.html
```

The compact review ZIP, including CSVs and PNG/PDF figures, is in
`$BUCEX_RESULTS_ROOT/exports/`. Saved `.bucex` posterior archives remain within
individual task directories. The report embeds its figures, so its HTML can
be shared on its own. The ZIP is preferable for further analysis.

On BIOBOT or another host with a working Python environment, rebuild a partial
or complete report without refitting:

```bash
bash COLLECT_VALIDATION.sh screen reference
# If both pooled and fixed batches have finished:
bash COLLECT_VALIDATION.sh screen all
```

On Gallade let the automatic compute-node collector run; do not execute a
Gallade-specific environment on the login node. The original full-record
reference and sensitivity batches remain available under their old batch names.

Settings live in `research/seasonal/config/horizon_validation.json`.
Keep settings fixed once a run starts. A changed design needs a fresh results
root. `paper` runs the same design with four longer chains; it is an option for
confirming forecasts whose screening diagnostics need more sampling.
