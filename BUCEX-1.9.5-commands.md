> Screening update: use this ZIP in a fresh directory and the fresh results root below. All 106 experiments are retained, with 1,000 warm-up + 1,000 retained draws per chain. No wall-time cutoff. Do not launch the old combined paper+screen queue alongside this run.

# BUCEX 1.9.5 — pooled half-normal experiments on BIOBOT

This is a fresh release and results tree. The default study pools the three innovation-prior SDs across the six summaries. Unpooled fixed priors and separate one-response hierarchies remain available only in the `deferred` batch. They are not in `all`.

## Launch all screening experiments

Upload/extract the new ZIP on BIOBOT. Use the Python >= 3.10 environment that already runs BUCEX. Run these commands from the directory containing the ZIP:

```bash
unzip bucex-1.9.5.zip -d screening_run
cd -P screening_run/bucex-1.9.5

export BUCEX_PYTHON="$(command -v python3)"
export BUCEX_RESULTS_ROOT="$PWD/results/serra_195_screen1k"
export BUCEX_CPUS=32
export BUCEX_MEMORY_GB=150
export BUCEX_MAX_JOBS=32
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

"$BUCEX_PYTHON" -c 'import bucex, scipy, pandas, matplotlib; print(bucex.__version__, bucex.__file__); assert bucex.__version__ == "1.9.5"'
bash RUN_OVERNIGHT_BIOBOT.sh --tier screen --batch all --dry-run

mkdir -p "$BUCEX_RESULTS_ROOT"
nohup bash RUN_OVERNIGHT_BIOBOT.sh --tier screen --batch all \
  > "$BUCEX_RESULTS_ROOT/overnight.log" 2>&1 < /dev/null &
echo "$!" > "$BUCEX_RESULTS_ROOT/overnight.pid"
```

The `nohup` command survives an SSH disconnect. This is a local BIOBOT process queue, not a scheduler submission. It runs a syntax check and three tiny two-chain startup fits first. If those fail, production fits do not start.

The overnight command launches **all 106 screening fits / 212 chains**, with no paper runs. Each chain has **1,000 warm-up and 1,000 retained draws**. Paper fits remain available separately for later. Each fit runs **two chains in parallel**; independent fits run concurrently. A pooled chain updates all six summaries with three shared innovation scales. Splitting the six summaries into separate processes would remove that pooling.

The queue uses one CPU and memory budget across the selected fits. With 32 workers it can run at most 16 two-chain fits; memory reservations can lower that count. `BUCEX_MAX_JOBS=32` is an additional ceiling, not a promise of 32 simultaneous fits. BLAS/OpenMP threads are limited to one per worker. Memory reservations account for worker states, merging and report construction; they are planning estimates rather than OS-enforced limits. Available system memory is also checked before a new job starts. The 32-worker / 150-GiB defaults leave headroom on a large BIOBOT host; change them to the resources actually available to you. Do not launch independent queues against the same results tree.

The seasonal reference and pre-2019 reference start first. Monthly screening fits start early because they are slower; the central half-t and half-Cauchy comparisons follow. All remaining sensitivity and validation jobs stay in the queue. Completed reports are available immediately in each task directory.

**Timing:** plan roughly **5–8 hours** for the screening suite at 32 workers, subject to actual BIOBOT speed and load. The short development benchmark implies about 4.4 hours of idealized sampling work before extra monthly cost, reporting and contention. This is not a BIOBOT timing measurement or a completion guarantee. **There is no eight-hour cutoff:** the queue continues until all jobs and collection finish. Paper budgets are untouched and are not launched by the command above. To benchmark the actual host:

```bash
"$BUCEX_PYTHON" -m research.seasonal.benchmark \
  --output "$BUCEX_RESULTS_ROOT/benchmark" --cpus "$BUCEX_CPUS"
```

Run the benchmark before starting the overnight queue. It uses disposable short chains and does not contribute draws to the study. Its output gives a rough aggregate sampling estimate and the longest individual fit. Monthly fits, mixing, reporting, competing workloads and memory limits can make the actual run longer.

## Monitor from another terminal

Set `BUCEX_RESULTS_ROOT` to the same absolute directory if opening a fresh shell.

```bash
cd -P /path/to/bucex-1.9.5
export BUCEX_RESULTS_ROOT="$PWD/results/serra_195_screen1k"
tail -f "$BUCEX_RESULTS_ROOT/overnight.log"
```

Other useful commands, from the release directory:

```bash
bash RUN_OVERNIGHT_BIOBOT.sh --tier screen --batch all --status

tail -f "$BUCEX_RESULTS_ROOT/screen/logs/posterior_reference.log"
tail -f "$BUCEX_RESULTS_ROOT/screen/logs/pre2019_reference_2019-05.log"
```

`completed` means the process finished writing its report. It does **not** mean convergence passed. The final assessment is in `convergence.json`, `final_check.json` for the paper reference, and the collected `status.json`.

## Fits and chain budgets

All counts below are **per tier**. All active fits use pooled shrinkage. Leave-one-response-out fits pool the remaining five summaries.

| Batch | Fits | Contents |
|---|---:|---|
| `posterior` / `experiments` | 41 | Seasonal reference and the complete full-record sensitivity grid |
| `pre2019` | 6 | Fits ending May 2019: reference, half-t4, half-Cauchy, narrow/wide shape, constant dispersion |
| `validation` | 43 | Seven reference five-year origins and 12 alternatives at three recent origins |
| `validation10` | 9 | Reference, half-t4 and half-Cauchy at 60%, 80%, 90% splits; ten-year forecasts |
| `block_validation` | 6 | Monthly and seasonal fits at 2010, 2015, 2020; identical held-out seasonal targets |
| `monthly` | 1 | Full-record monthly reference for the supplement, with a 30-year forecast |
| **`all`** | **106** | Every row above |
| `core` | 9 | Seasonal reference, reference pre-2019, seven reference validation origins; subset of `all` |
| `comparison` | 3 | Full-record reference, half-t4 and half-Cauchy; subset of `posterior` |
| `deferred` | 12 | Six unpooled fixed fits and six private-hierarchy fits; excluded from `all` |

| Fit | Chains | Warm-up per chain | Retained per chain |
|---|---:|---:|---:|
| Every screen fit | 2 | 1,000 | 1,000 |
| Paper seasonal reference and reference pre-2019 | 2 | 6,000 | 20,000 |
| Other paper seasonal fits and standard validation | 2 | 4,000 | 12,000 |
| Paper matched-block fits and monthly supplement | 2 | 1,500 | 3,000 |

The monthly and matched-block jobs have an explicitly shorter paper budget because they are expensive supplementary checks. They retain the paper convergence thresholds. A flagged fit needs further computation before its results support a claim.

Screen forecasts use 4,000 draws and screen validation uses 2,000. Paper forecasts and validation use 12,000 draws; paper pre-2019 risk uses 20,000. Forecasts cover 120 seasonal transitions or 360 monthly transitions, except pre-2019's one-season prediction. Posterior and predictive intervals are **95%**; validation also reports **90%, 95%, 99%** coverage. Interval widths are estimated from simulation, so they need not rise monotonically at every horizon.

Screen checks flag R-hat above 1.05 or bulk/tail ESS below 200. Paper checks use 1.01 and 400. Initial coefficients, learned scales and scientific targets are checked, rather than relying on attractive trajectories. Short screening chains may fail these checks; their job is to identify problems and sensitivity patterns, not establish manuscript robustness. Screening output is never promoted into the paper tier and is not used to shorten paper warm-up automatically.

## Exact reference hierarchy

For each component c in level, slope and seasonal:

```text
s[c,j] | tau[c] ~ Normal(0, tau[c]^2)
tau[c]         ~ HalfNormal(A[c])
A              = (0.01, 0.0001, 0.01) per seasonal transition
```

`A` is the scale of the underlying zero-centred normal. It equals the marginal RMS signed innovation coefficient after integrating the half-normal scale. There is no median conversion and no lognormal hyperprior in the reference. The legacy lognormal API and old archives retain their original meanings.

The six initial slopes have **separate fixed** `Normal(0, 0.01^2)` priors: `P_beta0 = 1e-4`, or 0.4 °C/decade. They do not share a fourth hyperparameter. Initial level and seasonal-coordinate SDs are 20. The reference keeps the GEV shape `Normal(0,0.3^2)`, the declared observation-scale priors, seasonal dispersion and exact-likelihood Laplace/MH state updates. There is no copula or shared latent warming trajectory.

The three innovation priors imply 30-year displacement SDs of approximately **0.110, 0.075 and 0.077 °C**, respectively. The initial rate contributes **1.20 °C** over 30 years. These are prior standard deviations, not warming estimates. The monthly model is calibrated to the same three 30-year innovation effects and the same initial decadal-rate SD, with its own monthly transition units.

The direct API is:

```python
shared = bx.SharedShrinkage.half_normal(
    {"level": 0.01, "slope": 0.0001, "seasonal": 0.01}
)
```

## Sensitivity grid

The 41 full-record seasonal settings contain:

- 20 structural settings including the reference: half/double individual anchors, initial-rate SD, all three innovations together, all four prior scales together; a complete 3-by-3 level/slope grid; quarter seasonal and combinations of broader slope with tighter seasonal evolution.
- Six hyperprior alternatives: half-t4 and half-Cauchy, each at the central calibration and half/double that scale.
- Nine observation/structural checks: narrow/wide shape priors, bounded normal and bounded uniform shapes, constant dispersion, narrow/wide seasonal log-scale priors, fixed location seasonality and initial seasonal-coordinate SD 2.25.
- Six leave-one-summary-out fits, assessing how individual responses influence the pooled scales and remaining trajectories.

Half-t4 uses scale `A/sqrt(2)` to match the half-normal second moment. Half-Cauchy uses approximately `0.15425251*A` to match the 95th percentile of the shared scale. Its second moment is infinite; calibration reports mark this explicitly and compare quantiles instead of asserting a finite SD. The half/double alternatives test the consequences of that calibration choice.

The exact values and task budgets are in `docs/SENSITIVITY_GRID_195.csv`, `docs/EXPERIMENT_PLAN_195_screen.csv` and `docs/EXPERIMENT_PLAN_195_paper.csv`. The resolved per-task configurations are also saved with the fits. Pooling is evaluated through these checks; stability is not assumed in advance. Hyperparameters are learned afresh within each validation training window.

## Risk, forecast and validation outputs

Every seasonal full-record fit writes levels, slopes, observation scales, innovation effects, prior/posterior comparisons, in-sample PIT/QQ inputs, posterior predictive checks, and a 30-year forecast. Forecast uncertainty reports distinguish observation intervals from latent-location intervals. There are no cross-summary historical contrast claims or compound-risk reports in this study.

Risk reports include the specified hot/cold thresholds, TXx 36.6 and 39.7 °C, period-average risks, stationary-equivalent return periods, and 10/20/50/100-block return levels. The pre-2019 fits write all six observed-versus-predicted JJA summaries and the exact threshold probability curve. Probabilities integrate observation noise analytically within each state/parameter draw; they are not counts of a few simulated rare events.

Annual maxima/minima use products of seasonal CDFs/survival functions **within each paired latent draw**, then integrate posterior uncertainty. Annual return levels solve that conditional aggregate distribution. Only complete **December–November meteorological years** are retained; incomplete boundary years are not silently treated as annual values. Annual mean summaries use day weights. Conditional temporal independence is the aggregation assumption. Return periods are local stationary equivalents, not nonstationary waiting-time predictions.

Validation keeps forecast cases paired across alternatives and writes CRPS, forecast PITs, interval coverage/width, tail misses, negative log predictive density, quantile scores and fixed-event scores. Five-year reference origins end in November of 1970, 1980, 1990, 2000, 2010, 2015 and 2020. Alternative five-year fits use 2010, 2015 and 2020. Monthly/seasonal validation uses the same daily file, cutoffs and seasonal targets. Numerical flags remain attached to comparisons.

## Collection, plots and returning the results

After a tier finishes, the queue automatically collects its reports, simulates the three reference prior families, builds compact figures from the seasonal reference, and creates a dated ZIP under:

```text
results/serra_195/exports/
```

The ZIP contains numerical reports, configurations, logs, diagnostics and generated figures. Large `.bucex` posterior archives stay in the original task directories. Keep those for later analysis. Full fit archives can require substantial disk space; check the chosen results volume before a long run.

You can collect partial results while fits continue, or repeat collection later:

```bash
"$BUCEX_PYTHON" -m research.seasonal.finish --root "$BUCEX_RESULTS_ROOT" --tier screen --batch all
"$BUCEX_PYTHON" -m research.seasonal.finish --root "$BUCEX_RESULTS_ROOT" --tier paper --batch all
```

If the reference has not passed its numerical checks, figures go into `diagnostic_figures_NOT_FINAL` and the status remains visible. Passing convergence alone does not establish prior robustness or predictive adequacy. Send back the dated review ZIPs, especially `collected/all/status.json`, the paired validation reports and prior/risk sensitivity tables.

## Resume, stop, or run a smaller batch

The queue skips completed fits only if source, data and configuration fingerprints match. An ordinary process failure preserves its attempt. After inspecting its log, relaunch the same queue with:

```bash
nohup bash RUN_OVERNIGHT_BIOBOT.sh --tier screen --batch all --retry-failed \
  >> "$BUCEX_RESULTS_ROOT/overnight.log" 2>&1 < /dev/null &
echo "$!" > "$BUCEX_RESULTS_ROOT/overnight.pid"
```

A task left marked `running` is recoverable only when the recorded worker is no longer alive on the same host. A live or uncertain task is not overwritten. A numerically flagged but completed fit is not retried by `--retry-failed`; it needs a reviewed, longer configuration in a fresh results root. Do not edit the source/configuration while fits run.

Graceful stop:

```bash
kill -TERM "$(cat "$BUCEX_RESULTS_ROOT/overnight.pid")"
```

The queue stops its own workers and preserves finished outputs. A second queue cannot acquire the same results-tree lock. For a smaller first night, choose **one** of these instead of the complete launch:

```bash
bash RUN_OVERNIGHT_BIOBOT.sh --tier screen --batch all
bash RUN_OVERNIGHT_BIOBOT.sh --tier paper --batch core
```

Run these sequentially, or use `--tier both --batch core` for one combined queue. Later `--batch all` skips already completed matching tasks. Individual batches use the same command with `posterior`, `pre2019`, `validation`, `validation10`, `block_validation` or `monthly`. Do not run two queues concurrently to bypass the shared resource budget.

The optional later comparisons are explicitly opt-in:

```bash
bash RUN_OVERNIGHT_BIOBOT.sh --tier screen --batch deferred
```

They are retained for future pooling assessment and are not part of the current default overnight study.
