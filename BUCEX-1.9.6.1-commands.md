# BUCEX 1.9.6.1 — joint level–slope calibration

This patch keeps the statistical model from 1.9.6 and adds a focused experiment to
look for a **region of stable inference**, rather than a single setting with the
best CRPS. Both level and slope innovations remain stochastic. Only half-normal
hyperprior scales are varied; initial rates remain separately calibrated.

## Grid and fixed settings

Every combination of these scales is fitted: **25 grid cells**.

| Quantity | Values |
|---|---|
| Level hyperprior scale, Aα | 0.005, 0.01, 0.02, 0.05, 0.10 |
| Slope hyperprior scale, Aβ | 0.0001, 0.0002, 0.0005, 0.001, 0.002 |
| Seasonal hyperprior scale, Aγ | **0.01 throughout** |
| Reference cell | **Aα = 0.01, Aβ = 0.0002** |

These are the scales in `τc ~ HalfNormal(Ac)`, with response coefficients
`s_cj | τc ~ Normal(0, τc²)`. They are not fixed innovation variances.
The reference remains a comparison point, not an already established optimum.
Relative to it, the grid spans 0.5–10 times both hyperprior scales.

Other settings are held fixed:

- Six responses share three innovation shrinkage scales; conditional residual independence, no copula.
- Separate initial-rate SD 0.01 °C per season, equivalent to 0.4 °C per decade.
- Initial-level SD 10 °C; initial seasonal **orthonormal contrast SD 10 °C**, as in 1.9.6.
  This gives marginal seasonal SD `10 sqrt(3/4) = 8.66 °C` and an exchangeable zero-sum cycle.
- Dynamic level, slope and seasonality; seasonal observation scales with log-contrast SD 0.30.
- Observation variance IG(2,2); unbounded Normal GEV shape prior, SD 0.30.
- The existing MH correction is retained. No new orthogonalization or likelihood change.
- Full-record forecasts: 120 seasonal steps / **30 years**, 50,000 future paths.
- Posterior intervals: **95%**. Validation coverage: **90%, 95%, 99%**.
- Screen: **2 parallel chains, each 1,000 warmup + 1,000 retained draws**.
  Numerical thresholds remain R-hat ≤1.05 and ESS ≥200, now also checked for slope
  states and allocation targets. These are screening thresholds, not a claim of convergence.
- Validation uses 10,000 future paths per screen fit.

The focused launchers do not also run the earlier shape, seasonality, private-fit,
monthly or pre-2019 comparisons. The previous `all` suite remains available separately.
Once a stable region is identified, the ordinary nuisance-prior sensitivities should
be revisited around its selected reference.

## Divide the screen between the two hosts

| Host / batch | Work | Fits | CPUs per fit |
|---|---|---:|---:|
| Gallade: `sweetspot_hpc` | All 25 full-record fits, plus all cells at 1990 and 2000 origins | 75 | 2 |
| BIOBOT: `sweetspot_validation` | All cells at 2010, 2015 and 2020 origins | 75 | 2 |
| Combined: `sweetspot` | Both disjoint batches | 150 | 2 |

Each fit jointly updates all six responses. Two chains therefore require **two
workers per fit**, not twelve. At a Gallade array limit of 16, the model array can
use 32 CPUs; BIOBOT with 48 workers can run at most 24 fits concurrently.

Origins are after SON in the stated year. Each hindcast requests 30 years and stops
at the available record. The observed horizons are 120, 103, 63, 43 and 23 seasons
for 1990, 2000, 2010, 2015 and 2020 respectively. The shared scales are refitted
using only each training window. All cells have the same verification cases.
Overlapping origins are not independent validation replicates. Because these folds
inform calibration, describe them as tuning/validation rather than an untouched test set.

### Gallade: reuse the working Python 3.11 environment

Put the ZIP in the VO directory below. Extract into the new version directory;
keep earlier source directories and their results intact.

```bash
cd -P /kyukon/data/gent/vo/000/gvo00048/vsc42619/GitHub
unzip bucex-1.9.6.1.zip
cd -P bucex-1.9.6.1

export BUCEX_VENV=/kyukon/data/gent/vo/000/gvo00048/vsc42619/GitHub/bucex/bucex_env_gallade_py311_193
export BUCEX_PYTHON="$BUCEX_VENV/bin/python"
export BUCEX_ENV_SETUP="$BUCEX_VENV/environment.sh"
export BUCEX_RESULTS_ROOT="$PWD/results/serra_1961_hpc"
export BUCEX_SCHEDULER=slurm
export VSC_CLUSTER=gallade
export VSC_PROJECT=gvo00048
export VSC_ARRAY_LIMIT=16
unset VSC_PARTITION
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

test -x "$BUCEX_PYTHON" && test -r "$BUCEX_ENV_SETUP"
bash RUN_SWEETSPOT_HPC.sh --dry-run
bash RUN_SWEETSPOT_HPC.sh
```

The environment path above is the one that previously worked in this session.
If you moved it, change that single `BUCEX_VENV` assignment to its physical path.
Do not activate the Gallade binary on the login node. Planning uses standard-library
Python on the login node; fitting uses the specified compute environment and this
release's source directory. A previously loaded Python module is unloaded before
the environment's module setup is sourced.

Submission creates a probe, an array of **75 tasks**, and a collection job.
The probe fits tiny examples at the reference and both grid corners, then simulates
1,000 prior replicates for every cell. The model array starts only after the probe
succeeds. Collection runs after the array finishes, also when some elements fail.
It preserves missing-fit and convergence flags.

You can change `VSC_ARRAY_LIMIT` before submission. The scheduler/account limits
still apply. Screen tasks request 12 GiB and six hours each; this is an allocation
limit, not an estimate or a six-hour minimum runtime.

If you need a **new** environment instead of the working one, use this block before
submission, from the extracted release directory:

```bash
export VSC_CLUSTER=gallade VSC_PROJECT=gvo00048
export BUCEX_VENV="$PWD/bucex_env_gallade_py311_1961"
bash SETUP_HPC_ENV.sh
export BUCEX_PYTHON="$BUCEX_VENV/bin/python"
export BUCEX_ENV_SETUP="$BUCEX_VENV/environment.sh"
```

Setup resolves directory aliases before scheduling. It rejects an unresolved
`/user/data` alias and will not overwrite an incomplete environment. Use a fresh,
compute-visible writable directory if a previous setup attempt left one behind.

Progress and logs:

```bash
squeue --clusters=gallade -u "$USER"
ls -lt "$BUCEX_RESULTS_ROOT/scheduler_logs" | head
ls -lt "$BUCEX_RESULTS_ROOT/screen/logs" | head
```

For a failed job, substitute the printed ID:

```bash
sacct --clusters=gallade -j JOB_ID \
  --format=JobID%24,State%20,ExitCode,Elapsed
```

### BIOBOT: recent-origin validation in parallel

Put the ZIP in your home directory:

```bash
cd ~
unzip bucex-1.9.6.1.zip
cd ~/bucex-1.9.6.1
conda activate bastiaan
export BUCEX_PYTHON="$(command -v python)"
export BUCEX_RESULTS_ROOT="$PWD/results/serra_1961_biobot"
export BUCEX_CPUS=48
export BUCEX_MAX_JOBS=24
export BUCEX_MEMORY_GB=150
mkdir -p "$BUCEX_RESULTS_ROOT"

bash RUN_SWEETSPOT_BIOBOT.sh --dry-run
nohup bash RUN_SWEETSPOT_BIOBOT.sh \
  > "$BUCEX_RESULTS_ROOT/overnight.log" 2>&1 < /dev/null &
echo "$!" > "$BUCEX_RESULTS_ROOT/overnight.pid"
tail -f "$BUCEX_RESULTS_ROOT/overnight.log"
```

The launcher uses this directory's source and the existing environment's
dependencies. No environment reinstall is required if your 1.9.6 run worked.
The queue checks CPU availability, limits BLAS to one thread per chain, reserves
memory and prevents a second queue from owning the same results root. Memory
availability can reduce concurrency. Prior simulations run before the fits.
`Ctrl-C` exits `tail`; the detached queue continues.

Status and retries:

```bash
bash RUN_SWEETSPOT_BIOBOT.sh --status
# After inspecting a recorded failure:
bash RUN_SWEETSPOT_BIOBOT.sh --retry-failed
```

Completed matching fits are skipped. Numerical flags are different from execution
failures: `--retry-failed` does not silently replace completed but unconverged fits.
Do not modify source/settings while jobs are running. A changed fit specification
requires a fresh results root.

To run the **entire study on BIOBOT instead of splitting it**, use:

```bash
nohup bash RUN_SWEETSPOT_BIOBOT.sh --batch sweetspot \
  > "$BUCEX_RESULTS_ROOT/overnight.log" 2>&1 < /dev/null &
```

Choose either the split or all-BIOBOT execution; do not launch overlapping queues.
For only the full-record grid on Gallade, the alternative is
`bash RUN_SCREEN_EXPERIMENTS.sh sweetspot_posterior` (25 fits).

## Outputs and review

Both hosts automatically create:

- `<results>/screen/collected/<batch>/sweetspot_review.html`: self-contained HTML,
  with title-free trajectory figures, heatmaps, declared stability checks and
  matched validation tables. BIOBOT-only output has validation tables; the full
  trajectory grid appears in the HPC and combined reports.
- `<results>/exports/bucex1961_screen_<batch>_<timestamp>.zip`: compact evidence
  for review, including figures and HTML. Send both host ZIPs for the combined assessment.
- `<results>/screen/<task>/report/`: fitted reports and posterior archives.
- `<results>/screen/prior_simulations/`: the 25-cell prior calibration checks.

Each grid fit exports paired level/slope variance draws, terminal rates, allocation
fractions at 10 and 30 years, correlations, and local hyperprior sensitivities.
The derivative diagnostic is

`d E[f | y] / d log(Ac) = Cov(f, τc² / Ac² | y)`.

This diagnoses local posterior-mean sensitivity; it is not a replacement for grid
refits. Chain-specific derivative estimates and convergence diagnostics are retained.
The forecast variance decomposition includes the covariance of terminal level and
slope. Fractions attributed to new slope innovations exclude current-state,
seasonal and observation uncertainty; the report labels that distinction.

If automatic collection is interrupted, rebuild without refitting:

```bash
# Gallade: submits collection on a compute node
bash COLLECT_HPC_RESULTS.sh screen sweetspot_hpc --figures

# BIOBOT: run from the release directory/environment
"$BUCEX_PYTHON" -m research.seasonal.finish \
  --root "$BUCEX_RESULTS_ROOT" --tier screen --batch sweetspot_validation
```

To combine downloaded compact exports locally, extract them into **two separate
directories**, each containing `screen/`, then run from this release with numerical
dependencies installed:

```bash
python -m research.seasonal.sweetspot_report \
  --root /path/to/extracted-hpc \
  --other-root /path/to/extracted-biobot \
  --tier screen --batch sweetspot \
  --output /path/to/combined-review
```

The report rejects changed source/configuration/data and duplicate task identities.
Posterior `.bucex` archives are not needed for this combined review; keep those on
the fitting hosts for later analysis.

## How we will judge the result

The report checks every 2×2 rectangle of adjacent grid cells. **All six pairwise
comparisons, including diagonals, must agree for all six responses.** A recent-only
pass is distinguished from stability over the complete record.

| Quantity | Maximum absolute change across settings |
|---|---:|
| Historical level median | 0.10 °C |
| Either endpoint of the level's pointwise 95% interval | 0.20 °C |
| Rate median | 0.05 °C/decade |
| Either endpoint of the rate's pointwise 95% interval | 0.10 °C/decade |
| 10/30-year level and observation forecast medians | 0.20 °C |
| Level forecast interval endpoints | 0.40 °C |
| Observation forecast interval endpoints | 0.50 °C |
| Fixed-threshold predictive risk mean | 0.02 probability |
| Slope fraction of new trend variance: median and interval endpoints | 0.10 |

These are **declared practical screening tolerances**, not universal scientific
cutoffs or significance tests. Missing outputs and numerical flags prevent a pass.
The CSV reports retain the actual differences, so we can assess whether a cutoff is
scientifically appropriate without hiding borderline results. Do not edit the
configuration during fitting to move the goalposts.

Inspect predictive calibration and CRPS separately, across origins, response and
horizon. A similar total level curve is insufficient if its slope decomposition or
long-horizon forecasts remain sensitive. A flat median with changing interval width
is also insufficient. Conversely, Monte Carlo instability must be resolved before
calling a difference substantive.

If the grid reveals a convincing stable region, confirm its boundaries and interior
with longer chains before choosing the paper reference. Do not simply select the
widest prior or the lowest in-sample/hindcast score. If no region appears, report
that honestly and reconsider identification/calibration rather than automatically
broadening again. Reparameterization alone cannot guarantee a stable decomposition.

## Paper tier, after reviewing the screen

The same launchers can run the complete grid at the paper budget:

```bash
# Gallade, using the environment/root exports above
bash RUN_SWEETSPOT_HPC.sh paper --dry-run
bash RUN_SWEETSPOT_HPC.sh paper

# BIOBOT; screen and paper outputs are in separate tier directories
nohup bash RUN_SWEETSPOT_BIOBOT.sh --tier paper \
  > "$BUCEX_RESULTS_ROOT/paper.log" 2>&1 < /dev/null &
```

Ordinary paper cells use four chains, 4,000 warmup and 12,000 retained draws per
chain; the full-record reference uses 6,000 and 20,000. Paper diagnostics use
R-hat ≤1.01 and ESS ≥400. Prior simulations use 5,000 replicates per cell.
The full-record paper reference requests 48 GiB on Gallade; other paper grid fits request 32 GiB.
At 48 workers BIOBOT can run at most 12 four-chain fits; its memory budget may
reduce that further. Full-grid paper commands are provided for completeness;
screen results should guide which cells actually need confirmation.

Runtime depends on the machine, convergence and forecasting/reporting. The screen
uses 1,000 retained draws, not the paper budget, and has no three-hour cutoff.
`python -m research.seasonal.runtime_estimate --batch sweetspot_validation --cpus 48`
prints an approximate wave-based estimate; it excludes queueing and overhead.
