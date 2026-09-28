# BUCEX 1.9.1 — HPC sensitivities and biobot validation

Use the unpacked `bucex-1.9.1` directory on each machine. Keep each release in
its own directory so running jobs retain their source. The commands below use
your existing working Python/module environment. No jobs have been submitted
as part of preparing this release.

## Install once on each machine

```bash
export BUCEX_PYTHON="$(command -v python3)"
"$BUCEX_PYTHON" -m pip install -e '.[plot]'
"$BUCEX_PYTHON" -c 'import bucex; print(bucex.__version__)'
export BUCEX_RESULTS_ROOT="$PWD/results/serra_191_parallel"
```

The version must print `1.9.1`. Set `BUCEX_PYTHON` to the absolute path of your
working interpreter if it is not `python3`. The explicit results root prevents
an inherited 1.9.0 environment variable from selecting the old results tree.
Load your usual HPC modules before these commands; no module names or account
allocations are guessed by the release.

## Screen — sensitivities on HPC

```bash
export BUCEX_PYTHON="$(command -v python3)"
export BUCEX_RESULTS_ROOT="$PWD/results/serra_191_parallel"
export BUCEX_SCHEDULER=slurm
export VSC_ARRAY_LIMIT=8

bash RUN_SCREEN_EXPERIMENTS.sh --dry-run
bash RUN_SCREEN_EXPERIMENTS.sh
squeue -u "$USER"
```

This submits **23 full-record fits**: the reference and 22 alternatives. Each
uses 2 chains, 700 warm-up and 1,300 retained draws per chain. Each request is
2 CPUs, 12 GB RAM and 6 hours. Eight fits can run simultaneously. The dry run
compiles the declarations and prints the exact submission without scheduling.

After completion, collect on HPC or on a machine with the same release and
result tree:

```bash
"$BUCEX_PYTHON" -m research.seasonal.collect_jobs \
  --tier screen --batch posterior --root "$BUCEX_RESULTS_ROOT" --figures
```

The collector records missing/failed tasks and numerical flags. Screen
estimates remain provisional even when all jobs finish.

## Screen — validation on biobot

Run in a persistent terminal/session on biobot, from the 1.9.1 directory:

```bash
export BUCEX_PYTHON="$(command -v python3)"
export BUCEX_RESULTS_ROOT="$PWD/results/serra_191_parallel"
export BUCEX_MAX_JOBS=8

bash RUN_SCREEN_VALIDATION.sh validation --dry-run
bash RUN_SCREEN_VALIDATION.sh
"$BUCEX_PYTHON" -m research.seasonal.collect_jobs \
  --tier screen --batch validation --root "$BUCEX_RESULTS_ROOT" --figures
```

This runs **52 expanding-window fits** with at most eight simultaneous fits,
using at most 16 chain workers. It is a foreground workstation runner, not a
scheduler submission. It uses the same screen MCMC budget as the posterior
fits. Every origin is fitted from its training data; saved full-record fits
are not reused for validation.

The reference is assessed at November 1970, 1980, 1990, 2000, 2010, 2015 and
2020. Fifteen alternatives are compared at the same November 2010, 2015 and
2020 origins. Every forecast covers 20 complete seasons (five years).
These default windows preserve the previous five-year design. The ten-year
checks below are a separate study.

## Paper — sensitivities on HPC

Run after reviewing the screen results. Paper fits start afresh in a separate
`paper/` directory; they do not append draws to screen fits.

```bash
export BUCEX_PYTHON="$(command -v python3)"
export BUCEX_RESULTS_ROOT="$PWD/results/serra_191_parallel"
export BUCEX_SCHEDULER=slurm
export VSC_ARRAY_LIMIT=8

bash RUN_PAPER_EXPERIMENTS.sh --dry-run
bash RUN_PAPER_EXPERIMENTS.sh
squeue -u "$USER"
```

| Paper task | Chains | Warm-up / retained per chain | CPUs | RAM | Walltime |
|---|---:|---:|---:|---:|---:|
| Reference | 4 | 3,000 / 8,000 | 4 | 32 GB | 24 h |
| Each of 22 alternatives | 4 | 2,000 / 4,000 | 4 | 20 GB | 12 h |

The reference is a separate one-element array. The cap applies to each array,
so up to nine fits may run concurrently. Walltimes are limits, not runtime
predictions. After completion:

```bash
"$BUCEX_PYTHON" -m research.seasonal.collect_jobs \
  --tier paper --batch posterior --root "$BUCEX_RESULTS_ROOT" --require-complete --figures
"$BUCEX_PYTHON" -m research.seasonal.manuscript_figures \
  --run "$BUCEX_RESULTS_ROOT/paper/posterior_reference/report" \
  --output "$BUCEX_RESULTS_ROOT/paper/manuscript_figures"
```

Paper collection with `--require-complete` rejects incomplete or numerically
flagged fits. The thresholds are four chains, R-hat at most 1.01 and bulk/tail
ESS at least 400 for assessed quantities. A numerical pass does not establish
forecast calibration.

## Paper — validation on biobot

```bash
export BUCEX_PYTHON="$(command -v python3)"
export BUCEX_RESULTS_ROOT="$PWD/results/serra_191_parallel"
export BUCEX_MAX_JOBS=6

bash RUN_PAPER_VALIDATION.sh validation --dry-run
bash RUN_PAPER_VALIDATION.sh
"$BUCEX_PYTHON" -m research.seasonal.collect_jobs \
  --tier paper --batch validation --root "$BUCEX_RESULTS_ROOT" --require-complete --figures
```

The same 52 fits use 4 chains, 2,000 warm-up and 4,000 retained draws per chain.
Six simultaneous fits use at most 24 chain workers. This leaves headroom on
biobot for memory and other work. Run one local batch at a time; limits are
per launcher, so two launchers would add their resource use.

## Optional — original 60%, 80%, 90% splits with ten-year forecasts

These are **three additional reference fits**, all with identity dependence
and separate initial-rate priors. Training uses `floor(fraction * 538)` complete
seasonal blocks and each forecast contains 40 seasons. The actual calendar
cutoffs are recorded in `plan/resolved_settings.csv`. These are the original
fractions applied to the updated seasonal record, not the original manuscript's
old calendar cutoffs.

On biobot, after the corresponding five-year batch finishes:

```bash
# Screen version
BUCEX_MAX_JOBS=3 bash RUN_SCREEN_VALIDATION.sh validation10
"$BUCEX_PYTHON" -m research.seasonal.collect_jobs \
  --tier screen --batch validation10 --root "$BUCEX_RESULTS_ROOT" --figures

# Paper version, when ready
BUCEX_MAX_JOBS=3 bash RUN_PAPER_VALIDATION.sh validation10
"$BUCEX_PYTHON" -m research.seasonal.collect_jobs \
  --tier paper --batch validation10 --root "$BUCEX_RESULTS_ROOT" --require-complete --figures
```

Five- and ten-year designs are collected separately, including when collecting
`--batch all`. Comparisons are paired within a design and identical origins.
These additional checks and other overlapping windows are not independent
replications and should not be pooled as such.

## Scientific specification

All active posterior, sensitivity, validation and pre-2019 tasks use
`analysis: joint` with a fixed identity correlation matrix. This is conditional
independence of the observation residuals. The six trajectories still share
three innovation-scale hyperparameters. No residual correlation parameters
are estimated, and there is no shared initial-rate hyperparameter.

Writing `b0_j = 40 * beta0_j` in degrees C per decade:

- Reference: separate `b0_j ~ Normal(0, 0.5^2)` priors.
- Initial-rate sensitivities: SD 0.25 and 1.0 C/decade.
- In seasonal-update units the corresponding SDs are 0.00625, 0.0125 and 0.025.
- `beta0_j` is the rate before the first block, not an overall mean or the
  end-of-record rate. The sampler estimates it separately for every summary.

The three shared innovation scales retain the 1.9.0 reference:

| Component | Hyperprior median anchor, per seasonal update |
|---|---:|
| Level innovation SD | 0.004836005867750226 |
| Rate innovation SD | 0.00004836005867750226 |
| Seasonal innovation SD | 0.004836005867750226 |

The log-scale hyperprior SD is `log(3)`. Given a shared scale m, the signed
innovation coefficient has SD `m / Phi^-1(0.75)`. These are anchors for SDs,
not variances. Initial level and initial seasonal coefficient SDs remain 20 C;
seasonal observation log-scale SD remains 0.30; shape has the unbounded
Normal(0,0.30^2) prior. Exact-likelihood MH correction is retained, with ASIS
off. The posterior fit forecasts 120 seasons (30 years).

| Sensitivity group | Alternatives | Task names / settings |
|---|---:|---|
| Innovation hyperprior shape | 5 | `narrow_log2_matched`, `wide_log4_matched`; `narrow_log2_fixed_anchors`, `wide_log4_fixed_anchors`; `original_innovation_anchors` |
| Innovation anchors | 6 | `half_` / `double_` level, slope and seasonal anchors, one at a time |
| Initial rates | 2 | `half_initial_slope`, `double_initial_slope`: fixed SD 0.25 / 1.0 C per decade |
| GEV shape | 4 | `xi_narrow` (.15), `xi_wide` (.60); bounded Normal(.30) and Uniform on [-.5,.5] |
| Scale and seasonal structure | 5 | Constant observation scale; seasonal log-scale SD .15 / .60; fixed location seasonal cycle; initial seasonal coefficient SD 2.25 |

The matched-width controls preserve innovation coefficient second moments and
shift their anchors. The fixed-anchor controls change only width. Neither
changes the fixed initial-rate prior. `original_innovation_anchors` restores
(.01,.0001,.01) for the three innovations while retaining log(3) width and the
new initial-rate prior. It is not an old copula fit relabelled as a new result.

The validation alternatives are the two fixed-anchor widths, the original
innovation anchors, six individual innovation anchors, two initial-rate SDs,
fixed seasonality, constant dispersion and two unbounded shape-SD alternatives.

## Outputs and interpretation

Each task has `resolved_config.json`, `task.json`, a saved fit and a `report/`
directory. Separate trees are used for `screen` and `paper`. Reports retain:

- Prior/posterior comparisons for the three shared scales and six signed
  initial rates; the shared-scale figure now has three panels.
- Levels, rates, within-summary period changes and marginal threshold risks.
- Innovation contribution SDs over 10 and 30 years and practical-effect
  probabilities; predictive forecasts have 95% intervals.
- Validation CRPS/log scores, signed errors, predictive PITs, threshold scores,
  event counts, and coverage at 90%, 95% and 99%.
- Forecast-origin level and rate diagnostics in `targets_ORIGIN.csv`, allowing
  us to distinguish a warm level shift from a steeper extrapolation.
- Serial and cross-summary residual checks. Identity does not claim to remove
  residual dependence. Seasonal labels now use DJF/MAM/JJA/SON consistently
  under identity, constant and seasonal dependence specifications.

Main manuscript figure generation omits copula matrices and compound-event
risk panels. Cross-summary contrasts are disabled in this workflow. The
internal product joint log score and ordering diagnostics remain audit outputs;
they do not establish an adequate model for compound extremes. Generic copula
classes and frozen historical configurations remain available for reproducing
older releases; active 1.9.1 jobs enforce independence.

Screen uses 1,500 forecast draws and 400 PPC replicates; paper alternatives
use 4,000 / 2,000; the long paper reference uses 12,000 / 2,000. Forecast
validation uses 1,500 or 4,000 predictive draws per origin.

## HPC startup and retries

The main wrappers prefer native `sbatch` when available. `BUCEX_SCHEDULER=pbs`
selects the retained qsub frontend. Set `VSC_PROJECT` or `VSC_PARTITION` only to
your actual allocation/partition; the native Slurm launcher passes them through.
The scheduler scripts use non-login shells, retain the submitting environment,
use physical absolute working paths and print the host, interpreter, release,
batch and task index before fitting. PBS also exports the active environment.

The previous `0:53` accounting output alone did not identify its cause. These
startup changes and the explicit logs make the next failure diagnosable; they
are not evidence that a cluster-side failure has been resolved. Slurm logs are:

```text
job_scripts/logs/screen_posterior_standard_JOBID_ARRAYINDEX.log
job_scripts/logs/paper_posterior_standard_JOBID_ARRAYINDEX.log
job_scripts/logs/paper_posterior_long_JOBID_ARRAYINDEX.log
```

Biobot writes `job_scripts/logs/local_TIER_BATCH_INDEX.log`. To inspect Slurm:

```bash
sacct -j JOBID --format=JobID%24,JobName%24,State%20,ExitCode,Elapsed
```

Completed tasks are reused only when source, settings and daily-data hashes
match. Failed/incomplete tasks are not silently overwritten. After checking
that no copy is still running, retry into a fresh root, for example:

```bash
# Run this only inside a compute allocation or on biobot.
"$BUCEX_PYTHON" -m research.seasonal.jobs --tier screen \
  --task posterior_half_initial_slope --root results/serra_191_retry
```

A retry tree is collected separately. Do not pool different provenance or edit
settings during a running batch. The optional prospective JJA-2019 record
analysis remains four fits under `--batch pre2019`; it is not part of the
posterior or validation commands above.
