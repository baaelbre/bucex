# BUCEX 1.9.2 — separate analyses on biobot

Each task fits **one** of TXm, TNm, TXx, TXn, TNx or TNn. It sees only that response. There is no copula, no shared latent state, no shared innovation hyperparameter and no pooled initial slope. Identical numerical prior settings are reused in six independent analyses. Each task has its own output directory and reproducible random seed.

## Install once

Unzip this release into its own directory; keep the 1.9.1 code and results separately. From `bucex-1.9.2`, use an existing Python environment with Python >=3.10, or create one:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[plot]'
export BUCEX_PYTHON="$VIRTUAL_ENV/bin/python"
export BUCEX_RESULTS_ROOT="$PWD/results/serra_192_parallel"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$BUCEX_PYTHON" -c 'import sys, bucex, numpy, scipy, pandas, matplotlib; print(sys.executable); print("BUCEX",bucex.__version__)'
```

Check that this prints `BUCEX 1.9.2`. The launchers preserve the selected environment. No Slurm, PBS, `VSC_ARRAY_LIMIT` or HPC module is involved in these local commands.

## Screen: all experiments and validation

Run inside a persistent terminal such as a tmux session. On the previously described biobot (36 physical cores, 72 logical CPUs, about 187 GiB RAM), start with **12 simultaneous fits**, each using two chain workers. This is a concurrency setting, not a memory guarantee; reduce it if other work is running or RAM usage rises substantially.

```bash
export BUCEX_MAX_JOBS=12
bash RUN_SCREEN_ALL.sh --dry-run
bash RUN_SCREEN_ALL.sh
"$BUCEX_PYTHON" -m research.seasonal.collect_jobs \
  --tier screen --batch all --root "$BUCEX_RESULTS_ROOT" --figures
"$BUCEX_PYTHON" -m research.seasonal.export_results \
  --tier screen --root "$BUCEX_RESULTS_ROOT" \
  --output bucex-1.9.2-screen-results.zip
```

`RUN_SCREEN_ALL.sh` queues **524 single-response fits** under one concurrency limit: 148 full-record posterior fits, four prospective pre-2019 TXx fits, 354 five-year validation fits and 18 original-design ten-year validation fits. It does not launch 524 processes simultaneously.

If you want the sensitivities first and validation later, use these instead of `RUN_SCREEN_ALL.sh`:

```bash
bash RUN_SCREEN_EXPERIMENTS.sh --dry-run
bash RUN_SCREEN_EXPERIMENTS.sh
bash RUN_SCREEN_VALIDATION.sh
bash bash_scripts/run_local.sh screen validation10
```

The experiments wrapper covers the 148 posterior and four pre-2019 tasks. `--series TXm TNm` restricts any local wrapper to the named responses. This is useful for checking the first results before queuing the other responses. A later unrestricted invocation skips those completed tasks.

## Paper: same scientific comparisons, longer chains

After reviewing the screen, the following runs the **same declared settings and comparisons** at paper budgets. The script does not select a winning prior from the screen or silently promote doubled slope to the reference.

```bash
export BUCEX_MAX_JOBS=6
bash RUN_PAPER_ALL.sh --dry-run
bash RUN_PAPER_ALL.sh
"$BUCEX_PYTHON" -m research.seasonal.collect_jobs \
  --tier paper --batch all --root "$BUCEX_RESULTS_ROOT" --figures
"$BUCEX_PYTHON" -m research.seasonal.export_results \
  --tier paper --root "$BUCEX_RESULTS_ROOT" \
  --output bucex-1.9.2-paper-results.zip
```

Six fits × four chain workers = 24 chain workers. Start there; if actual CPU and RAM usage leave room, eight fits would use 32 chain workers. `BUCEX_MAX_JOBS` controls fit concurrency; numerical-library threads are fixed at one per worker. Run one tier at a time. The local lock prevents overlapping runners within the same root/tier; it does not coordinate different roots or tiers.

For a paper reference only:

```bash
bash bash_scripts/run_local.sh paper reference
```

For paper sensitivities without validation, use `RUN_PAPER_EXPERIMENTS.sh`. The separate validation commands are `RUN_PAPER_VALIDATION.sh` and `bash bash_scripts/run_local.sh paper validation10`.

## Budgets and model settings

| Task | Chains | Warm-up per chain | Retained per chain |
|---|---:|---:|---:|
| Screen, every task | 2 | 700 | 1,300 |
| Paper, alternatives and validation | 4 | 2,000 | 4,000 |
| Paper, six full-record references and TXx pre-2019 reference | 4 | 3,000 | 8,000 |

Full-record forecasts cover 120 seasons (30 years). The prospective 2019 study forecasts one season using data through MAM 2019. Five-year validation uses 20 held-out seasons; original 60%, 80%, 90% splits use 40 held-out seasons. All splits are refitted using their training observations only. Reports use 95% intervals and validation exports central 90%, 95%, 99% coverage plus directional quantiles and threshold scores.

The reference keeps the **marginal normal–lognormal innovation priors of 1.9.1**:

| Quantity | Reference |
|---|---:|
| Level innovation anchor | 0.004836005867750226 |
| Slope innovation anchor | 0.00004836005867750226 |
| Seasonal innovation anchor | 0.004836005867750226 |
| Log scale hyperprior SD | log(3) |
| Initial slope Normal SD, per seasonal update | 0.0125 |
| Initial slope Normal SD, °C per decade | 0.5 |
| Initial level and seasonal coefficient Normal SD | 20 °C |
| GEV shape prior | Normal(0, 0.3²), unbounded |
| Seasonal log observation-scale contrast SD | 0.3 |

For response j and innovation component c, independently across responses and components:

```
log(m[j,c] / anchor[c]) ~ Normal(0, log(3)^2)
s[j,c] | m[j,c] ~ Normal(0, (m[j,c] / 0.6744897501960817)^2)
innovation_SD[j,c] = abs(s[j,c])
```

Each response learns its own m from its own coefficient and observations. No m is shared. Keeping m random preserves the earlier marginal prior; setting it to its anchor would reduce the coefficient's prior RMS by exp(log(3)^2), about 3.34. With one response, the private hyperparameter can remain weakly identified. Judge the induced innovation effect, trajectories and validation as well as the hyperparameter plot; independence is not a guarantee of stronger prior-to-posterior movement.

The initial slope is the actual rate at the beginning of the state trajectory, not the end-of-record warming rate. It stays separately Normal regularized, with half/double SD controls. The 20 °C seasonal value is the initial seasonal coefficient prior SD, not its innovation variance and not an MCMC starting value.

## Comparisons

The grid retains the existing reference, matched-second-moment log(2)/log(4) controls, fixed-anchor log(2)/log(4) controls, old-anchor control, half/double level, slope and seasonal anchors, half/double initial-rate SD, constant observation dispersion, half/double seasonal log-scale SD, fixed location seasonality, initial-seasonal SD 2.25 °C, and shape-prior controls.

It adds:

- `quarter_seasonal`: seasonal anchor ×0.25, stronger shrinkage towards a fixed seasonal pattern.
- `double_slope_half_seasonal`: slope anchor ×2 and seasonal anchor ×0.5.
- `double_slope_quarter_seasonal`: slope anchor ×2 and seasonal anchor ×0.25.

These change the anchors, not the initial seasonal prior or lognormal width. The shape variants (SD 0.15, 0.6, bounded Normal and bounded Uniform) run only for TXx, TXn, TNx, TNn. Running them for the Gaussian means would duplicate an unchanged analysis. `fixed_location_seasonality` retains a fitted seasonal pattern but sets its evolution to zero; it does not remove seasonality.

Reference validation origins: November 1970, 1980, 1990, 2000, 2010, 2015 and 2020. Selected sensitivities, including all three new seasonal controls, use November 2010, 2015 and 2020. The full list is in `research/seasonal/config/experiments.json`. Comparisons are made per response on identical origins and cases, with numerical flags retained. A better CRPS in the old pooled analysis is a reason to test doubled slope here, not evidence that it must win in the new analysis.

## Outputs, restarts and checks

```
results/serra_192_parallel/
  screen/ or paper/
    plan/resolved_settings.csv
    logs/<task_id>.log
    <task_id>/task.json
    <task_id>/resolved_config.json
    <task_id>/report/<response>/...
    collected/all/...
```

Examples: `posterior_reference_TXm`, `posterior_double_slope_TNn`, `forecast_quarter_seasonal_TXx_2015-11`. Every saved `.bucex` archive contains one response. `independent_shrinkage.csv` and its trace/effect tables replace the shared-shrinkage tables; separate slope, scale, shape, risk, serial-residual and prior/posterior diagnostics remain. There are no fitted copula tables, cross-response contrasts or compound-risk estimates. The named single-channel model container is an implementation detail; no second response enters its likelihood or prior.

Re-running the same command skips completed tasks **only if source, configuration and daily-data fingerprints match**. A changed setting or changed source needs a new results root. Completed-but-numerically-flagged tasks are not silently rerun or accepted as converged. A process failure does not stop other queued tasks; the runner exits nonzero after finishing its queue and records which jobs failed.

To retry recorded failures without discarding their first attempt:

```bash
bash RUN_SCREEN_ALL.sh --retry-failed
```

This archives each failed attempt before retrying. A task still marked `running` after a hard interruption is not assumed safe to overwrite: inspect running processes, then use a fresh root for affected work. Fit logs are appended across attempts. Do not edit code or configs while a batch is running. A new paper tier is separate from the screen tier, so both can use the same root sequentially.

For a strict paper numerical gate, after producing the comparison tables:

```bash
"$BUCEX_PYTHON" -m research.seasonal.collect_jobs \
  --tier paper --batch all --root "$BUCEX_RESULTS_ROOT" --require-complete
```

The gate requires all declared tasks and their numerical checks; it does not establish predictive adequacy. A `needs_review` fit remains visible in the reports and must be investigated before using it for manuscript conclusions. The six long reference fits also receive individual `final_check.json` assessments.

The export ZIP includes compact numerical reports, logs, source/configuration manifests and CSV traces, but excludes `.bucex` state archives and figures. Keep the state archives locally for replotting. This preserves the evidence needed for an updated HTML review without uploading every posterior state array.

## Compatibility and provenance

The package still supports joint/copula models for historical work, but the 1.9.2 job grid never calls them. Frozen `reference_191.json` retains the earlier identity/shared-innovation model; older comparison configs use that reference. The monthly workflow remains separate. The recommended 1.9.2 entry points are the root local launchers and `research.seasonal.jobs`.

This release was built from the available saved 1.9.1 source release. The previously uploaded screen results recorded a different source fingerprint, so an exact match to the unprovided biobot working tree is not claimed. The supplied daily data match the previously recorded data checksum. See `RELEASE_VALIDATION.json` for the checks actually run on 1.9.2; production-length fits are for biobot.
