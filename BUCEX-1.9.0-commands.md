# BUCEX 1.9.0 — screen and paper commands

Run these commands from the unpacked `bucex-1.9.0` directory, in your existing
VSC Python environment. Use the same cluster module and project account as
before. Each submission runs the new full-record reference and its 24
sensitivity alternatives. Validation and the pre-2019 record-event fits have
separate commands below.

## Install once

```bash
export BUCEX_PYTHON="$(command -v python3)"
"$BUCEX_PYTHON" -m pip install -e '.[plot,test]'
"$BUCEX_PYTHON" -c 'import bucex; print(bucex.__version__)'
```

This should print `1.9.0`. If your working interpreter is elsewhere, set
`BUCEX_PYTHON` to its absolute path. Keep `VSC_PROJECT` set to your actual
account where required by your cluster. `VSC_ARRAY_LIMIT=8` is the default
simultaneous-task cap **per submitted array**; the paper reference is in a
separate one-element array, so the paper batch can run nine fits at once.
The commands below use the existing PBS-compatible `qsub` launcher.

## Screen: 25 fits

```bash
bash RUN_SCREEN_EXPERIMENTS.sh --dry-run
bash RUN_SCREEN_EXPERIMENTS.sh
qstat -u "$USER"
```

After those jobs finish:

```bash
"$BUCEX_PYTHON" -m research.seasonal.collect_jobs --tier screen --figures
```

Each fit has **2 chains, 700 warm-up and 1,300 retained iterations per chain**.
The request is 2 CPUs, 12 GB and 6 hours per fit. Forecasts span **120 seasons
(30 years)** with 1,500 predictive paths; predictive checks use 400 replicates.
Both the fit archive and tables are saved. Screen diagnostics use R-hat ≤1.05
and ESS ≥100, and are explicitly provisional.

## Paper: 25 fits

```bash
bash RUN_PAPER_EXPERIMENTS.sh --dry-run
bash RUN_PAPER_EXPERIMENTS.sh
qstat -u "$USER"
```

After those jobs finish:

```bash
"$BUCEX_PYTHON" -m research.seasonal.collect_jobs --tier paper --require-complete --figures
"$BUCEX_PYTHON" -m research.seasonal.manuscript_figures \
  --run results/serra_190_parallel/paper/posterior_reference/report \
  --output results/serra_190_manuscript_figures
"$BUCEX_PYTHON" -m research.seasonal.dynamic_comparison \
  --run results/serra_190_parallel/paper/posterior_reference/report \
  --output results/serra_190_dynamic_comparison
```

| Paper fits | Chains | Warm-up / retained per chain | Forecast paths | Request per fit |
|---|---:|---:|---:|---|
| Reference | 4 | 3,000 / 8,000 | 12,000 | 4 CPUs, 32 GB, 24 hours |
| 24 alternatives | 4 | 2,000 / 4,000 | 4,000 | 4 CPUs, 20 GB, 12 hours |

All have 30-year forecasts, 2,000 predictive-check replicates, saved fit
archives and 95% intervals. Paper diagnostics require four chains, R-hat ≤1.01
and ESS ≥400. Requested walltime is a resource limit, not a runtime prediction.
The collector reports missing or numerically flagged fits before comparing
results; `--require-complete` fails if that batch is incomplete or flagged.
A passed numerical gate does not establish scientific adequacy.

Screen and paper use separate result trees. Neither is a continuation of the
other: the paper fits run their own warm-up and retained iterations. The new
scientific reference is not an earlier 1.8.9 fit relabelled as 1.9.0.

## The reference and its 24 alternatives

The four hyperparameters are conditional **median absolute coefficients**.
Writing q=Phi^{-1}(0.75), each signed coefficient conditional on its shared
scale m has Normal(0,(m/q)^2) prior. For each component,
log(m/anchor) ~ Normal(0,tau²), with **tau=log(3)** in the new reference.

| Component | 1.9.0 anchor per seasonal update |
|---|---:|
| Level innovation | 0.004836005867750226 |
| Slope innovation | 0.00004836005867750226 |
| Seasonal innovation | 0.004836005867750226 |
| Initial slope | 0.004836005867750226 |

The anchors are the old (.01, .0001, .01, .01) values multiplied by
exp(log(2)²-log(3)²)=0.4836005867750226. This preserves the four marginal
coefficient second moments while changing the hyperprior shape. The new
95% hyperprior interval is approximately [0.116,8.61] times its anchor,
versus [0.257,3.89] under log(2). Because the anchor moves, this is not simply
an increase in every notion of prior dispersion. The explicit fixed-anchor
comparison below separates those two changes.

| Comparison | Alternatives | Interpretation |
|---|---:|---|
| Hyperprior width | 3 | Old log(2) and wider log(4), with matched second moments; log(3) with the original anchors held fixed |
| Individual anchors | 8 | Half / double level, slope, seasonal and initial-slope anchors, one at a time around the wide reference |
| GEV shape | 4 | Unbounded Normal SD .15 or .60; Normal SD .30 truncated to [-.5,.5]; Uniform[-.5,.5] |
| Scale and seasonality | 5 | Constant observation scale; seasonal log-scale SD .15 / .60; static location seasonality; initial seasonal coefficient SD 2.25 |
| Dependence | 4 | LKJ eta 2 / 4; constant copula; identity copula with the same shared shrinkage |

The reference retains initial level and seasonal coefficient SD 20°C,
unbounded xi~Normal(0,.30²), seasonal log-scale SD .30, and a seasonal Gaussian
copula with LKJ(1) central prior and transformed-partial-correlation SD .25.
The sampler retains exact-likelihood MH correction and ASIS remains off.
The global hyperprior width applies separately to four shared scales; it does
not force the four learned scales to be equal. The package's generic/monthly
API defaults and historical configurations are not silently changed.

Wider hyperpriors allow more mass close to zero and in the upper tail. They do
not guarantee either more or less posterior shrinkage. A posterior resembling
its prior can indicate weak identification, particularly for the slope at the
beginning of the record; it does not by itself identify the best prior. Assess
seasonal evolution, current slopes, period contrasts and forecast risks across
these comparisons. Innovation *variance* is the square of the coefficient SD:
none of the anchors above is a variance parameter.

## Files to inspect

`results/serra_190_parallel/TIER/plan/resolved_settings.csv` gives all resolved
settings. Each task has `resolved_config.json`, `task.json` and a `report/`
directory. The latter contains `fit.bucex`, convergence diagnostics, prior and
posterior summaries, trajectories, shape-support checks and forecast/risk
outputs. Collection writes `collected/posterior/status.json`, `tasks.csv` and
a comparison report under `collected/posterior/posterior/`. The complete
25-fit comparison keeps all tables together and places readable figure groups
under `collected/posterior/posterior/by_study/`, each including the reference.

The collector combines the initial-slope, shared-scale physical effects,
innovation effects and forecast-width tables across variants. The release
also exports, for each response:

- `*_innovation_effects_by_horizon.csv`: posterior SD contributed by future
  level, slope and seasonal innovations over 10 and 30 years (40 / 120 steps).
- `*_innovation_effect_probabilities.csv`: probabilities that these effects
  are below or above 0.05, 0.10 or 0.20°C.
- `*_forecast_uncertainty.csv`: widths of the 30-year predictive intervals.

The innovation effects exclude observation noise and current-state uncertainty.
Their threshold probabilities measure practical magnitude, not the posterior
probability of an exactly constant seasonal pattern. The legacy
`*_innovation_effects.csv` remains a 10-year summary. Shared initial-slope
hyperparameters are correctly converted from median absolute coefficient to
conditional rate SD by 40/q; this fixes a reporting error, not the sampler.
Marginal, serial and copula predictive-check envelopes now all explicitly use
95%, and their exports record that level. Figures use the manuscript style.

## Deferred batches — do not run as part of the commands above

The existing validation design is available separately; it is not needed for
collecting the posterior batch. The reference has seven five-year forecast
origins (1970, 1980, 1990, 2000, 2010, 2015, 2020); 15 selected alternatives use
the 2015 / 2020 origins. This makes 37 validation tasks, with comparisons paired
only over identical origins and forecast cases. The four pre-2019 fits assess
reference, constant dispersion and the two unbounded shape-SD alternatives.
They train through MAM 2019 and predict JJA 2019; their horizon is intentionally
one season. We can revise the validation design before submitting that batch.

```bash
# Later, after discussing the validation design:
bash RUN_SCREEN_EXPERIMENTS.sh validation
bash RUN_PAPER_EXPERIMENTS.sh validation
"$BUCEX_PYTHON" -m research.seasonal.collect_jobs --tier paper --batch validation --require-complete --figures

# Separate prospective record-event analysis:
bash RUN_PAPER_EXPERIMENTS.sh pre2019
"$BUCEX_PYTHON" -m research.seasonal.collect_jobs --tier paper --batch pre2019 --require-complete --figures
```

## Other execution options and retries

For native Slurm, replace each launcher by
`bash bash_scripts/submit_vsc_slurm.sh screen` or `... paper`; the same optional
batch and `--dry-run` arguments apply. `VSC_PARTITION` and `VSC_PROJECT` pass
through when set. The two scheduler frontends declare the same fit budgets.
For a reserved compute node or workstation:

```bash
BUCEX_MAX_JOBS=4 bash bash_scripts/run_local.sh screen
```

For one fit inside an allocated compute session:

```bash
"$BUCEX_PYTHON" -m research.seasonal.jobs --tier screen --task posterior_double_seasonal
```

Completed tasks are skipped only if source, configuration and daily-data
fingerprints match. If a run is interrupted, inspect its log and confirm no
copy is still running. To retry that task without overwriting evidence:

```bash
"$BUCEX_PYTHON" -m research.seasonal.jobs --tier paper --task posterior_double_seasonal \
  --root results/serra_190_retry
```

Batch launchers accept the equivalent `BUCEX_RESULTS_ROOT` environment
variable. Pass the same path with `collect_jobs --root ...`; a retry tree is
collected on its own and is not silently pooled with the original tree.
Do not edit model source/settings mid-batch: the collector rejects mixed
provenance. `--list` shows task IDs; `--batch reference` selects just the
reference; `--batch all` explicitly includes all 66 tasks, including deferred
work. Do not run MCMC on a cluster login node.
