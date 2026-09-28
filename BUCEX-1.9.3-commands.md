# BUCEX 1.9.3: fixed priors and six parallel analyses

## Reference specification

TXm, TNm, TXx, TXn, TNx and TNn are fitted separately. Each fit receives only its own response. All six use the same fixed prior settings. There is no copula, parameter pooling, shared hyperparameter or private shrinkage hyperprior in this experiment grid. Innovation coefficients, initial slopes, states, observation scales and applicable GEV shapes are still inferred.

The manuscript specifies Normal prior SDs directly for the three signed innovation coefficients, and a variance for the initial slope:

| Coefficient | Manuscript setting | Normal prior SD |
|---|---:|---:|
| Level innovation, alpha | tau_alpha = 0.01 | 0.01 |
| Slope innovation, beta | tau_beta = 0.0001 | 0.0001 |
| Seasonal innovation, gamma | tau_gamma = 0.01 | 0.01 |
| Initial slope, beta0 | P_beta0 = 0.0001 (variance) | sqrt(P_beta0) = 0.01 |

There is **no median-to-SD conversion**. These settings stay fixed throughout a fit; the innovation coefficients, initial slope and process variances are inferred. The same reference settings apply to all six separate analyses. The initial-rate prior SD is 0.40 degrees C per decade (40 seasonal updates), and its 30-year linear displacement SD is 1.20 degrees C. At 30 years the level, slope and same-season innovation contributions have prior SDs 0.109545, 0.075420 and 0.077460 degrees C, matching the manuscript after rounding.

This corrects the first 1.9.3 archive, which incorrectly carried forward a median-absolute convention and divided every SD by Phi^-1(.75), making it 1.4826 times wider. The correction is tagged `paper_sd_20260928`. Keep earlier results separately; they are a wider-prior experiment, not results under the manuscript reference.

The initial level and initial seasonal-coordinate SDs stay at 20 degrees C. The reference GEV shape prior is unbounded Normal(0, 0.3^2); seasonal log-scale contrast SD is 0.3. Full-record fits cover 538 complete seasonal blocks, MAM 1892 through JJA 2026, and forecast 120 seasons (30 years). Reports use 95% intervals; held-out coverage is recorded at 90%, 95% and 99%.

## Direct prior-SD sensitivity grid

Factors below multiply the **Normal prior SD**, never the variance. An SD factor of 0.5 gives a variance factor of 0.25. All unlisted settings remain at the reference.

| Variant | Level SD factor | Slope SD factor | Seasonal SD factor | Initial-slope SD factor | Other change |
|---|---:|---:|---:|---:|---|
| reference | 1 | 1 | 1 | 1 | Reference above |
| half_all_shrinkage | 0.5 | 0.5 | 0.5 | 0.5 | |
| double_all_shrinkage | 2 | 2 | 2 | 2 | |
| half_level | 0.5 | 1 | 1 | 1 | |
| double_level | 2 | 1 | 1 | 1 | |
| half_slope | 1 | 0.5 | 1 | 1 | |
| double_slope | 1 | 2 | 1 | 1 | |
| half_seasonal | 1 | 1 | 0.5 | 1 | |
| double_seasonal | 1 | 1 | 2 | 1 | |
| half_initial_slope | 1 | 1 | 1 | 0.5 | |
| double_initial_slope | 1 | 1 | 1 | 2 | |
| quarter_seasonal | 1 | 1 | 0.25 | 1 | |
| double_slope_half_seasonal | 1 | 2 | 0.5 | 1 | |
| double_slope_quarter_seasonal | 1 | 2 | 0.25 | 1 | |
| xi_narrow | 1 | 1 | 1 | 1 | Shape SD 0.15 |
| xi_wide | 1 | 1 | 1 | 1 | Shape SD 0.6 |
| xi_bounded | 1 | 1 | 1 | 1 | Normal shape SD 0.3 truncated to [-0.5, 0.5] |
| xi_uniform_bounded | 1 | 1 | 1 | 1 | Uniform shape on [-0.5, 0.5] |
| constant_dispersion | 1 | 1 | 1 | 1 | No seasonal observation-scale contrasts |
| narrow_seasonal_log_scale | 1 | 1 | 1 | 1 | Log-scale contrast SD 0.15 |
| wide_seasonal_log_scale | 1 | 1 | 1 | 1 | Log-scale contrast SD 0.6 |
| fixed_location_seasonality | 1 | 1 | inactive | 1 | Seasonal pattern constant across years |
| previous_initial_season_sd | 1 | 1 | 1 | 1 | Initial seasonal-coordinate SD 2.25 |

The global half/double controls replace learned hyperprior-width controls. Matched-width controls and a return-to-old-anchor variant would be redundant after specifying the fixed Normal family. There are 23 unique specifications including the reference. The existing targeted grid is retained. The additional combined level–slope grid mentioned in the latest manuscript is not part of this grid, and the base IG(2,2) observation-variance prior is not varied. Those manuscript checks remain to be added or the prose narrowed to the experiments actually run.

Every full-record specification runs all six responses. Shape-prior changes have no effect on TXm/TNm, which are Gaussian; those two fits are unchanged controls and must not be counted as additional evidence about shape sensitivity. Stable per-response seeds make that redundancy reproducible. The separate pre-2019 check concerns only TXx and is consequently a one-response exception to the six-fit bundle.

## HPC layout and budgets

One Slurm array element is one experiment, or one validation specification/origin. It launches six independent Python processes. Each process runs two chains concurrently, with numerical-library thread counts fixed at one. Both tiers therefore reserve **12 cores per experiment job**. The pre-2019 TXx jobs retain that resource request but use only two chain workers. Setup and collection jobs use one core; the startup probe uses twelve.

| Tier / experiment | Chains per response | Warm-up per chain | Retained per chain | CPUs per job | Memory | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| Screen | 2 | 700 | 1,300 | 12 | 24 GiB | 6 h |
| Paper sensitivities and validation | 2 | 2,000 | 4,000 | 12 | 64 GiB | 12 h |
| Paper reference and pre-2019 reference | 2 | 3,000 | 8,000 | 12 | 96 GiB | 24 h |

The screen numerical thresholds are R-hat <= 1.05 and ESS >= 100; paper thresholds are R-hat <= 1.01 and ESS >= 400. Both require two chains. A completed scheduler job is not proof of convergence; per-fit numerical flags are retained. If particular fits fail these checks, extend or diagnose those fits before using their results.

| Batch | HPC experiment jobs | Separate response fits |
|---|---:|---:|
| posterior | 23 | 138 |
| reference (subset of posterior) | 1 | 6 |
| pre2019 | 4 | 4 |
| experiments = posterior + pre2019 | 27 | 142 |
| validation: five-year forecasts | 58 | 348 |
| validation10: original 60/80/90% splits, ten years | 3 | 18 |
| all = experiments + both validation batches | 88 | 508 |

Each submission also creates one short compute-node probe. It verifies the configurations and executes a tiny fit for each of the six summaries, with two chains each. Experiment arrays depend on its successful exit. Probe draws are only a startup check, not scientific results. A failed probe prevents the whole grid from starting; its logs identify the response and error.

`VSC_ARRAY_LIMIT` limits concurrent experiment jobs **per resource array**. At 4, a screen array can occupy 48 cores. Paper experiments use two arrays: 25 standard jobs and two long jobs. With a cap of 4, at most four standard plus two long jobs (72 cores) can run together, subject to scheduler limits. Submitting other batches adds separate array caps. Set a smaller cap if needed; do not confuse it with the six analyses inside each job.

## One-time Gallade setup

Extract this release into a fresh directory. Enter that directory and use physical paths. Do not copy or overwrite an older virtual environment. On the login node:

```bash
cd /path/to/bucex-1.9.3
cd -P .
export VSC_CLUSTER=gallade
export VSC_PROJECT=gvo00048
unset VSC_PARTITION
export BUCEX_VENV="$PWD/bucex_env_gallade_py311_193"
bash SETUP_HPC_ENV.sh
```

This command submits a setup job to Gallade and waits for it. Setup loads `Python/3.11.3-GCCcore-12.3.0`, creates a fresh environment, installs the release and checks imports **on a compute node**. Its log is `job_scripts/logs/bx193_setup_<jobid>.log`. If it fails, inspect that log before submitting experiments; a partially created environment is preserved. To retry a failed installation, select a new unused `BUCEX_VENV` path and rerun setup.

Python environments can depend on the cluster architecture. The submission code therefore uses the login node's `/usr/bin/python3` with standard-library modules only; it never invokes the Gallade numerical Python on the login host. This also avoids the earlier Python 3.9 import failure during submission. The installed package itself still requires Python >= 3.10.

Set the paths for each new login session (if you chose a custom `BUCEX_VENV`, use that path):

```bash
export VSC_CLUSTER=gallade
export VSC_PROJECT=gvo00048
unset VSC_PARTITION
export BUCEX_VENV="$PWD/bucex_env_gallade_py311_193"
export BUCEX_PYTHON="$BUCEX_VENV/bin/python"
export BUCEX_ENV_SETUP="$BUCEX_VENV/environment.sh"
export BUCEX_RESULTS_ROOT="$PWD/results/serra_193_paperpriors"
export BUCEX_SCHEDULER=slurm
export VSC_ARRAY_LIMIT=4
```

Do not source `BUCEX_ENV_SETUP` or activate this cluster environment on the login node. Batch scripts load it on compute nodes. No hard-coded partition name is required. The explicit `--clusters=gallade` submission prevents an inherited cluster setting from selecting Skiddo.

## Screen submission

```bash
bash RUN_SCREEN_EXPERIMENTS.sh --dry-run
bash RUN_SCREEN_EXPERIMENTS.sh
```

This submits the 23 full-record settings and four prospective pre-2019 TXx checks, without held-out validation. It prints the probe ID and array IDs. For the full grid including both validation designs, use `RUN_SCREEN_ALL.sh` **instead**:

```bash
bash RUN_SCREEN_ALL.sh --dry-run
bash RUN_SCREEN_ALL.sh
```

Alternatively add validation later:

```bash
bash RUN_SCREEN_VALIDATION.sh
bash bash_scripts/submit.sh screen validation10
```

## Paper submission

After reviewing screen diagnostics and sensitivities, run the same predeclared grid with the longer paper budget:

```bash
bash RUN_PAPER_EXPERIMENTS.sh --dry-run
bash RUN_PAPER_EXPERIMENTS.sh
```

Use `RUN_PAPER_ALL.sh` instead for all 88 experiments, or add held-out validation separately:

```bash
bash RUN_PAPER_VALIDATION.sh
bash bash_scripts/submit.sh paper validation10
```

Screen and paper write separate subdirectories beneath the same results root. Do not submit overlapping batches concurrently for the same tier. Each bundle has a process lock; per-response manifests also reject source/configuration/data mismatches. Completed fits with matching provenance are skipped. Numerically flagged fits remain visible and are not silently treated as accepted paper fits.

## Logs, retry and collection

Scheduler logs are in `job_scripts/logs/`. Detailed response logs are in `$BUCEX_RESULTS_ROOT/<tier>/logs/`; bundle outcomes are in `<tier>/bundles/<experiment>/bundle.json`; every response has its own `task.json`, resolved config and report. Submission IDs, commands and bundle plans are stored in `<tier>/submissions/`. Startup checks are in `<tier>/startup_probes/`.

For a failed Slurm job, inspect its scheduler log and the indicated response log. On Gallade, explicit cluster queries are:

```bash
squeue --clusters=gallade -u "$USER"
# Replace JOB_ID with the printed probe or array ID:
sacct --clusters=gallade -j JOB_ID --format=JobID%24,State%20,ExitCode,Elapsed
```

To retry **recorded failed** response fits after resolving a transient runtime issue, resubmit the same batch with `--retry-failed`. Completed matching fits are skipped, and failed attempts are archived. Interrupted fits still recorded as running require inspection and a fresh results root; they are not overwritten automatically. A scientific configuration, source or data change also requires a new results root. Resume does not extend a completed chain.

After the relevant jobs have finished, collect on a compute node:

```bash
bash COLLECT_HPC_RESULTS.sh screen experiments --figures
bash COLLECT_HPC_RESULTS.sh paper experiments --figures
```

Use batch `all` if you ran all three batches; `validation` or `validation10` collects only that design. Collection logs report the tables/figures directory and a compact review ZIP. The ZIP includes CSV/JSON/log evidence from the tier and excludes posterior state archives and figures. Keep the `.bucex` files on the HPC for later reporting or copy them explicitly when needed. Figures remain under `<tier>/collected/<batch>/`.

After inspecting results, `--require-complete` is an additional paper-only collection gate. It rejects incomplete, provenance-mismatched or numerically flagged results; it never certifies scientific predictive adequacy. Ordinary collection is allowed to report incomplete runs.

## Validation and interpretation

The reference five-year forecasts use origins 1970, 1980, 1990, 2000, 2010, 2015 and 2020 (end of SON). Seventeen sensitivity variants use the common 2010/2015/2020 origins. Comparisons pair the same response, origin and target observations; incomplete origin groups are not ranked. The original split design uses 60%, 80%, 90% of the 538-block record and a ten-year horizon. Future forecasts from the full fit remain thirty years.

Compare initial-slope and innovation prior/posterior distributions, level and rate trajectories, the terminal rate, 10-/30-year innovation contributions and exceedance probabilities. For seasonal shrinkage, inspect both quarter/half SDs and fixed seasonality. A continuous prior can concentrate innovations close to zero but does not give exact posterior probability to zero; the fixed-seasonality fit provides that separate model comparison. For shape sensitivity, compare risk probabilities and upper-endpoint summaries where applicable.

No hyperparameter prior/posterior plots are produced because there are no sampled shrinkage hyperparameters. `fixed_prior_settings.csv` records the declared SDs and their calibration; `initial_slope_prior_posterior.csv` and the coefficient/innovation plots still assess learning. CRPS, bias, PITs, coverage and interval widths remain marginal predictive checks. These separate fits do not establish cross-summary joint risks or uncertainty for paired cross-summary contrasts.

Existing local/biobot execution remains available via `bash bash_scripts/run_local.sh screen experiments` and `paper experiments`; the root `RUN_*` launchers in 1.9.3 now submit HPC jobs. The local runner schedules individual response fits, not HPC bundles. Its dry run reports the requested concurrency and chain-worker budget.

Official cluster references used for the setup design:

- [UGent Python environments](https://docs.hpc.ugent.be/Linux/setting_up_python_virtual_environments/)
- [UGent available Python modules](https://docs.hpc.ugent.be/Linux/available_software/detail/Python/)
- [Slurm sbatch and dependency options](https://slurm.schedmd.com/sbatch.html)
