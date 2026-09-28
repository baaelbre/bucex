# BUCEX 1.9.4 — screen runs

This release compares three specifications with independent observation residuals:

- **Shared:** one learned Normal coefficient SD per structural component, shared across the six summaries.
- **Independent mixture:** the same one-response marginal priors, with a private learned SD in each separate fit.
- **Fixed:** six separate fits with fixed Normal coefficient SDs.

All initial rates remain separate. The screen command runs full-record sensitivity only; validation is a separate batch. There are **49 experiments / 159 fits**: 27 shared fits and 22 bundles of six separate fits.

## Reference priors and sampling

For level, slope and seasonal signed innovation coefficients, the direct SD anchors are **(0.01, 0.0001, 0.01)** per seasonal update. The initial rate has fixed Normal SD **0.01**, variance **0.0001**, or **0.4 °C/decade**.

In the two mixture specifications:

    log(tau_c / a_c) ~ Normal(0, log(3)^2)
    s_cj | tau_c ~ Normal(0, tau_c^2)

The independent specification learns one tau_cj per response; the shared specification learns one tau_c across responses. Neither pools the initial rate. The anchors a_c are conditional Normal SDs at the centre of the hyperprior. They are not marginal RMS SDs. Integrating the reference hyperprior multiplies the coefficient RMS SD by exp(log(3)^2) = 3.34327. The independent/shared comparison has exactly matched marginal priors; the fixed Normal comparator has a different marginal prior shape.

There is **no Phi^-1(0.75) conversion** and no automatic recentering when hyperprior width changes. Initial level/seasonal SDs remain 20 °C; the shape prior is Normal(0, 0.3^2), unbounded; seasonal observation log-scale SD is 0.3. The exact-likelihood MH correction is retained, with ASIS off.

Every screen fit uses **2 chains, 2,000 warm-up + 5,000 retained draws per chain**, a 30-year forecast, and 95% intervals. This longer budget is a screen, not a convergence guarantee. Screen flags use R-hat > 1.05 or bulk/tail ESS < 200; paper checks use 1.01 and 400. Shared-scale traces and scientific-target traces are saved, including chain membership. Full fit archives are retained on the compute host.

## HPC — reuse the working Gallade environment

Place the ZIP in `/kyukon/data/gent/vo/000/gvo00048/vsc42619/GitHub/` and extract into its new version directory. This leaves the old source and results available.

```bash
cd -P /kyukon/data/gent/vo/000/gvo00048/vsc42619/GitHub
unzip bucex-1.9.4.zip
cd -P bucex-1.9.4

export BUCEX_VENV=/kyukon/data/gent/vo/000/gvo00048/vsc42619/GitHub/bucex/bucex_env_gallade_py311_193
export BUCEX_PYTHON="$BUCEX_VENV/bin/python"
export BUCEX_ENV_SETUP="$BUCEX_VENV/environment.sh"
export BUCEX_RESULTS_ROOT="$PWD/results/serra_194"
export BUCEX_SCHEDULER=slurm
export VSC_PROJECT=gvo00048 VSC_CLUSTER=gallade SLURM_CLUSTERS=gallade
export VSC_ARRAY_LIMIT=16
unset VSC_PARTITION
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

bash RUN_SCREEN_EXPERIMENTS.sh --dry-run
bash RUN_SCREEN_EXPERIMENTS.sh
```

The release source is placed first on PYTHONPATH on the compute node, so the existing environment imports **BUCEX 1.9.4 from the new directory**. The probe prints its executable and BUCEX version. The login-node submitter uses only Python's standard library and does not execute the Gallade binary on the login CPU.

Submission creates a probe followed by two arrays dependent on successful probe completion:

| Array | Experiments | CPUs per experiment | RAM | Time limit |
|---|---:|---:|---:|---:|
| Shared | 27 | 2 | 16 GB | 12 hours |
| Separate | 22 | 12 | 36 GB | 12 hours |

The shared sampler updates the six channels inside each chain. Two chains therefore use two workers. Each separate bundle runs six responses × two chains. `VSC_ARRAY_LIMIT=16` caps **each array**, not both arrays together. The actual screen configuration is compiled and the matched marginal priors are checked in the probe before either array can run.

If a new environment is needed, use the existing setup mechanism instead of the reuse block:

```bash
export VSC_PROJECT=gvo00048 VSC_CLUSTER=gallade SLURM_CLUSTERS=gallade
export BUCEX_VENV="$PWD/bucex_env_gallade_py311_194"
bash SETUP_HPC_ENV.sh
export BUCEX_PYTHON="$BUCEX_VENV/bin/python"
export BUCEX_ENV_SETUP="$BUCEX_VENV/environment.sh"
```

Do not run the sampling commands on a login node. The probe and arrays run on compute nodes.

### Collect when the arrays finish

From the same release directory with the same environment variables:

```bash
bash COLLECT_HPC_RESULTS.sh screen experiments --figures
```

This submits a collection job. Its log prints the compact ZIP to send back. The ZIP includes configurations, diagnostics, traces and numerical reports, and excludes large posterior state archives. Figures remain under `results/serra_194/screen/collected/experiments/`.

To retry recorded process failures, after confirming the original jobs have ended:

```bash
bash RUN_SCREEN_EXPERIMENTS.sh --retry-failed
```

Completed tasks with matching source, data and configuration are skipped; failed attempts are preserved. An interrupted task still marked `running` requires a fresh results root. A numerically flagged completed fit is not automatically rerun by `--retry-failed`.

## Biobot — optional local execution

From the new release directory, activate the existing Python >= 3.10 environment that already runs BUCEX, then:

```bash
export BUCEX_PYTHON="$(command -v python)"
export BUCEX_RESULTS_ROOT="$PWD/results/serra_194"
export BUCEX_MAX_JOBS=12
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

bash RUN_SCREEN_BIOBOT.sh --dry-run
bash RUN_SCREEN_BIOBOT.sh
```

This allows up to **12 fit tasks × 2 chains = 24 chain workers**. A task is one shared fit or one separate response fit. Local scheduling is bounded by fit tasks, not HPC experiment bundles. Run in a persistent terminal session if disconnecting from SSH.

After completion:

```bash
"$BUCEX_PYTHON" -m research.seasonal.collect_jobs \
  --root "$BUCEX_RESULTS_ROOT" --tier screen --batch experiments --figures
"$BUCEX_PYTHON" -m research.seasonal.export_results \
  --root "$BUCEX_RESULTS_ROOT" --tier screen \
  --output "$BUCEX_RESULTS_ROOT/bucex-1.9.4-screen-review.zip"
```

## What the screen includes

| Check | Settings | Scopes | Experiments |
|---|---|---|---:|
| Structural regularization | Reference; half/double level; half/double slope; half/double seasonal; quarter seasonal; half/double initial rate | All three | 30 |
| Hyperprior width | log(2), log(4), keeping SD anchors fixed | Shared and independent mixture | 4 |
| Other assumptions | Shape SD 0.15/0.6; bounded Normal/uniform shape; constant dispersion; seasonal log-scale SD 0.15/0.6; fixed location seasonality; initial seasonal SD 2.25 | Shared | 9 |
| Influence on pooling | Omit TXm, TNm, TXx, TXn, TNx or TNn | Shared, five summaries per fit | 6 |
| **Total** | | | **49** |

The complete values are in `docs/SENSITIVITY_GRID_194.csv` and `research/seasonal/config/experiments.json`. The compute preflight writes resolved per-task settings to `screen/plan/resolved_settings.csv` and the 72 matched marginal-prior checks to `screen/plan/matched_marginal_priors.csv`.

There are no combined level/slope settings, IG-prior alternatives, historical contrasts, recovery experiments or endpoint-validation jobs in this screen. Within-fit terminal levels and rates remain in the convergence diagnostics because those are scientific outputs whose Monte Carlo precision must be checked.

## Smaller batches and later validation

To run just the three reference specifications first, use `comparison` instead of the default full grid:

```bash
# HPC: 3 experiment jobs, with the same probe gate
bash RUN_SCREEN_EXPERIMENTS.sh comparison --dry-run
bash RUN_SCREEN_EXPERIMENTS.sh comparison

# Or biobot: 13 fit tasks
bash RUN_SCREEN_BIOBOT.sh comparison
```

Use one of those routes; it is not an additional required step. A later full-grid launch skips completed matching reference tasks.

Validation stays separate. `validation` has seven five-year reference origins and selected alternatives at 2010, 2015 and 2020 (43 experiments / 73 fits). `validation10` compares all three references at the 60%, 80% and 90% splits with ten-year forecasts (9 experiments / 39 fits). Hyperparameters are fitted only to each training window. The usual 90%, 95% and 99% coverage checks remain available. On biobot, after the sensitivity runner finishes:

```bash
bash RUN_SCREEN_BIOBOT.sh validation
# Optional original split design:
bash RUN_SCREEN_BIOBOT.sh validation10
```

## Interpretation

The study tests whether pooling stabilizes the results; it does not assume that it will. Shared hyperparameters do not account for residual correlation between summaries from the same temperature record. Interpret pooling alongside its influence checks and marginal diagnostics. Changes in hyperprior width can change mass near zero as well as the upper tail. Review the physically calibrated prior/posterior effects, recent rates, forecasts and tail probabilities together with sampling diagnostics.
