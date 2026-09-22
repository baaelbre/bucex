# Start with BUCEX 1.8.0

Run from the extracted `bucex-1.8.0` directory, beside `pyproject.toml`.
The existing univariate API and saved fits remain supported.

```bash
python -m pip install -e ".[plot]"
python -c "import bucex; print(bucex.__version__, bucex.__file__)"
```

Expect **1.8.0** and the path of this checkout. Four chains run in four local
processes; no Slurm installation is needed on biobot. Numerical thread pools
are limited inside each chain. Candidates and forecast origins run sequentially.
Do not launch several large joint jobs before checking available memory.

## 1. Check the complete pipeline

```bash
python -m research.serra.prior_assessment --config research/serra/config/hierarchy/smoke.json --stage all
```

This checks all four candidates, six responses, monthly observation scales,
the Gaussian copula, hyperparameter reports and a held-out forecast year.
It uses January 2023–August 2026, three warmup iterations and four retained draws
per chain. **Its results are only an execution check.** Convergence warnings
are expected and are retained. It creates a timestamped assessment directory.

## 2. Inspect the planned experiment

```bash
python -m research.serra.prior_assessment --config research/serra/config/hierarchy/pilot.json --stage plan
```

All four specifications fit the same 1,616 monthly observations, January
1892–August 2026. Each has private FS level, slope and seasonality, repeating
monthly observation scales, fixed-in-time GEV shapes and an estimated constant
Gaussian copula. No SSVS, ASIS, shared trajectory or residual-memory process is
introduced. Only the innovation prior changes:

| Candidate | Level median or hyperprior anchor | Slope median or hyperprior anchor | Learned common medians? |
|---|---:|---:|---|
| `fixed_half` | .005 | .000025 | No |
| `fixed_quarter` | .0025 | .0000125 | No |
| `pooled_quarter` | .0025 | .0000125 | Yes, level and slope |
| `pooled_half` | .005 | .000025 | Yes, level and slope |

The first two specify medians of each physical innovation SD. In the pooled
models these values anchor lognormal hyperpriors: `log_sd = log(2)`, placing
about 95% of each shared median's prior between 0.257 and 3.89 times its anchor.
Each response retains its own innovation SD. Conditional coefficient priors
are normal; after integrating the hierarchy they are normal scale mixtures.

The seasonal-innovation prior median stays .02; initial-slope prior SD stays
.0025 per month; monthly log-scale contrast prior SD stays .3. Shape has a
Normal(0, .3²) prior truncated to [-.5, .5]. These settings are identical across
candidates. See [SHARED_SHRINKAGE.md](docs/SHARED_SHRINKAGE.md) for the equations.

The pilot uses **500 warmup + 500 retained draws per chain**, with four chains.
There are four full-record joint fits and sixteen historical joint fits:

| Training ends | Held-out forecast period |
|---|---|
| December 2000 | 2001–2005 |
| December 2010 | 2011–2015 |
| December 2015 | 2016–2020, including the 2019 heat record |
| December 2020 | 2021–2025 |

Each origin estimates its own posterior using training data only, including
the common hyperparameters. These are five-year forecasts from the origin,
not rolling one-month forecasts. Few origins give descriptive comparisons,
not precise model-ranking uncertainty. No simulation study is run.

## 3. Run the posterior comparison first

```bash
python -m research.serra.prior_assessment --config research/serra/config/hierarchy/pilot.json --stage sensitivity
```

Keep the printed `Assessment directory`. Read the reports before committing
to all sixteen historical refits. If the chains show no useful movement or
major between-chain disagreement, resolve that before interpreting the plots.
A 500-draw budget is a screen, not a guarantee of adequate effective sample size.

Continue the same assessment with forecasts:

```bash
python -m research.serra.prior_assessment --run YOUR_ASSESSMENT_DIRECTORY --stage predictive
```

Replace `YOUR_ASSESSMENT_DIRECTORY` with the path printed above. Alternatively,
`--config research/serra/config/hierarchy/pilot.json --stage all` runs both
stages. Do not do both if you already started the first stage.

## 4. Read the comparison

The assessment's `comparison/` directory contains:

| Files | Question |
|---|---|
| `convergence.csv`, `joint_mcmc.csv` | Are physical SDs, initial slopes, monthly scales, hyperparameters and copula parameters mixing? Detailed per-fit warnings remain in `convergence.json`. |
| `shared_shrinkage.csv`, `shared_shrinkage_updates.csv`, `shared_shrinkage.png` | What do all six responses say about common regularization? Does shifting the anchor materially change its posterior? |
| `prior_updates.csv`, `*_prior_posterior.png` | How do the individual innovation SDs update from their unconditional priors? |
| `scientific_targets.csv`, `joint_scientific_targets.csv` | Are level changes, period slope contrasts and cross-response differences robust? |
| `*_level.png`, `*_slope.png`, `*_risk.png` | Does the smoother representation preserve conclusions and event probabilities? |
| `predictive_comparison.csv`, `scores_by_origin.csv` | Compare CRPS/log scores on identical held-out observations, including first-year versus later horizons. |
| `joint_log_scores_comparison.csv`, `compound_heat_scores_comparison.csv` | Does the specification improve joint prediction and simultaneous hot-condition probabilities? |
| `coverage_by_month.csv`, `pit_by_month.csv`, `*_forecasts.png` | Are intervals calibrated by season, and are important held-out observations missed? |
| `event_counts.csv` | How many actual threshold events inform each marginal risk score? A rare-event score alone can favor predicting almost no events. |
| `shared_shrinkage_forecasts.csv` | How were common medians learned at each historical origin? |

Within `sensitivity/pooled_quarter/joint/` and `pooled_half/joint/`, inspect
`shared_shrinkage_traces.png`, `shared_shrinkage_traces.csv.gz`, the channel
trace/PIT/QQ plots, `residual_serial.csv`, `residual_dependence.csv` and
`ordering_*.csv`. The comparison uses compact reports; pilot fits are not saved
as large `.bucex` archives. Set `save_fits: true` before a new run if wanted.

All scores are losses: lower is better. `improvement = baseline - candidate`;
positive values favor the candidate. Do not choose a prior because it produces
narrower intervals or stronger acceleration. Favor the common specification
whose scientific conclusions survive reasonable anchor changes, whose chains
mix, and whose predictive calibration is acceptable. Shared shrinkage does
not guarantee that quarter-strength regularization will be retained.

## 5. Confirm only the remaining question

If pooled quarter and pooled half are both plausible but pilot Monte Carlo
error obscures their comparison:

```bash
python -m research.serra.prior_assessment --config research/serra/config/hierarchy/confirm.json --stage all
```

This uses 1,000 warmup and 2,000 retained draws per chain for those two
candidates and the same origins. It is optional. You can select a pilot subset
with `--variants fixed_half pooled_quarter --baseline fixed_half`; the selected
baseline must be included. Changes to data, priors or budgets require a new
run. Interrupted stages keep their partial files and are not silently merged.

## 6. Produce the chosen paper fit

Only after the checks support the pooled-quarter specification:

```bash
python -m research.serra.preflight --config research/serra/config/hierarchy/final.json
python -m research.serra.copula --config research/serra/config/hierarchy/final.json
```

This proposes 2,000 warmup + 4,000 retained draws in four chains, saves the
posterior archive and generates complete manuscript-style reports. Increase
draws if ESS/Monte Carlo precision still requires it. Preflight reports array
storage; worker transfer, merging and diagnostics require additional memory.
`final.json` is a candidate specification, not a declaration that it has passed.

For a matched check of the copula's effect, preserving the learned hierarchy:

```bash
python -m research.serra.copula --config research/serra/config/hierarchy/final.json --independence
```

This sets R=I. The responses still share hyperparameters; it is not six
independent posterior fits. Do this only if the dependence comparison is
needed for the paper, not as a prerequisite for the prior pilot.

If pooling proves hard to identify or compute, the simpler common fixed-half
specification remains available with exactly matched independent/copula priors:

```bash
python -m research.serra.univariate --config research/serra/config/hierarchy/independent.json
python -m research.serra.copula --config research/serra/config/hierarchy/fixed_half.json
```

For exploratory Figures 1 and 2, no MCMC is needed:

```bash
python -m research.serra.explore
```

Rebuild an existing assessment's comparisons without refitting:

```bash
python -m research.serra.prior_assessment --run YOUR_ASSESSMENT_DIRECTORY --stage report
```

The old `config/priors/` and `config/revision/` experiments remain available for
reproduction and supplementary checks. They are not additional steps required
by this focused workflow. The intended story remains evolving temperature
location and environmental risk, with monthly dispersion and joint dependence;
acceleration is a result to assess, not a condition for a publishable story.
