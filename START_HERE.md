# Start with BUCEX 1.8.1

This is the complete command guide for the SERRA revision experiments. Run from
the extracted `bucex-1.8.1` directory, beside `pyproject.toml`.

The current candidate model has six private FS location models and **three
shared shrinkage hyperparameters: level, slope and seasonal innovations**.
Every response retains its own three innovation SDs, initial seasonal pattern,
latent paths, repeating monthly observation scales and, for GEV responses,
constant-in-time shape. A Gaussian copula accounts for contemporaneous residual
dependence. No SSVS or ASIS is used in these configurations.

The primary data span is January 1892–August 2026: 1,616 monthly blocks.
All scientific settings are in JSON. The package handles model construction,
inference, diagnostics, forecasts, risks, comparison tables and figure style.

## 0. Install and verify the version

```bash
python -m pip install -e ".[plot]"
python -c "import bucex; print(bucex.__version__, bucex.__file__)"
```

Expect `1.8.1` and the path of this checkout. On biobot, four chains use four
local processes; no Slurm installation is needed. Numerical thread pools are
limited within each chain. Candidates and forecast origins run sequentially.
Avoid launching multiple large final fits simultaneously. Preflight reports
state-array storage; workers, merging and diagnostics need additional RAM.

Start new assessments for the three-hyperparameter model. Saved 1.8.0 fits
remain readable but their declared two-component hierarchy is not changed by
installing 1.8.1. New configurations write under `results/serra_181_*`.

## What to run, in order

| Stage | Purpose | Full-record fits / historical refits if both stages run |
|---|---|---:|
| Exploration | Observed-data Figures 1 and 2 | 0 / 0 |
| Smoke | Execution check on a short record | 4 / 4 short fits |
| Prior pilot | Fixed half/quarter versus three-component pooled half/quarter | 4 / 16 |
| Seasonal hierarchy check | Two versus three pooled components; seasonal-anchor sensitivity | 4 / 16 |
| Dependence check | R=I versus estimated Gaussian copula | 2 / 8 |
| Structural adequacy | Monthly versus constant scales; fixed versus evolving location seasonality | 3 / 12 |
| Targeted prior checks | Initial slope, shape, monthly dispersion, copula prior and hyperprior width | Select the subsets below |
| Confirmation, if needed | Resolve Monte Carlo uncertainty between the pooled candidates | 2 / 8 |
| Final fit and figures | Archive the selected posterior and produce manuscript results | 1 / 0 |

These are focused comparisons, not a factorial search over every possible
structure. Start with posterior sensitivity. Run the historical stage for the
main prior comparison and dependence/adequacy questions; extend the remaining
checks to forecasts when their posterior differences matter. Do not interpret
an unstable short-chain estimate as a model difference. No simulation study is
automatically run. Previously completed supplementary computation studies can
still be used if their assumptions match the revised model.

The optional `--stage all` commands below execute both stages. Use either the
combined command or the split-stage route for a given assessment, not both.

## 1. Generate exploratory Figures 1 and 2

```bash
python -m research.serra.explore
```

No MCMC is needed. The script uses the general `explore_monthly` API and writes
source/summary CSVs and manuscript-style PNG/PDF figures below
`results/serra_exploration/`. The current record ends in August 2026. Settings
are in `research/serra/config/revision/exploration.json`.

## 2. Check the complete workflow once

```bash
python -m research.serra.prior_assessment --config research/serra/config/hierarchy/smoke.json --stage all
```

This uses all six responses from January 2023 to August 2026, four workers,
three warmup iterations and four retained draws per chain. It checks the four
candidates, three-component shared-scale reports, a held-out year and comparison
figures. **The smoke outputs are execution checks only.** `needs_review` is
expected in convergence files. Do not choose hyperparameters from this run.

For a short, readable example of the general model-building API:

```bash
python -m docs.examples.shared_shrinkage
```

That example also uses tiny chains and is not a scientific experiment.

## 3. Compare fixed and learned innovation shrinkage

Inspect the resolved dates, complete prior declarations, model settings and fit
counts before running MCMC:

```bash
python -m research.serra.prior_assessment --config research/serra/config/hierarchy/pilot.json --stage plan
```

| Candidate | Level median/anchor | Slope median/anchor | Seasonal median/anchor | Learned shared components |
|---|---:|---:|---:|---|
| `fixed_half` | .005 | .000025 | .020 | None |
| `fixed_quarter` | .0025 | .0000125 | .020 | None |
| `pooled_quarter` | .0025 | .0000125 | .020 | Level, slope, seasonal |
| `pooled_half` | .005 | .000025 | .020 | Level, slope, seasonal |

The fixed rows specify medians of physical innovation SDs. Pooled rows specify
anchors of lognormal hyperpriors on the shared conditional prior medians;
`log_sd = log(2)`. About 95% of each shared median's hyperprior lies between
0.257 and 3.89 times its anchor. The marginal innovation prior integrates over
this uncertainty. It is not a normal prior with its scale fixed at the anchor.

Initial-slope prior SD is .0025 per month. Initial seasonal coefficient SD is
2.25. Monthly log-scale contrast prior SD is .3. GEV shape has a Normal(0,.3²)
prior truncated to [-.5,.5]. These nuisance priors are matched across candidates.
Small seasonal innovations mean a slowly changing seasonal pattern; they do
not remove its initial amplitude or the repeating monthly observation scales.

Run the posterior comparison first:

```bash
python -m research.serra.prior_assessment --config research/serra/config/hierarchy/pilot.json --stage sensitivity
```

The pilot uses four chains, four workers and **500 warmup + 500 retained draws
per chain**. Keep the printed `Assessment directory` and inspect its
`comparison/` reports. If the chains have substantial disagreement or little
movement, resolve that before interpreting predictive differences.

Continue with the same saved settings and historical forecasts:

```bash
python -m research.serra.prior_assessment --run "PILOT_ASSESSMENT_DIRECTORY" --stage predictive
```

Replace `PILOT_ASSESSMENT_DIRECTORY` with the actual printed path. To run both
stages from a fresh directory in one command instead:

```bash
python -m research.serra.prior_assessment --config research/serra/config/hierarchy/pilot.json --stage all
```

The four training cutoffs are December 2000, 2010, 2015 and 2020. They predict
2001–2005, 2011–2015, **2016–2020 including the 2019 record**, and 2021–2025.
Each fit learns all three shared hyperparameters from its own training data.
These are five-year forecasts from each origin, not rolling one-step forecasts.
Comparisons across four origins are descriptive; no precise ranking interval
is obtained by treating all months as independent replicates.

## 4. Check the seasonal hierarchy and its anchor

```bash
python -m research.serra.prior_assessment --config research/serra/config/hierarchy/seasonality.json --stage plan
python -m research.serra.prior_assessment --config research/serra/config/hierarchy/seasonality.json --stage sensitivity
```

The four candidates keep quarter level/slope anchors and the same likelihood:

| Candidate | Seasonal treatment |
|---|---|
| `pooled_quarter` | Learned common seasonal median, anchor .020 |
| `pooled_level_slope` | Original two-component hierarchy; fixed seasonal prior median .020 |
| `season_anchor_half` | Three-component hierarchy; seasonal anchor .010 |
| `season_anchor_double` | Three-component hierarchy; seasonal anchor .040 |

This separates seasonal pooling from the choice of its anchor. It does not
change initial seasonality or monthly observation dispersion. Start with
`shared_shrinkage.csv`/`.png`, then individual seasonal innovation comparisons,
monthly location changes, level/slope contrasts and risk. The unpooled seasonal
hyperparameter is marked `not pooled` in comparison figures.

If the differences affect the scientific conclusions or calibration, continue:

```bash
python -m research.serra.prior_assessment --run "SEASONAL_ASSESSMENT_DIRECTORY" --stage predictive
```

Or, from a fresh run, execute both stages directly:

```bash
python -m research.serra.prior_assessment --config research/serra/config/hierarchy/seasonality.json --stage all
```

Do not interpret convergence problems as evidence against seasonal pooling.
Check the seasonal hyperparameter and six individual seasonal innovation traces.

## 5. Assess what the copula contributes

```bash
python -m research.serra.prior_assessment --config research/serra/config/hierarchy/dependence.json --stage plan
python -m research.serra.prior_assessment --config research/serra/config/hierarchy/dependence.json --stage all
```

This compares fixed R=I with an estimated constant Gaussian copula. Both models
learn the same three shared shrinkage hyperparameters. Thus the R=I model
still has hierarchical dependence through its priors; it is not six separate
posterior fits. Compare trajectories, paired cross-summary contrasts, compound
risks and held-out joint log scores. Read residual dependence and ordering
checks as well as the marginal scores.

Season/month-dependent copulas remain available through `copula --structure`,
but are not part of the minimal story. If constant dependence leaves a clear
seasonal residual failure, assess that extension as a separately declared check.

## 6. Check the two main structural assumptions for the supplement

```bash
python -m research.serra.prior_assessment --config research/serra/config/hierarchy/adequacy.json --stage plan
python -m research.serra.prior_assessment --config research/serra/config/hierarchy/adequacy.json --stage all
```

This compares the reference with one change at a time:

- A constant observation scale for each response, replacing its monthly scales.
- Fixed location seasonality, retaining each response's initial seasonal cycle.

The fixed-seasonality candidate explicitly omits the seasonal innovation
hyperparameter because that innovation is fixed at zero. It retains the
level/slope hierarchy and monthly observation scales. Check month-specific
PIT/QQ, predictive coverage, residual serial correlation and scientific targets.
These comparisons support the adequacy of a chosen structural framework; they
are not a search over every combination of location and scale evolution.

## 7. Run targeted prior checks for the revision

The following subsets use `reviewer_sensitivity.json` and compare one prior
choice at a time with `reference`. Each command creates a separate assessment
and writes its resolved declarations. Begin with posterior sensitivity; extend
only meaningful or ambiguous differences to forecasts. All use the pooled
quarter reference with **three** shared hyperparameters.

Initial slope, relevant to slow mixing and the slope decomposition:

```bash
python -m research.serra.prior_assessment --config research/serra/config/hierarchy/reviewer_sensitivity.json --variants reference initial_slope_half initial_slope_double --stage sensitivity
```

GEV shape, including a uniform prior across [-.5,.5] and narrower/wider normal
priors on that same support:

```bash
python -m research.serra.prior_assessment --config research/serra/config/hierarchy/reviewer_sensitivity.json --variants reference xi_normal_tighter xi_normal_wider xi_uniform --stage sensitivity
```

Monthly observation-scale contrast prior:

```bash
python -m research.serra.prior_assessment --config research/serra/config/hierarchy/reviewer_sensitivity.json --variants reference monthly_scale_prior_half monthly_scale_prior_double --stage sensitivity
```

Copula LKJ concentration (eta = 1, 2, 4):

```bash
python -m research.serra.prior_assessment --config research/serra/config/hierarchy/reviewer_sensitivity.json --variants reference lkj_2 lkj_4 --stage sensitivity
```

Shared-hyperprior width, if common medians or their scientific consequences
remain sensitive: log-SD log(1.5), log(2) and log(3), with anchors held fixed:

```bash
python -m research.serra.prior_assessment --config research/serra/config/hierarchy/reviewer_sensitivity.json --variants reference hyperprior_tighter hyperprior_wider --stage sensitivity
```

For any of those saved assessments, the forecast continuation is:

```bash
python -m research.serra.prior_assessment --run "THAT_ASSESSMENT_DIRECTORY" --stage predictive
```

These checks surround the declared pooled-quarter reference. If another common
specification is adopted, copy its resolved `priors` settings from its saved
`config.json` into the relevant supplementary/final configuration before new
runs, then inspect `--stage plan`. Do not change the settings inside an existing
assessment directory. All series must use the same declared common specification.

## 8. Read the evidence before choosing the final specification

Each assessment's `comparison/` directory contains:

| Output | What it tells you |
|---|---|
| `convergence.csv`, `joint_mcmc.csv` | R-hat/ESS warnings for initial slopes, innovation SDs, monthly scales, shared medians and copula parameters |
| `shared_shrinkage.csv`, `shared_shrinkage_updates.csv`, `shared_shrinkage.png` | Hyperprior/posterior learning and sensitivity for level, slope and seasonal regularization |
| `prior_updates.csv`, `*_prior_posterior.png` | Individual innovation SDs versus unconditional priors that integrate the shared hyperparameters |
| `scientific_targets.csv`, `joint_scientific_targets.csv` | Period warming/rate changes and paired cross-summary differences |
| `*_level.png`, `*_slope.png`, `*_risk.png` | How smoothing affects scientific trajectories and risks |
| `predictive_comparison.csv`, `scores_by_origin.csv` | Matched marginal forecast losses; positive improvement favors the candidate |
| `joint_log_scores_comparison.csv`, `compound_heat_scores_comparison.csv` | Joint forecast and simultaneous hot-condition performance |
| `coverage_by_month.csv`, `pit_by_month.csv`, `*_forecasts.png` | Seasonal predictive calibration and the actual held-out observations |
| `event_counts.csv` | How many observed threshold events inform each marginal risk score |
| `shared_shrinkage_forecasts.csv` | Training-only common-median posteriors at each forecast origin |

Under `sensitivity/VARIANT/joint/` also inspect `shared_shrinkage_traces.png`,
`shared_shrinkage_traces.csv.gz`, channel parameter/scale traces, PIT/QQ figures,
`residual_serial.csv`, `residual_dependence.csv` and `ordering_*.csv`.
A contemporaneous copula does not remove serial residual memory or guarantee
minimum/mean/maximum ordering. Do not sort predictive samples to hide violations.

Choose a defensible common specification based on adequate computation, stable
scientific conclusions and acceptable predictive calibration. Posterior movement
away from its prior or narrower intervals is not a target to maximize. Strong
shrinkage is useful if those checks support it. If acceleration is sensitive,
report that while retaining robust warming and risk findings.

Pilot files do not include large posterior archives. They do include compact
CSV/JSON files, PNGs and compressed traces. Keep these together when sharing
results; the baseline declarations and forecast case dates are essential.

## 9. Increase MCMC only where the comparison remains uncertain

To confirm the two pooled level/slope anchors with seasonal pooling in both:

```bash
python -m research.serra.prior_assessment --config research/serra/config/hierarchy/confirm.json --stage all
```

This uses 1,000 warmup + 2,000 retained draws per chain. For another targeted
comparison, copy its JSON, increase its `mcmc` budget and give it a new output
root, then start a new assessment. Short-chain status is never silently waived.
Completed stages are reused by `--run`; interrupted stages retain partial files
and require a fresh run to avoid mixing incomplete outputs.

## 10. Fit the selected model and save the full posterior

If the checks support the pooled-quarter, three-component specification:

```bash
python -m research.serra.preflight --config research/serra/config/hierarchy/final.json
python -m research.serra.copula --config research/serra/config/hierarchy/final.json
```

If the pooled-half specification is chosen instead:

```bash
python -m research.serra.preflight --config research/serra/config/hierarchy/final_half.json
python -m research.serra.copula --config research/serra/config/hierarchy/final_half.json
```

Run the selected one, not both by default. These configurations request four
chains, four workers, 2,000 warmup + 4,000 retained draws per chain, save the
`.bucex` archive and produce full manuscript-style reports. The default ten-year
forecast includes month-specific, annual and seasonal summaries/risk curves.
Longer MCMC is still needed if effective sample sizes or Monte Carlo precision
are inadequate. A `final.json` filename does not certify publication readiness.

For a longer matched R=I fit, only if the pilot dependence comparison needs it:

```bash
python -m research.serra.copula --config research/serra/config/hierarchy/final.json --independence
```

Use `final_half.json` there if that is the chosen prior specification.

## 11. Generate manuscript panels and re-report without fitting

The final fit already creates component, diagnostic, forecast and risk figures.
To assemble the manuscript panel recipes from its compact CSV report directory:

```bash
python -m research.serra.figures --reports "FINAL_REPORT_DIRECTORY" --formats png
```

Replace `FINAL_REPORT_DIRECTORY` with the directory printed by the final fit.
For selected recipes only:

```bash
python -m research.serra.figures --reports "FINAL_REPORT_DIRECTORY" --panels monthly_scales annual_forecasts --formats png
```

Rebuild an assessment's comparison tables and figures from existing outputs:

```bash
python -m research.serra.prior_assessment --run "ANY_ASSESSMENT_DIRECTORY" --stage report
```

Exploratory Figures 1 and 2 come from step 1; `figures` handles fitted-model
panels. Figures use the package's manuscript style. Choose PDF/SVG through
`--formats` if needed.

## 12. Fixed-prior fallback: six univariate analyses, then the copula

If the hierarchy proves difficult to identify or compute, use the matched
fixed-half normal priors with monthly observation scales:

```bash
python -m research.serra.univariate --config research/serra/config/hierarchy/independent.json
python -m research.serra.copula --config research/serra/config/hierarchy/fixed_half.json
```

To inspect one independent response first:

```bash
python -m research.serra.univariate --config research/serra/config/hierarchy/independent.json --series TXm
```

The same `--series` option accepts TNm, TXx, TXn, TNx and TNn. The univariate API
is unchanged. These fits have no cross-series shrinkage hyperparameters.

The intended paper story remains tracking the evolving temperature distribution
and environmental risk under slow climatic change and substantial weather
variability. Three-component hierarchical shrinkage supplies a common form of
regularization; the copula carries cross-summary dependence into inference and
risk. Neither makes a particular acceleration result a prerequisite for the
paper. Keep the focused adequacy/prior comparisons in the supplement where they
support that story.
