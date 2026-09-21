# What to run next: SERRA revision with BUCEX 1.7.4

For the immediate prior-calibration task, start with [PRIOR_ASSESSMENT.md](PRIOR_ASSESSMENT.md). The broader stages below remain available; they are not prerequisites for that focused comparison. Research configs now use four process workers for four chains.

Start with the six independent models with repeating monthly observation scales, using the same normal innovation priors as your current fits. Then assess the copula with those same marginal specifications. The current constant-scale runs remain the comparison baseline.

The manuscript is a complete working draft, not a claim that the remaining experiments have been completed. Its populated results come only from your six independent constant-scale archives. All 115 references are retained. Boxes R1–R6 and S1–S2 mark the missing evidence.

## 1. Put the configurations into your package

The release already includes `research/serra/config/revision/`. Use those configurations directly; no copying from a manuscript bundle is needed.

Run the commands below from the BUCEX package root, where `pyproject.toml`, `bucex/` and `research/` are located. On Windows, open a terminal in that directory and use the Python environment in which BUCEX is installed. All commands are single-line and work without shell-specific continuation characters.

If that checkout has not been installed in the active environment:

```text
python -m pip install -e .
```

Check that you are importing the intended version and inspect the new specification:

```text
python -c "import bucex; print(bucex.__version__); print(bucex.__file__)"
python -m research.serra.preflight --config research/serra/config/revision/monthly_scale.json
```

Expect version **1.7.4**, 1,616 months, endpoint **2026-08-01**, period 12, normal priors, ASIS off, and `seasonal_scale: true`. The date labels identify monthly blocks: August's label is 1 August, not a claim that daily observations stop on that date.

The shared `revision/protocol.json` retains prior medians 0.01 for level, 0.00005 for slope and 0.02 for location seasonality; the initial-slope SD is 0.0025 per month. Shape is normal with SD 0.3 on [-0.5, 0.5]. Monthly log-scale contrasts have prior SD 0.3. Scale varies by calendar month and repeats across years; it has no secular trend or random walk.

## Generate descriptive Figures 1 and 2

```text
python -m research.serra.explore
```

The configuration is `revision/exploration.json`. Output consists of PNG/PDF
figures, plotted numerical tables and original observations in a timestamped
directory below `results/serra_exploration/`. No posterior is fitted. This
replaces the manuscript-only `rebuild_exploration.py`; the existing `figures`
entry point continues to handle fitted-model reports. See
[EXPLORATION.md](EXPLORATION.md) for calculations, periods and styling.

## 2. Run monthly scales first

A short smoke run verifies your installation and the reporting route:

```text
python -m research.serra.univariate --config research/serra/config/revision/monthly_scale_smoke.json
```

This uses 36 months, one chain and four retained draws. It is only an execution check. It cannot diagnose convergence or supply paper results.

**TNm is the most informative first full fit**, because its winter/summer residual discrepancy is pronounced:

```text
python -m research.serra.univariate --config research/serra/config/revision/monthly_scale.json --series TNm
```

Then run the other five:

```text
python -m research.serra.univariate --config research/serra/config/revision/monthly_scale.json --series TXm TXx TXn TNx TNn
```

Alternatively, omit `--series` to run all six sequentially. Each individual series can also be a separate job. Keep the `.bucex` archives locally so figures and contrasts can be regenerated without refitting; you do not need to upload them all for a first assessment.

Each command creates a fresh dated directory under `results/serra_revision`. It prints that path. Independent runs contain a directory for each series.

### What to inspect

| Output | Main question |
|---|---|
| `config.json`, `run.json`, `data_window.json` | Was the intended model fitted through August 2026? |
| `convergence.json`, `mcmc.csv`, `scientific_targets.csv` | Are initial slopes, physical innovation SDs, scales, shape and reported contrasts sufficiently explored? |
| `*_scale_by_month.csv/png` | Do winter and summer scales differ, and how uncertain are those differences? |
| `*_scale_mcmc.csv`, `*_scale_traces.png` | Are the new monthly scale coefficients mixing? |
| `*_pit_qq_residuals.png`, `*_smoothed_pit.csv` | Did the gross cold-tail and seasonal-dispersion problems diminish? Inspect PIT separately by month, not just pooled. |
| `residual_serial.csv` | Does lag-one dependence remain after improving scale? |
| `*_level.*`, `*_slope_C_per_decade.*`, `period_contrasts.csv` | Does the scale extension change the climatic interpretation? |
| `*_period_risks.csv`, monthly risk plots | How much does it alter the probabilities you intend to report? |

For numerical screening, use the configured R-hat threshold of 1.01 and ESS floor 400, together with traces and Monte Carlo precision for actual scientific targets. A pass is not a fit test. Your current weakest parameters are initial slopes, especially TNx; inspect them explicitly.

Keep the current innovation priors for this comparison. Smoother curves alone are not the objective. Stronger priors should be judged through sensitivity and predictive consequences.

`monthly_scale_final.json` is the same model with four chains, 4,000 warmup and 4,000 retained iterations. Use it for final refits if the shorter runs leave inadequate precision. This is a starting budget, not a convergence guarantee. It refits from scratch; it does not extend a saved chain automatically.

## 3. Check fixed versus evolving location seasonality

You already have evolving-seasonality/constant-scale fits, and step 2 supplies evolving-seasonality/monthly-scale fits. The two missing structural corners can be fitted with:

```text
python -m research.serra.sensitivity --config research/serra/config/revision/structure.json --variants fixed_location_seasonality fixed_location_seasonality_monthly_scale
```

Add `--series TNm` or another name to start with one response. `static` location seasonality still estimates a seasonal cycle; it simply prevents that cycle from evolving over years.

This sensitivity route exports compact prior/posterior and scientific summaries. To generate the full set of standard diagnostics for a saved sensitivity fit, use:

```text
python -m research.serra.report --fit PATH_TO_CASE/fit.bucex --format png
```

These full-record fits answer what changes under each assumption. Held-out comparisons answer whether the extra flexibility helps prediction. They fill manuscript box R3.

## 4. Add the copula with the same monthly-scale margins

First check execution:

```text
python -m research.serra.copula --config research/serra/config/revision/copula_monthly_smoke.json
```

Then inspect memory requirements and run the full model:

```text
python -m research.serra.preflight --config research/serra/config/revision/copula_monthly.json
python -m research.serra.copula --config research/serra/config/revision/copula_monthly.json
```

This estimates a **constant residual correlation matrix**, with each series retaining its own evolving location and repeating monthly scales. It is a full joint fit; it does not attach a correlation estimate to frozen univariate trajectories.

Use the identical marginal specification with R fixed to the identity for a matched joint baseline:

```text
python -m research.serra.copula --config research/serra/config/revision/copula_monthly.json --independence
```

The independent marginal fits remain valid comparison results. The joint R=I route additionally supplies the joint reporting and validation quantities needed for direct comparisons of joint likelihoods and risks.

The base joint state array is about **8.07 GB** at 4 × 2,000 retained draws; inference and reporting need additional memory. The `copula_monthly_final.json` configuration doubles retained draws and that array to about **16.14 GB**. Inspect resources before choosing the larger budget. No full joint fit is launched by this manuscript bundle.

Inspect `copula_correlations.csv`, residual-dependence diagnostics, `period_contrasts.csv`, `compound_heat_forecast.csv`, `ordering_in_sample.csv` and `ordering_forecast.csv`, alongside the marginal and sampler diagnostics. Look for changes in contrasts and their uncertainty, not a universal reduction in every interval.

The `copula.py --structure` argument controls **dependence seasonality**, not monthly marginal scales. For this first comparison leave it at constant. A contemporaneous copula cannot explain residual correlation between different months and does not enforce min/mean/max ordering. Do not repair predictive draws by sorting them.

This stage fills R4. If the joint fit is still difficult, continue the six marginal analyses and limit the paper's empirical joint claims until the joint results are reliable.

## 5. Prior sensitivity: run named subsets

The new sensitivity files retain monthly scales. Do not accidentally switch back to constant-scale margins when changing an innovation prior.

A targeted answer to your smoothing question:

```text
python -m research.serra.sensitivity --config research/serra/config/revision/sensitivity_monthly.json --variants normal_reference level_half level_double
```

This compares level innovation-SD prior medians 0.01, 0.005 and 0.02 while retaining the other priors. Add `--series` to start with a particular response.

A broader innovation and initial-slope check:

```text
python -m research.serra.sensitivity --config research/serra/config/revision/sensitivity_monthly.json --variants innovation_half innovation_double initial_slope_half initial_slope_double
```

In `innovation_half/double`, **all three** innovation-prior scales change together. If a conclusion changes, use a one-component comparison to locate the cause. Initial-slope variants change its SD to 0.00125 or 0.005; they do not change slope innovations.

Shape-prior and support checks, restricted to the GEV responses:

```text
python -m research.serra.sensitivity --config research/serra/config/revision/sensitivity_monthly.json --series TXx TXn TNx TNn --variants xi_uniform xi_sd_0.2 xi_wider_support
```

Monthly-scale prior sensitivity:

```text
python -m research.serra.sensitivity --config research/serra/config/revision/sensitivity_monthly.json --variants monthly_scale_prior_half monthly_scale_prior_double
```

These set log-scale contrast SDs to 0.15 and 0.60. They do not change the location-seasonality innovation prior.

Inspect `prior_posterior.csv`, `prior_posterior_targets.csv`, `period_and_endpoint_targets.csv`, `sensitivity.csv` and `status.csv`. The current independent sensitivity route supplies the prior/posterior export; joint sensitivity uses the full joint reporter, with different output files. Compare the *scientific* conclusions across fits as well as the parameter distributions. Posterior shifts away from the prior and robustness to prior changes are separate reviewer requests.

After joint inference is working, repeat the consequential checks with `sensitivity_copula_monthly.json`, using the same variant names. Copula-specific prior checks are:

```text
python -m research.serra.sensitivity --config research/serra/config/revision/sensitivity_copula_monthly.json --variants normal_reference lkj_2 lkj_4
```

This fills R2. Normal priors remain the principal analysis; another lasso/triple-gamma comparison is optional unless needed for the conclusions.

## 6. Held-out predictive comparisons

Start with a single response to inspect output and computational cost:

```text
python -m research.serra.validate --config research/serra/config/revision/constant_scale.json --series TNm
python -m research.serra.validate --config research/serra/config/revision/monthly_scale.json --series TNm
```

The full protocol uses 35 annual forecast origins, predicting calendar years 1991–2025. Each origin refits the model using only its training prefix. These are expensive analyses, not report-only commands. August 2026 is included in the final descriptive fit; the incomplete 2026 calendar year is not counted as another annual validation fold.

For the four marginal candidates:

```text
python -m research.serra.model_comparison --config research/serra/config/revision/model_comparison.json --stage margins --candidates reference fixed_location_seasonality monthly_scale fixed_location_seasonality_monthly_scale
```

That command is a large grid: four specifications × six responses × 35 origins. First establish satisfactory fitting on individual cases. To narrow this grid, add `--series TNm` (or another subset) to the command.

For the matched copula comparison, both candidates explicitly retain monthly scales:

```text
python -m research.serra.model_comparison --config research/serra/config/revision/model_comparison.json --stage dependence --candidates independence_monthly copula_monthly
```

Use `runs.csv` to identify the two validation directories, then compare paired cases:

```text
python -m research.serra.compare PATH_TO_BASELINE_VALIDATION PATH_TO_CANDIDATE_VALIDATION --output score_comparison.csv
```

The exported scores use the lower-is-better convention, including negative log predictive density. Positive `improvement` means the candidate has the lower score. Do not give a parent containing several candidates to `compare`, because it recursively reads score files and requires unique matched cases. Consider `--block-years 2` as a sensitivity check when yearly score differences remain dependent.

Inspect `scores.csv`, `held_out_pit.csv`, `coverage_by_case.csv`, joint log scores, compound Brier scores and per-fold convergence. Summarize PIT and tails by calendar month, preserving sample sizes. Version 1.7.2 also writes `coverage_by_month.csv` and `held_out_pit_by_month.csv`, with case and distinct-date counts; final comparison panels can be assembled from these and the case-level exports. Extreme quantiles based on 4,000 predictive trajectories have Monte Carlo error; increase predictive draws or use suitable conditional calculations when this affects a conclusion.

If the constant copula still misses seasonal dependence, the same comparison configuration has `harmonic_monthly` as an additional candidate. Assess it against `copula_monthly`; it is a follow-up, not a prerequisite for the first joint fit. This stage fills the predictive portions of R3–R5.

## 7. Endpoint, recovery and forecast checks

The actual July 2019 record case with monthly scales:

```text
python -m research.serra.endpoint --config research/serra/config/revision/endpoint_monthly.json
```

The output distinguishes full-record smoothing from a fit trained before July 2019. The event was selected for its extremeness; its forecast is not an overall calibration test. Version 1.7.2 uses the configured interval level for event-window and pre-event probability summaries and records it in the tables. Older 1.7.1 endpoint exports used 90%; do not relabel those old bounds as 95%.

The matched legacy approximation check must retain constant scales in both engines:

```text
python -m research.serra.endpoint --config research/serra/config/endpoint/laplace_benchmark.json
```

For univariate recovery, first run a smoke configuration, then the existing paper configurations:

```text
python -m research.serra.simulate --config research/serra/config/simulation/smoke.json
python -m research.serra.simulate --config research/serra/config/simulation/paper.json
python -m research.serra.simulate --config research/serra/config/simulation/zero_paper.json
python -m research.serra.simulate --config research/serra/config/simulation/weak_paper.json
python -m research.serra.simulate --config research/serra/config/simulation/endpoint_paper.json
```

The shape grid spans -0.5 to 0.5 with wider fitted support. The main paper grid has 50 replicates per shape; schedule this separately rather than launching every grid together. The endpoint stress grid is conditional and is not ordinary repeated-sampling coverage. Preserve failed and unconverged cases in the experiment accounting.

An included small joint recovery configuration exercises Gaussian, upper-GEV and reflected-GEV channels with repeating monthly scales:

```text
python -m research.serra.joint_recovery --config research/serra/config/revision/joint_recovery_pilot.json --replicate 0
```

It fits R=I and estimated R to common synthetic data. This three-channel, three-replicate pilot is not sufficient for final coverage precision. The original `joint_recovery_full.json` uses six channels but constant scales; do not describe that existing configuration as a monthly-scale recovery experiment. Scale a chosen, checked design to an adequate number of independent replicates for the final report. The current joint recovery export does not automatically produce every desired contrast-coverage panel; plan that summary explicitly.

Re-export forecasts from an already fitted TXx model without refitting:

```text
python -m research.serra.report --fit PATH_TO_TXx_FIT/fit.bucex --horizon 240 --months 1 7 8 --format png
python -m research.serra.forecast_check --config research/serra/config/revision/forecast_TXx_20y.json --fit PATH_TO_TXx_FIT/fit.bucex
```

The saved fit determines the model. Passing a monthly-scale report configuration to a constant-scale fit does not change its likelihood. Use the appropriate saved fit. The second configuration labels its channel as TXx; use an appropriately edited channel configuration for another series.

Check `forecast_uncertainty.csv`, annual forecast summaries and `annual_aggregation.csv`. The forecast-check width plot uses the configured interval level (95% by default); its uncertainty CSV retains 90%, 95% and 99% as well. Use the chosen level consistently in final figures. Complete-year maxima/minima and day-weighted means answer different questions from at least one threshold exceedance during the year.

These checks fill R1 and R5, with detailed results in S1.

## 8. Check the data extension

Retain the current 2026 endpoint, but do not describe mixed-source observations as homogenized. The source audit is included as `preliminary_data/UCCLE_SOURCE_AUDIT.md`.

```text
python -m research.serra.univariate --config research/serra/config/revision/historical_monthly_scale.json
```

This fits through December 2022. For comparisons with the full fit, inspect common dates and common period contrasts, rather than confusing different endpoint years or different recent windows with a source effect. Re-exporting the full saved fit with the historical comparison configuration recalculates the common 1892–1921 versus 1993–2022 contrasts; it does not remove observations from that fit:

```text
python -m research.serra.report --fit PATH_TO_FULL_FIT/fit.bucex --config research/serra/config/revision/historical_monthly_scale.json --format png
```

Authoritative station/source metadata are still needed for the documented overlap and daily-window discrepancies. This completes S2; it cannot be settled by MCMC.

## What to send back first

Send the new **TNm monthly-scale report**, then the other five reports. Include configurations, numerical diagnostics, scales by month, PIT exports, scientific contrasts and selected PNGs. The large `.bucex` archives can stay with you initially. Keep them for subsequent posterior calculations.

After that, send the constant-copula/monthly-scale report and its matched baseline. The most consequential questions are whether monthly calibration improves, whether slopes and risks are stable, and whether the copula changes cross-summary uncertainty and compound predictions reliably.

## Figures and release checks

Use `python -m research.serra.figures --reports PATH_TO_RUN` to rebuild the
manuscript panels from compact report exports. Supply several explicit series
directories when fits came from separate jobs. See `docs/FIGURES.md` for the
figure manifest and re-reporting saved 1.7.1 fits to obtain compact traces.

`validation/RELEASE_VALIDATION.md` records the tests and smoke runs actually
executed for the current release. No release check is a substitute for converged full-record
fits, recovery experiments or held-out validation. Keep the manuscript result
boxes until their stated scientific evidence is available.
