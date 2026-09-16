# SERRA workflow — BUCEX 1.7.0

Run commands from the extracted release directory. Install with
`python -m pip install -e ".[test]"`. Start with [../../START_HERE.md](../../START_HERE.md).

## The primary analysis

All six summaries are **monthly**: TXm/TNm Gaussian means; TXx/TNx GEV
maxima; TXn/TNn reflected GEV minima. The primary record is **January 1892 to
August 2026: 1616 monthly blocks**. The later data have mixed sources; see
[../../bucex/data/SOURCES.md](../../bucex/data/SOURCES.md). The named
`*_1892_2022.json` configurations retain the earlier record for comparison.
The `*_extended.json` files remain compatibility aliases for the full record.

Each channel has a dynamic local level, slope and dummy location seasonality.
Observation scale and GEV shape are unknown but constant. Reference priors are
continuous FS lasso; no SSVS, PGAS or dynamic-factor analysis is required.
The optional constant Gaussian copula is fitted jointly. ASIS is off.

`config/base.json` is the resolved scientific protocol. Its production starting
budget is four independently seeded chains, 2000 warm-up and 2000 retained
iterations each. This is not an empirical convergence guarantee. Every run
writes expanded settings and actual dates into a fresh directory.

```bash
python -m research.serra.preflight --config research/serra/config/copula_full.json
python -m research.serra.univariate --series TXx
python -m research.serra.copula
```

The last two commands default to **smoke runs**: 36 months, four warm-up and
four retained draws, one chain. They verify execution only.

## Primary fits and the independent fallback

```bash
python -m research.serra.univariate --config research/serra/config/independent_full.json --series TXx
python -m research.serra.univariate --config research/serra/config/independent_full.json
python -m research.serra.copula --config research/serra/config/copula_full.json --independence
python -m research.serra.copula --config research/serra/config/copula_full.json
```

Run each univariate series as a separate job with `--series`. The R=I joint
baseline uses identical marginal declarations. The estimated copula changes
the **joint posterior**, so trajectories, slopes and interval widths can change.
Raw correlations alone are not evidence of residual dependence; inspect
conditional score dependence after location, scale and seasonality.

If the joint fit remains impractical or mixes poorly, the six independent
analyses still run. Remove unsupported joint empirical claims; do not report
contrasts formed by arbitrarily pairing unrelated marginal MCMC indices as if
they came from the joint copula posterior.

The full joint centered state array is about **8.07 GB**, before forecasts,
diagnostics and temporary copies. `preflight` prints the budget; choose a
suitable machine or run the independent series separately.

## What the outputs mean

| Output | Inspect for |
|---|---|
| `data_window.json`, `config.json`, `run.json`, `declared_priors.json` | Actual endpoint, declaration, priors, engine and retained budget |
| `convergence.json`, `mcmc.csv` | R-hat, bulk/tail ESS, independent chain count, undefined diagnostics and constant draws |
| `scientific_targets.csv`, `period_contrasts.csv` | Climate-period changes, posterior sign probabilities and paired differences; chain dimensions retained |
| `contrast_definitions.json` | Exact dates, months, channel pairs and rate units |
| `sampler_metrics.csv`, `engine.json` | Laplace–MH acceptance, slice cost, GEV support failures, state-update diagnostics |
| `*_level.*`, `*_slope_C_per_decade.*`, `*_seasonal.*`, `*_location.*` | Distinct latent components with pointwise credible bands |
| `*_observation_scale.*`, `*_innovation_effects.csv` | Scale assumption and practical 120-month contribution of each innovation |
| `*_period_risks.csv`, `*_risk.*` | Original-tail risk, by calendar month and period |
| `*_traces*.png`, `*_pit_qq_residuals.png` | Chain movement and descriptive in-sample fit |
| `residual_serial.csv`, `residual_dependence*.csv` | Remaining serial, cross-series and seasonal dependence |
| `copula_correlations.csv`, `compound_heat_forecast.csv` | Posterior dependence and conditional compound probability |
| `ordering_in_sample.csv`, `ordering_forecast.csv` | Min/mean/max crossings; observations are never sorted or discarded |

`convergence.json` screens physical structural SDs, initial coefficients,
scale, shape, copula parameters and scientific contrasts. All shrinkage
parameters remain in `mcmc.csv`, including declared fixed hyperparameters.
A numerical pass is not proof of model adequacy: assess traces, precision of
risk probabilities, sensitivity and held-out calibration as well.

The main comparison is **January 1892–December 1921** versus
**September 1996–August 2026**, two complete 360-month windows. Report slopes
as period averages in °C/decade, and compare each calendar month's full
location. The historical configurations instead compare 1892–1921 with
1993–2022. Endpoint changes/final slopes are also retained as secondary
summaries. Equal weighting of monthly latent states is distinct from
**day-weighted annual observed means**.

Joint channel contrasts subtract quantities within the same draw. With fixed
scale/shape, within-margin quantile changes equal location changes; six fitted
summary distributions do not reconstruct the full daily temperature density.
A small continuous innovation SD is not an inclusion probability or a model
selection result. See [../../docs/FORECASTS.md](../../docs/FORECASTS.md) for
month-specific, seasonal and annual forecasts and risk figures.

## Core and targeted prior sensitivity

```bash
python -m research.serra.sensitivity --config research/serra/config/sensitivity/paper.json
python -m research.serra.sensitivity --config research/serra/config/sensitivity/joint.json
python -m research.serra.sensitivity --config research/serra/config/sensitivity/targeted.json
python -m research.serra.sensitivity --config research/serra/config/sensitivity/joint_targeted.json --variants lkj_2 lkj_4
```

The five core variants are lasso reference, **all** innovation medians halved,
all doubled, matched normal, matched triple-gamma. Location process SD prior
medians are (.02, .00005, .02). Initial level is N(0,20²), initial slope is
N(0,.0025²), initial seasonal coordinates are N(0,2.25²); sigma² is IG(2,2).
Shape is N(0,.3²) on [-.5,.5]. Lasso lambda²=1; TG a=c=.5 and global multiplier
1. Local mixing variables are inferred; the prior center is not estimated
from the data.

Targeted variants alter initial priors, observation variance, shape SD/support
and a uniform shape alternative. Joint targeted variants include LKJ(2/4)
against LKJ(1). Run cases individually with `--variants NAME`.
`sensitivity/joint_smoke.json` checks this joint-refit route. `status.csv`
distinguishes execution completion from numerical screening. Compare the
actual scientific contrasts/risks, not just prior shapes.

## Limited structural checks for the supplement

```bash
python -m research.serra.sensitivity --config research/serra/config/sensitivity/structure.json
python -m research.serra.sensitivity --config research/serra/config/sensitivity/joint_structure.json
python -m research.serra.model_comparison --config research/serra/config/model_comparison_full.json --stage margins
```

The four structural variants cross fixed/evolving **location seasonality**
with constant/monthly **observation scale**, retaining monthly observations.
A deterministic linear-location benchmark is additionally available in rolling
validation. This is a small declared comparison, not a search over every
possible source of nonstationarity. Full structural scale is demonstrated in
`docs/examples/parameter_evolution.py`, outside the paper workflow.

## Predictive validation and dependence comparison

```bash
python -m research.serra.validate --config research/serra/config/independent_full.json
python -m research.serra.validate --config research/serra/config/independence_full.json
python -m research.serra.validate --config research/serra/config/copula_full.json
python -m research.serra.model_comparison --config research/serra/config/model_comparison_full.json --stage dependence
python -m research.serra.compare PATH_TO_BASELINE_VALIDATION PATH_TO_CANDIDATE_VALIDATION --output comparison.csv
```

Use the joint R=I validation for a **joint log-score** comparison with the
copula; independent per-series outputs alone do not contain a joint log score.
Validation uses matched held-out months, 90/95/99% central coverage, direct
quantiles .005–.995, PIT and proper scores, including compound heat Brier
scores. Prior declarations are unchanged across origins. Outputs checkpoint
each completed fold, including convergence diagnostics.

With a 12-month forecast horizon and annual origins, the final complete test
year is 2025. The January–August 2026 observations are in the full model fit
but are not a complete annual validation block. A supplementary shorter-horizon
validation can include them; never label an incomplete year as an annual test.

Paired score differences use calendar-year block uncertainty; consider longer
blocks for persistent errors/overlapping origins. The optional harmonic copula
configuration is supplementary. It shrinks seasonal deviations and has an
order-dependent partial-correlation prior: test channel order/prior width
before claiming seasonal dependence. The Gaussian copula is not a model of
asymptotic tail dependence or residual serial dependence.

## Reviewer simulations, endpoint and aggregation

```bash
python -m research.serra.simulate --config research/serra/config/simulation/paper.json
python -m research.serra.simulate --config research/serra/config/simulation/zero_paper.json
python -m research.serra.simulate --config research/serra/config/simulation/weak_paper.json
python -m research.serra.simulate --config research/serra/config/simulation/endpoint_paper.json
python -m research.serra.joint_recovery --config research/serra/config/joint_recovery_full.json
python -m research.serra.endpoint --config research/serra/config/endpoint/paper.json
python -m research.serra.forecast_check --config research/serra/config/forecast/paper.json
```

Start with corresponding smoke configurations before expensive grids. Recovery
uses generating xi=-.5,-.4,...,.5 and wider fitted support [-.75,.75]. The endpoint
stress design is conditional, not ordinary frequentist coverage. Joint recovery
uses **constant** scale for both truth and fitting, with R=I versus estimated R.
`--replicate` runs one selected joint-recovery replicate; numerical tables
checkpoint each completed case. Report failures and between-replicate uncertainty.

`simulation/laplace_benchmark.json` and `endpoint/laplace_benchmark.json`
compare legacy approximate Laplace with corrected Laplace–MH under exactly
matched constant scales. The primary private kernel always uses MH correction.
Endpoint output distinguishes full-record smoothing from a fit ending before
the July 2019 event. A finite endpoint is a statistical support boundary, not
a physical limit.

Annual extrema use complete years and a product **within each posterior draw**
before posterior averaging. This assumes conditional residual independence
across months. Distinguish monthly-block return levels from annual return
levels, posterior predictive means from probability credible intervals, and
pointwise bands from simultaneous bands. Ordering is checked, not enforced.

## Reuse fitted posteriors and rebuild data

```bash
python -m research.serra.report --fit PATH_TO_RUN/fit.bucex --months 1 7 8 --horizon 120
python -m research.serra.check_updates --fit PATH_TO_RUN/fit.bucex
python -m research.serra.prepare_uccle
```

`report` never refits or adds observations. For a legacy saved configuration,
pass an updated `--config` to request the new period contrasts only when the
saved fit contains the complete requested periods. A genuinely different
scale model or prior needs a new fit. Data preparation is optional: all six
validated monthly files are bundled.

Read `models.py` for model construction, `run.py` for fitting, `report.py` for
public posterior APIs, `sensitivity.py` for matched refits, and `validate.py`
for forecasts. The helper modules contain research orchestration; inference,
period estimands, diagnostics, risks, forecasts and figure APIs live in BUCEX.
