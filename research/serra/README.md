# SERRA workflow for BUCEX 1.6.5

Run from the extracted release directory, after:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

The six trajectories are TXm/TNm (Gaussian monthly means), TXx/TNx (GEV
maxima), and TXn/TNn (reflected GEV minima). Full fits now use **1892–August 2026**
(`data.end: null`, the latest bundled month). The `independent_1892_2022.json`
and `copula_1892_2022.json` configurations retain the original manuscript window.
The extended record has mixed sources; see `bucex/data/SOURCES.md`.

`config/base.json` contains the scientific assumptions and a four-chain,
2000-warmup/2000-retained budget. `config/smoke.json` inherits it but uses 36
months and four draws: it checks execution only. Run directories are unique,
so a new job does not overwrite an earlier result. Every run saves expanded
settings, model declarations and declared priors.

## 1. Run and inspect one margin, then all six

```bash
python -m research.serra.univariate --series TXm
python -m research.serra.univariate --series TXx TNn
python -m research.serra.univariate --series TXx --prior normal --scale linear
python -m research.serra.univariate --config research/serra/config/independent_full.json
```

Each series is fitted independently and can be run as a separate job. The same
`Model → fs_priors → fit → diagnostics/risk/forecast` API is used for every series.
Read `models.py` first: it is the complete scientific model declaration. Uccle
metadata supplies only the observation family and the extrema orientation.

Look at these files before interpreting a figure:

| Output | What to assess |
|---|---|
| `run.json`, `config.json`, `declared_priors.json` | Correct dates, family, model, priors, engine and saved settings |
| `mcmc.csv` | Multiple chains, R-hat near 1 (investigate >1.01), bulk/tail ESS, constants and weakly identified parameters |
| `scientific_targets.csv` | Mixing and uncertainty of level changes and paired changes, retaining chain membership |
| `sampler_metrics.csv`, `engine.json` | Channel-specific path acceptance, coefficient slice cost, support failures and ASIS invariance errors |
| `*_innovation_effects.csv` | SD of each component's contribution over 120 months; probability this SD exceeds 0.1°C |
| `*_level.csv`, `*_seasonal.csv`, `*_observation_scale.csv` | Separate location evolution, location seasonality and observational variability |
| `*_risk.csv`, `*_return_level_100_blocks.csv` | Original-tail probabilities; **100 monthly blocks**, not 100 years |
| `residual_serial.csv`, `in_sample_pit.csv` | Descriptive lack-of-fit signals; in-sample smoothing is not held-out calibration |

The innovation-effect probability is a practical magnitude summary, not a
posterior inclusion probability. All main priors are continuous. Exact SSVS
remains in the package but is absent from the reference research protocol.
PNG traces of physical SDs, scale, shape and scientific contrasts are exported
automatically; sign-switching alone is not mixing evidence. Slopes are in °C per
decade. `*_pit_qq_residuals.png` is an in-sample smoothed check. The calendar
forecast/risk and scale files are explained in [../../docs/FORECASTS.md](../../docs/FORECASTS.md).

Re-export a saved fit without rerunning MCMC:

```bash
python -m research.serra.report --fit PATH_TO_RUN/fit.bucex
python -m research.serra.report --fit PATH_TO_RUN/fit.bucex --months 1 7 8 --horizon 120
python -m research.serra.check_updates --fit PATH_TO_RUN/fit.bucex
```

`report` defaults to the saved `config.json` and PNGs. It prints the actual
training window and never extends a saved posterior. `check_updates` compares
fixed-origin forecasts with later bundled observations; it does not refit.
To assimilate the new observations, run the full univariate or copula fit.

## 2. Check prior sensitivity with matched models

```bash
python -m research.serra.sensitivity --config research/serra/config/sensitivity/smoke.json --series TXm TXx
python -m research.serra.sensitivity --config research/serra/config/sensitivity/paper.json
```

Normal/lasso/TG share prior median innovation SDs and the same observation-scale
model. The full grid also changes each innovation median, initial slope and
seasonality, scale seasonality prior width, observation variance, ASIS setting,
shape-prior family and shape support. Review `prior_posterior.csv`,
`prior_posterior_targets.csv`, `scientific_targets.csv` and `sensitivity.csv`.
Sensitivity in an unconverged fit is not reliable prior sensitivity.

Reference: lasso lambda²=1; TG a=c=0.5 and global multiplier=1; all local
mixing variables sampled. TG a=0.1 is an additional spikier case. Initial
hyperparameters are declared without using the data to center them. The normal
shape prior has SD .30 on [-.5,.5]; the uniform alternative uses identical
support. Wider support and SD .20 are separate comparisons.

## 3. Compare a small set of model structures predictively

```bash
python -m research.serra.model_comparison --candidates reference static_season constant_scale
python -m research.serra.model_comparison --config research/serra/config/model_comparison_full.json --stage margins
```

Candidates include static/dynamic location seasonality, fixed/absent slope,
constant/seasonal observational scale, and seasonal-plus-linear/RW log scale.
Use the same held-out calendar months. The scripts do not choose a winner from
in-sample likelihoods and do not average unweighted models. Record a justified
marginal specification before the next stage; update the common config so
independence and dependence fits use **identical** marginal specifications.

`runs.csv` locates every candidate's validation output. Compare two directories:

```bash
python -m research.serra.compare PATH_TO_BASELINE_VALIDATION PATH_TO_CANDIDATE_VALIDATION --output comparison.csv
```

Lower proper scores are better. Comparison uncertainty resamples paired calendar
year blocks; consider longer blocks if errors persist across years. Smokes have
only one held-out year and cannot support score-comparison uncertainty.

## 4. Add contemporaneous residual dependence

```bash
python -m research.serra.copula --independence
python -m research.serra.copula
python -m research.serra.copula --structure harmonic
python -m research.serra.copula --config research/serra/config/copula_full.json --independence
python -m research.serra.copula --config research/serra/config/copula_full.json
```

These are private trajectories under R=I versus estimated R. Every parameter
and state update includes the copula; the univariate fits are not frozen inputs.
The independent six-series analysis remains available if joint inference is
impractical. Do not include unsupported joint conclusions in that fallback.

Inspect `copula_correlations.csv`, `residual_dependence.csv`,
`residual_dependence_by_month.csv`, paired scientific contrasts and how the
marginal paths/risks change. Positive raw temperature correlations can reflect
shared seasonality or warming; evidence here concerns **conditional residual
scores**. R-hat/ESS for R and the trajectories must both be satisfactory.

```bash
python -m research.serra.model_comparison --stage dependence
python -m research.serra.model_comparison --config research/serra/config/model_comparison_full.json --stage dependence
python -m research.serra.copula --config research/serra/config/copula_seasonal_full.json
```

The default seasonal alternative has one harmonic pair per Fisher partial
correlation, shrunk towards a common baseline. This avoids 12 unrelated
correlation matrices. `SeasonalGaussianCopula` also supports four-season or
monthly zero-sum contrasts. For seasonal models, check a different channel
ordering and contrast prior widths (.125/.25/.5); this prior is order dependent.
The dependence comparison includes LKJ eta=1,2,4 for the constant model.

## 5. Study recovery, shape and the July 2019 endpoint

```bash
python -m research.serra.simulate --config research/serra/config/simulation/smoke.json
python -m research.serra.simulate --config research/serra/config/simulation/paper.json
python -m research.serra.simulate --config research/serra/config/simulation/zero_paper.json
python -m research.serra.simulate --config research/serra/config/simulation/weak_paper.json
python -m research.serra.simulate --config research/serra/config/simulation/endpoint_paper.json
python -m research.serra.joint_recovery --config research/serra/config/joint_recovery_full.json
python -m research.serra.endpoint --config research/serra/config/endpoint/paper.json
```

The generating shape grid is -.5,-.4,…,.5; fitted support is [-.75,.75].
`endpoint_paper.json` is a **conditional stress test**, not ordinary coverage.
Shape recovery uses independent replicates and common random numbers across
shapes within a replicate. Report uncertainty across replicates and all failures;
time points within one simulated trajectory are not independent replications.
Joint recovery compares R=I versus estimated R with mixed margins.

Paired approximate/exact historical constant-scale comparisons are explicit:

```bash
python -m research.serra.simulate --config research/serra/config/simulation/laplace_benchmark.json
python -m research.serra.endpoint --config research/serra/config/endpoint/laplace_benchmark.json
```

They do not validate an approximate seasonal-copula sampler. The main private
kernel always corrects GEV path proposals by MH. July 2019 output distinguishes
full-record smoothing from a forecast trained strictly before the selected event.
A finite GEV endpoint is a statistical support boundary, not a physical limit.

## 6. Validate intervals, annual aggregation and compound risks

```bash
python -m research.serra.validate --config research/serra/config/independent_full.json
python -m research.serra.validate --config research/serra/config/copula_full.json
python -m research.serra.forecast_check --config research/serra/config/forecast/paper.json
```

Coverage is reported at 90/95/99%, with tail quantiles, PIT and proper scores.
Separate latent-level, full-location and observation prediction intervals.
Finite Monte Carlo variances do not prove finite posterior moments near xi=.5.
The annual product is formed within a draw before averaging; it assumes
conditional residual independence across months, and uses complete years.

Compound-risk quadrature integrates bivariate residual noise within each draw,
avoiding a zero estimate simply because no rare event was simulated. The mean
of these probabilities estimates posterior predictive risk. Forecast bands
also contain sampled future-state variability; isolating epistemic uncertainty
would require nested future-path integration. Three-or-more-channel events
remain available through simulation, without pretending their counts are
posterior credible intervals for a probability.

Always inspect `ordering_in_sample.csv` and `ordering_forecast.csv`. The model
does not enforce min ≤ mean ≤ max; no draws are sorted or discarded. If crossings
are material, qualify compound interpretations or change the observation model.

## Optional data preparation

The release already contains validated monthly data. To rebuild it explicitly:

```bash
python -m research.serra.prepare_uccle
```

The helper scripts are `models.py` (declarations), `experiment.py` (matched
sensitivity cases), and `report.py` (public result APIs). There are no conference
subdirectories. Keep outputs outside source control; tiny execution outputs are
not bundled as paper results.

## Resource planning

The default full joint fit retains roughly 7.8 GB of centred state draws alone
(4 chains × 2000 draws × 1573 state times × 78 states × 8 bytes), before
diagnostics, forecasts and temporary copies. The six univariate jobs reduce
peak memory per fit. Choose chain/draw budgets and storage deliberately;
changing them is an explicit configuration choice, not a convergence remedy.
For large simulation grids, set `save_fits: false` when compact numerical
outputs and saved configurations suffice; retain selected fits for auditing.
