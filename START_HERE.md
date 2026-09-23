# BUCEX 1.8.3 — monthly analysis and a separate seasonal assessment

Summer 2026 **is included**. Only January and February 1892 are excluded from
these research runs. The raw source and bundled monthly CSVs remain intact.

- Monthly reference: March 1892–August 2026, **1,614 months**.
- Seasonal alternative: MAM 1892–JJA 2026, **538 complete seasons**.
- DJF means December–February, named for the January/February year. Seasonal
  files are timestamped at the block start: `2026-06-01` includes all of
  June, July and August, through `2026-08-31`.
- Six private FS location models, repeating observation scales, constant-in-time
  GEV shape, shared level/slope/seasonal innovation and initial-slope
  regularization, and a Gaussian copula. No PGAS; the reference uses neither
  SSVS nor ASIS. Mixed inference uses exact-likelihood-corrected Laplace–MH.
- `research/serra` is the monthly workflow. `research/serra_seasonal` is the
  separate seasonal experiment. They reuse the same package and model builder.

The current seasonal model uses **one maximum/minimum per season (r=1)**.
Rank extraction and clustering diagnostics are included to assess the future
r-largest extension. An r>1 state-space likelihood is **not implemented**;
the code never treats the three ranked values as three independent GEV draws.

## 1. Install and keep the session alive

Extract into a new directory. Run these commands from its package root,
beside `pyproject.toml`, leaving earlier code and results available.

```bash
python -m pip install -e ".[plot,test]"
python -c "import bucex; print(bucex.__version__, bucex.__file__)"
```

Expect `1.8.3` and the new source path. On biobot, four chains use four local
processes with one numerical thread each. Slurm is not required. Start long
jobs inside tmux if available:

```bash
tmux new -s serra183
```

Detach with Ctrl-b then d; reconnect using `tmux attach -t serra183`.
No mid-chain checkpoint is available. The matched comparison can resume
**completed model/origin jobs**; an interrupted fit restarts from the beginning.
Changing the data, settings or package version requires a new comparison.

## 2. Check the data and the priors first — no MCMC

```bash
python -m research.serra.explore
python -m research.serra_seasonal.prepare
python -m research.serra_seasonal.preflight
python -m research.serra.preflight --config research/serra/config/draft/main.json --output results/serra_183_monthly_plan
python -m research.serra_seasonal.compare --plan
```

The first command generates the monthly exploratory manuscript figures.
`prepare` writes `results/serra_183_seasonal_data/`:

- `seasonal_summaries.csv`: the six summaries, directly from the daily record;
  means weight every day equally, including different month lengths.
- `block_audit.csv`, `quality_report.json`: included/excluded blocks, day counts
  and the last included day. Confirm **2026-08-31**.
- `seasonal_records.png`, `seasonal_distributions.png`: exploration only.

The two preflight directories contain resolved settings, sampling plans and
`prior_calibration.csv`. These are declarations, not fitted results.

### The new reference priors

This release applies the COMPSTAT **marginal second-moment calibration** discussed
in the appendix. It replaces the previous quarter anchors in the draft configs.
The generic prior API and historical configurations remain available.

| Quantity | Monthly hyperprior median | Seasonal hyperprior median |
|---|---:|---:|
| Level innovation absolute-coefficient median | 0.008343480538 | 0.014451332204 |
| Slope innovation absolute-coefficient median | 0.00002085870135 | 0.00010883964876 |
| Seasonal innovation absolute-coefficient median | 0.008343480538 | 0.008343480538 |
| Initial-slope normal SD | 0.001546257845 | 0.004638773534 |

All four learned scales have lognormal hyperpriors with log SD `log(2)`.
After integrating these hyperpriors, the 30-year SD contributions to location
are 0.379473°C from level innovations, 0.196769°C from slope innovations, and
0.154919°C from same-season changes. Initial-rate SD is 0.30°C/decade.
These are component-specific prior RMS effects, **not 95% forecast limits**.

A decade is 120 monthly updates or 40 seasonal updates. Numerical priors cannot
be copied across these grids. Matching these physical effects does not make
the two models identical or assert that their full path priors agree at every
horizon. The independent fallback matches the same marginal second moments
using fixed normal priors; it has no learned common hyperparameter.

## 3. Run the two execution checks

```bash
python -m research.serra_seasonal.fit --config research/serra_seasonal/config/smoke.json
python -m research.serra_seasonal.compare --config research/serra_seasonal/config/compare_smoke.json
```

These use four parallel chains, 3 warmup and 4 retained draws per chain, on a
short recent window. Their purpose is to test the install, data, sampler,
archives and reports. Convergence flags are expected. Do not interpret their
trajectories, posterior intervals or model-score rankings scientifically.

## 4. First useful seasonal fit

```bash
python -u -m research.serra_seasonal.fit --config research/serra_seasonal/config/pilot.json
```

This uses the full record and all six summaries, with 500 warmup + 500 retained
draws per chain. It is a screening run. Check these files in the printed folder:

| File | What to check |
|---|---|
| `convergence.json`, `mcmc.csv`, `scientific_targets.csv` | R-hat, bulk/tail ESS; especially initial slopes, process SDs, shared scales and scale contrasts |
| `*_parameter_traces.png`, `*_scale_and_slope_traces.png` | Mixing of magnitudes and scientific quantities, not just signed FS coefficients |
| `shared_shrinkage.png`, `shared_shrinkage_calibration.csv` | Prior/posterior changes and physical scales; a displaced posterior is not itself a selection criterion |
| `*_level.png`, `*_slope_C_per_decade.png` | Warming and rate uncertainty; smoothness alone does not establish acceleration |
| `*_pit_qq_residuals.png`, `*_pit_by_season.png` | Seasonal skewness, cold-tail failures and residual memory |
| `posterior_predictive_checks.csv` | Observed tail/dispersion/skewness/lag discrepancies versus replicated conditional normal-score statistics |
| `residual_dependence_by_season.csv`, `copula_correlations.csv` | Whether the four fitted correlation matrices represent remaining contemporaneous dependence |
| `*_scale_by_season.png` | Each margin's observation scale; Gaussian SD versus GEV scale |
| `*_forecast_by_season.png`, `*_forecast_*_risk_curves.png` | Predictive intervals and event probabilities, including future-state uncertainty |
| `ordering_in_sample.csv`, `ordering_forecast.csv` | Incompatible simulated combinations; ordering is diagnosed, not enforced |

PPCs reuse fitted data and are descriptive. A seasonal copula does not repair
arbitrary marginal tail misspecification or daily/serial clustering.

For the first manuscript draft after screening:

```bash
python -u -m research.serra.copula --config research/serra/config/draft/main.json
python -u -m research.serra_seasonal.fit --config research/serra_seasonal/config/main.json
```

Both use four chains with 1,000 warmup + 1,000 retained draws. This draw count
is not a convergence guarantee; extend only fits whose diagnostics need it.

## 5. The important comparison: same seasonal forecast targets

```bash
python -u -m research.serra_seasonal.compare --config research/serra_seasonal/config/compare.json
```

This fits both models at the ends of **November 2015 and November 2020**, then
forecasts five years. Both models see the same daily training span and predict
the same 20 complete seasons after each cutoff. Hyperparameters are learned
only from each training set. All six means/extrema are scored on the same
observed seasonal quantities. Four fits are required, each with four chains
of 500 warmup + 500 retained draws. Full historical fits can still take time.

Monthly predictive paths are aggregated into day-weighted means, maxima or
minima. Conditional observation noise is integrated analytically for seasonal
CDFs and densities, then averaged over posterior/future-state draws. This is
not a comparison of raw monthly and quarterly likelihoods, AIC or BIC.

Look in the comparison run's `comparison/` directory:

- `paired_summary.csv`: **seasonal minus monthly** scores; negative favours
  seasonal. Compare CRPS and negative log predictive density together.
- `scores_by_season.csv`: does any apparent gain survive winter/summer separation?
- `scores_by_horizon.csv`: distinguish the first forecast year from years 2–5.
  This addresses the longer-horizon cold bias seen in previous runs.
- `scores_by_origin.csv`: is the gain consistent across training cutoffs?
- `calibration_summary.csv`, `seasonal_forecast_pit.png`: PIT, coverage and
  interval width. Narrower bands alone are not an improvement.
- `calibration_by_case.csv`: observed thresholds, event probabilities and event
  counts. A near-zero Brier score with no observed events does not validate rare tails.
- `monthly/<origin>/convergence.json` and `seasonal/<origin>/convergence.json`:
  score differences are provisional while relevant fits are poorly mixed.

The scores are marginal seasonal scores. They do not supply the joint density
of all six aggregated variables. The copula is still fitted jointly and retained
in simulated paths. A few origins do not justify an iid standard error or a
formal declaration of a winner. Unrestricted shape support can permit infinite
moments; finite-ensemble CRPS estimates should be read alongside log scores,
PITs and probability calibration, not as proof that all predictive moments exist.

If the pilot comparison is informative, expand the origins and sampling:

```bash
python -u -m research.serra_seasonal.compare --config research/serra_seasonal/config/compare_full.json
```

This adds November 2000 and November 2010, with 1,000 + 1,000 draws per chain
and 4,000 forecast paths. It is a new run. To continue an interrupted comparison,
use its actual printed run directory in place of `RUN_DIRECTORY`:

```bash
python -u -m research.serra_seasonal.compare --run RUN_DIRECTORY
```

Completed model/origin jobs are preserved. An unfinished job restarts. The
command checks the saved settings, source-data fingerprints and package version.

## 6. Targeted adequacy and prior sensitivity — later appendix

These use the same package and common assessment engine; no simulation study.
Run `--stage plan` before `--stage all` to see the amount of work.

```bash
python -m research.serra.prior_assessment --config research/serra_seasonal/config/adequacy.json --stage plan
python -u -m research.serra.prior_assessment --config research/serra_seasonal/config/adequacy.json --stage all
python -m research.serra.prior_assessment --config research/serra_seasonal/config/sensitivity.json --stage plan
python -u -m research.serra.prior_assessment --config research/serra_seasonal/config/sensitivity.json --stage all
```

Adequacy compares the reference, a common observation scale across seasons,
and fixed location seasonality. Sensitivity compares hyperprior widths while
holding the declared physical marginal second moments fixed; shape SD 0.15,
0.30 and 0.60; and unrestricted normal versus bounded normal versus bounded
uniform shape priors. Width changes alone would otherwise change prior RMSs.

The corresponding monthly experiments are:

```bash
python -u -m research.serra.prior_assessment --config research/serra/config/draft/adequacy.json --stage all
python -u -m research.serra.prior_assessment --config research/serra/config/draft/sensitivity.json --stage all
python -u -m research.serra.prior_assessment --config research/serra/config/draft/anchors.json --stage all
```

The last experiment halves/doubles all four physical regularization scales.
It is distinct from changing the uncertainty about those scales.

## 7. Rank and clustering exploration — no additional MCMC

```bash
python -m research.serra_seasonal.clusters
```

In `results/serra_183_clustering/`, inspect raw top/bottom-three dates and ties,
`rank_proximity_summary.csv`, `cluster_counts.csv`, and the cluster files for
runs of 1, 3 and 5 non-exceeding days. Thresholds are calendar-month 95th/5th
percentiles from 1961–1990, saved in `thresholds.csv`. Clusters are formed before
season assignment and a cross-boundary episode is assigned once by its peak.

These are sensitivity diagnostics, not a proof of independent extremes or an
estimated r-largest model. The fixed historical thresholds do not remove
warming. Nearby ranked days may belong to the same episode; the r-largest
likelihood would handle ranking dependence but needs additional assumptions
or treatment for daily clustering. Seasons with fewer than three qualifying
clusters remain visible, never padded with ordinary days. See
`docs/SEASONAL_ANALYSIS.md` for the exact scope.

## 8. Keep the fallback and finalize only the chosen specification

```bash
python -u -m research.serra.univariate --config research/serra/config/draft/independent.json
python -u -m research.serra_seasonal.fit --config research/serra_seasonal/config/independent.json
python -u -m research.serra_seasonal.fit --config research/serra_seasonal/config/independence.json
python -u -m research.serra_seasonal.fit --config research/serra_seasonal/config/constant_copula.json
```

The first two run six independent fits. The third keeps shared shrinkage but
fixes residual R=I. The fourth retains joint shrinkage with constant copula
correlation. The main model has four shrunk seasonal correlation matrices.

If diagnostics and the common-target comparison support the chosen model:

```bash
python -u -m research.serra.copula --config research/serra/config/draft/final.json
python -u -m research.serra_seasonal.fit --config research/serra_seasonal/config/final.json
```

Choose the relevant command; there is no need to run both final models just to
choose a paper narrative. Each uses four chains with 2,000 warmup + 4,000
retained draws. Assess precision and convergence again.

For the revision, favour seasonal blocks only if the forecast/adequacy evidence
supports the change and their coarser event definition suits the question. A
seasonal analysis cannot produce monthly event probabilities from the quarterly
summaries. Seasonal yearly aggregates use **previous December–November**;
January–December annual means cannot be reconstructed exactly by splitting DJF.
Do not compare GEV location paths across block sizes as if they were identical
estimands. Compare observed seasonal summaries, quantiles and event risks.
