# Seasonal research

The seasonal model fits one mean or extreme per complete DJF/MAM/JJA/SON block, using the same model builder and sampler as [monthly research](../monthly/README.md). Season means weight all complete daily observations. The series ends with JJA 2026.

## Fix and verify the sampler initialization (1.8.6.1)

Earlier 1.8.5/1.8.6 runs initialized dummy-seasonal states with chronological
phase effects, although the state transition expects lag-ordered coordinates.
This could put a GEV chain far from a plausible fit and leave it drifting after
300 warm-up iterations. Existing short-grid scores are exploratory only; the
new grid uses a separate output directory and requires two passing reference
mixing checks before it starts. The likelihood and priors have not changed.

From the unpacked package directory on the biobot, run the reference model
at both grid forecast origins. Each command fits four parallel chains and
saves compact per-chain parameter and log-likelihood traces, without forecasts:

```bash
python -u -m research.seasonal.diagnose_mixing --origin 2015-11 --output results/serra_186_seasonal_mixing/2015_copula
python -u -m research.seasonal.diagnose_mixing --origin 2020-11 --output results/serra_186_seasonal_mixing/2020_copula
```

Each uses 1,500 warm-up and 1,500 retained draws per chain. Check
`convergence.json`, `parameters.csv`, `targets.csv`, `run.json` (quarter-by-quarter
log-likelihood averages) and `traces.npz` before treating either fit as
stable. The check requires four chains, R-hat below 1.01, and bulk and tail
ESS at least 400 for shape, scale, seasonal contrasts, initial coefficients,
shared shrinkage, copula parameters and scientific contrasts. A failed gate
exits with status 2.
If it fails, diagnose the listed parameters before running the grid; extending
warm-up alone is not a guarantee of mixing.

## Reviewer comment 5: held-out extremes (1.8.6)

`config/comment5.json` defines seven five-year seasonal forecast windows,
four-chain fits, and separate 90%, 95%, 99% coverage and directional 1%/5%
tail summaries. Run from the package root:

```bash
python -m research.seasonal.preflight --config research/seasonal/config/comment5.json --output results/serra_186_seasonal_comment5_plan
python -u -m research.monthly.validate --config research/seasonal/config/comment5.json
```

Inspect the printed result directory under `joint/`: `folds.csv`,
`convergence_*.json`, `central_coverage.csv`, `directional_tails.csv`,
`threshold_events.csv` and the corresponding case tables. Reports include
denominators by response, season and forecast year. With 140 held-out seasons
per response, the expected number above a 99th percentile is only 1.4, so
do not treat an absence of exceedances as evidence of tail calibration.
The 2015 and 2020 windows reuse the 1.8.5 prior-screen periods; if that
screen determines the prior, interpret those two windows as exploratory.

1. `python -m research.seasonal.prepare` audits and derives seasonal blocks.
2. `python -m research.seasonal.preflight` checks units, priors and computation.
3. `python -m research.seasonal.fit --config research/seasonal/config/smoke.json` checks execution.
4. `python -m research.seasonal.fit --config research/seasonal/config/main.json` fits the reference.
5. `python -m research.seasonal.compare --config research/seasonal/config/compare.json` scores matched monthly and seasonal forecasts on the same seasonal outcomes.

## Fast level/slope prior screen (after mixing passes)

From the unpacked package directory on the biobot:

```bash
python -m research.seasonal.grid --dry-run
python -m research.seasonal.grid --only reference
python -u -m research.seasonal.grid --jobs 10
```

The first command lists the ten **seasonal-step** prior settings. The other commands require the two passing reference mixing checks above. With `--jobs 10`, there can be ten setting processes plus **two chain workers per setting** (20 active MCMC chains); the two training origins run in sequence within each setting. Use `--jobs 4` to limit concurrent memory use. Progress for each setting is saved in `results/serra_1861_seasonal_grid/logs/`; only the driver writes the shared ranking files after a complete setting. It is safe to rerun the third command: completed settings with the same saved configuration are reused. The reference uses level/slope/seasonal medians `(0.01, 0.0001, 0.01)` per season, with an initial-slope scale of `0.003` per season (about `0.194 degC/decade` after integrating over the shared scale). Nine settings cross level and slope factors `(0.5, 1, 2)`; the tenth tests `0.0045` for the initial-slope scale at the reference level and slope medians.

Each setting uses **2 chains, 300 warmup and 300 retained draws**, at two predeclared origins (November 2015 and November 2020) with **12 held-out seasons** after each. The remaining 2024--2026 blocks are reserved for a separate confirmation after choosing candidates. The script retains compact scores rather than full validation fits. The output directory `results/serra_1861_seasonal_grid` contains `grid_summary.csv`, `grid_by_response.csv`, `selection.json`, the exact configuration and source-data checksum, and per-case CRPS, PIT, joint log score and convergence reports for each setting. It compares **mean marginal CRPS** across the same six responses and the same 144 forecast cases; inspect each response's score, joint score and PIT too. Numerical checks use pilot thresholds of R-hat 1.05 and ESS 100; no candidate is selected if none passes. Any pilot candidate requires long four-chain runs and a fresh held-out assessment before its inferential results enter the paper. Trying different priors on the same validation blocks uses those blocks for selection; the selected score is not an unbiased estimate of final forecast skill.

[`config/adequacy.json`](config/adequacy.json) compares the reference seasonal observation scales with `constant_dispersion` and fixed location seasonality. Run `research.monthly.prior_assessment --config research/seasonal/config/adequacy.json --stage plan` to inspect those candidates. `clusters.py` diagnoses raw daily ranks and clustering; the fitted likelihood remains one extreme per season.
