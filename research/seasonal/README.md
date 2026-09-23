# Seasonal research

The seasonal model fits one mean or extreme per complete DJF/MAM/JJA/SON block, using the same model builder and sampler as [monthly research](../monthly/README.md). Season means weight all complete daily observations. The series ends with JJA 2026.

1. `python -m research.seasonal.prepare` audits and derives seasonal blocks.
2. `python -m research.seasonal.preflight` checks units, priors and computation.
3. `python -m research.seasonal.fit --config research/seasonal/config/smoke.json` checks execution.
4. `python -m research.seasonal.fit --config research/seasonal/config/main.json` fits the reference.
5. `python -m research.seasonal.compare --config research/seasonal/config/compare.json` scores matched monthly and seasonal forecasts on the same seasonal outcomes.

## Fast level/slope prior screen (1.8.5)

From the unpacked package directory on the biobot:

```bash
python -m research.seasonal.grid --dry-run
python -m research.seasonal.grid --only reference
python -u -m research.seasonal.grid --jobs 10
```

The first command lists the ten **seasonal-step** prior settings. The second checks one actual fit and its forecast exports; the third runs all unfinished settings concurrently. With `--jobs 10`, there can be ten setting processes plus **two chain workers per setting** (20 active MCMC chains); the two training origins run in sequence within each setting. Use `--jobs 4` to limit concurrent memory use. Progress for each setting is saved in `results/serra_185_seasonal_grid/logs/`; only the driver writes the shared ranking files after a complete setting. It is safe to rerun the third command: completed settings with the same saved configuration are reused. The reference uses level/slope/seasonal medians `(0.01, 0.0001, 0.01)` per season, with an initial-slope scale of `0.003` per season (about `0.194 degC/decade` after integrating over the shared scale). Nine settings cross level and slope factors `(0.5, 1, 2)`; the tenth tests `0.0045` for the initial-slope scale at the reference level and slope medians.

Each setting uses **2 chains, 300 warmup and 300 retained draws**, at two predeclared origins (November 2015 and November 2020) with **12 held-out seasons** after each. The remaining 2024--2026 blocks are reserved for a separate confirmation after choosing candidates. The script retains compact scores rather than full validation fits. The output directory `results/serra_185_seasonal_grid` contains `grid_summary.csv`, `grid_by_response.csv`, `selection.json`, the exact configuration and source-data checksum, and per-case CRPS, PIT, joint log score and convergence reports for each setting. It ranks **mean marginal CRPS** across the same six responses and the same 144 forecast cases; inspect each response's score, joint score and PIT too. Numerical checks use pilot thresholds of R-hat 1.05 and ESS 100; `best_passing_pilot_setting` is null if none passes. Even the winning pilot setting requires long four-chain runs and a fresh held-out assessment before its inferential results enter the paper. Trying different priors on the same validation blocks uses those blocks for selection; the selected score is not an unbiased estimate of final forecast skill.

[`config/adequacy.json`](config/adequacy.json) compares the reference seasonal observation scales with `constant_dispersion` and fixed location seasonality. Run `research.monthly.prior_assessment --config research/seasonal/config/adequacy.json --stage plan` to inspect those candidates. `clusters.py` diagnoses raw daily ranks and clustering; the fitted likelihood remains one extreme per season.
