# BUCEX 1.7.4 — descriptive exploration in the SERRA workflow

Generate exploratory Figures 1 and 2 from the package checkout:

```bash
python -m pip install -e ".[plot]"
python -m research.serra.explore
```

The command uses the six monthly Uccle summaries, currently available through
August 2026, and writes manuscript-style PNG/PDF figures, source observations,
numerical tables and metadata in a fresh results/serra_exploration directory.
No posterior fitting is needed. Existing figure filenames are preserved.

- Figure 1: observed calendar-month means and empirical interquartile bands
  for 1892–1921 and September 1996–August 2026.
- Figure 2: residual interquartile ranges after separate linear detrending by
  calendar month in 1892–1936, 1937–1981 and 1982–August 2026.
- `explore_monthly` accepts other monthly Series/DataFrames; it is independent
  of Uccle and of any Gaussian/GEV model. It validates dates, records sample
  counts and leaves original observations unchanged.
- `MonthlyExploration` exposes tables, plots and a small report writer. The
  statistical, plotting and persistence responsibilities remain modular.
- `revision/exploration.json` controls windows, response order, colours,
  linestyles, panel layout, units and output formats.
- Existing MCMC, priors, forecasts, copulas and archive formats are unchanged.
  The `figures` script continues to rebuild figures from fitted-model reports.

These exploratory summaries do not constitute model validation or posterior
inference. Final scientific fits and their convergence remain a separate task.
Read docs/EXPLORATION.md and validation/RELEASE_VALIDATION.md for the API and
checks actually executed for this release.
