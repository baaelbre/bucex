# Seasonal Uccle analysis — 1.9.2

The current job grid fits TXm, TNm, TXx, TXn, TNx and TNn separately. All six use the same reference prior settings, with independent innovation hyperparameters, initial slopes and state trajectories. There is no copula or borrowing across responses.

The [command guide](../../BUCEX-1.9.2-commands.md) gives bounded parallel screen/paper commands for biobot. The full grid has 148 posterior tasks, four prospective pre-2019 TXx tasks, 354 five-year validation tasks and 18 original-design ten-year tasks. Each task fits one response. Full-record fits use 538 complete seasons, MAM 1892–JJA 2026, and forecast 120 seasons.

`config/main.json` declares the reference. `config/experiments.json` declares the studies, origins and budgets. `jobs --verify` checks the actual single-response models, prior units and complete folds. `local` schedules jobs; `collect_jobs` pairs validation by response and identical cases; `export_results` writes a compact review ZIP.

Private scale outputs are called `independent_shrinkage*.csv`. The single-channel model container preserves the established Gaussian/GEV sampler, minima transformations and seasonal scale handling; it never receives the other five responses.

Frozen `reference_191.json`, `reference_189.json` and `reference_20260923.json` retain historical specifications. Legacy grid/compare tools and old command guides are not the current 1.9.2 workflow. Use the root launchers for the new analysis.
