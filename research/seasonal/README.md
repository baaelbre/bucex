# Seasonal Uccle analysis: 1.9.3

TXm, TNm, TXx, TXn, TNx and TNn are fitted separately, with the same fixed Normal prior settings and separate trajectories, innovation coefficients and initial slopes. There are no learned shrinkage hyperparameters, copula or cross-response borrowing in the active grid.

The [command guide](../../BUCEX-1.9.3-commands.md) lists the calibration, all 23 settings and Gallade screen/paper commands. Six fits run simultaneously within each experiment job, with two parallel chains each and twelve reserved cores. The full grid has 138 posterior fits, four prospective pre-2019 TXx fits, 348 five-year validation fits and 18 original-design ten-year fits. Full-record fits use 538 complete seasons and forecast 120 seasons.

`config/main.json` declares the reference; `config/experiments.json` declares studies, origins and budgets. `job_plan` builds the submission plan with the standard library alone. `bundles` launches six response tasks; `jobs` enforces one response per task and verifies configurations. `collect_jobs` pairs marginal scores on identical cases. `export_results` writes compact evidence. `local` remains available for local/biobot scheduling.

`fixed_prior_settings.csv` records fixed coefficient SDs and physical calibration. Initial-slope and innovation prior/posterior reports remain available; hyperparameter posterior plots are absent. The single-channel container preserves Gaussian/GEV samplers, minima transformations and seasonal scale handling and never receives the other responses.

Frozen `reference_*.json` and `config/history_192/` retain older specifications for reproducibility. Legacy copula/hierarchical tools and 1.9.1/1.9.2 command guides are historical, not the active 1.9.3 grid.
