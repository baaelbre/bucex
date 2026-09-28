# Seasonal analysis in 1.9.4

The active protocol is in `research/seasonal/config/experiments.json` and
`BUCEX-1.9.4-commands.md`. It uses 538 complete seasonal blocks, MAM 1892 to
JJA 2026, from the bundled daily source. All three specifications use the same
observation models and direct SD convention.

The screen contains the three-way shrinkage comparison and sensitivity checks.
It does not include historical contrasts, recovery or endpoint validation.
Terminal levels and rates are retained as MCMC diagnostic targets. The
full-record fits forecast 120 seasons with 95% intervals.

Validation is run separately. Five-year rolling forecasts and the original
60/80/90% splits with ten-year forecasts are available as distinct batches.
Shared hyperparameters are re-estimated within each training period. Collection
pairs only matching origins and forecast cases and preserves convergence flags.

The archived configurations under `config/history_193` and `config/history_192`
are historical evidence, not active screen settings.
