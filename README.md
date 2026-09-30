# BUCEX 1.9.8

Bayesian structural time-series models for temperature averages and extremes.
This release tests a broader half-normal calibration region and matched pooled/private fits.

Start with [BUCEX-1.9.8-commands.md](BUCEX-1.9.8-commands.md) for the settings, screen commands, CPU budgets, logs and results.

- Core 3×3 grid: A_alpha = 0.05, 0.1, 0.2; A_beta = 0.001, 0.002, 0.004; A_gamma = 0.1.
- Reference (0.1, 0.002, 0.1); seasonal checks 0.05 and 0.2 at the central level/slope setting.
- All 11 calibrations have shared and private full-record analyses with matched one-response priors. Private fits retain shrinkage but do not pool scales.
- Two screen chains per fit, 1,000 warmup and 2,000 retained draws each.
- Gallade: 111 fits in 46 groups. BIOBOT: 51 recent-origin validation fits. The batches are disjoint.
- Initial rates remain independent, SD 0.01 per season. Initial level/seasonal contrast SDs remain 10 °C.
- Identity copula; half-normal scale priors; exact GIG updates retained from 1.9.7.
- Thirty-year forecasts, truncated held-out windows, 95% trajectory intervals and 90/95/99% validation coverage.
- Automatic title-free figures, separate grid/seasonal/pooling comparisons and explicit numerical gates.

```bash
python -m pip install -e '.[plot,test]'
python -m research.seasonal.jobs --verify --tier screen
python -m research.seasonal.jobs --list --tier screen --batch sweetspot
```

Verification compiles the models, checks folds/resources and exact marginal prior matching. It does not establish convergence. Short compute probes exercise both pooled and private Gaussian/GEV fits. See `RELEASE_VALIDATION.json` for release checks and [the GIG notes](docs/GIG_UPDATES_197.md) for the sampler derivation.

The legacy `all`, `deferred` and explicit experimental variants remain available for compatibility; they are not included by the two focused launchers. The preceding 1.9.7 calibration files are archived under `research/seasonal/config/history_197/`. Use a fresh results root with this source.
