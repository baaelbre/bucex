# BUCEX 1.9.6.1

Bayesian structural time-series models for temperature averages and extremes.
This patch adds a focused joint level–slope calibration study while preserving the statistical model from 1.9.6.

Start with **[BUCEX-1.9.6.1-commands.md](BUCEX-1.9.6.1-commands.md)** for complete BIOBOT
and VSC instructions, settings, job counts and output locations.

- Reference innovation scales: `(0.01, 0.0002, 0.01)` for level, slope and season.
- Separate initial slopes: `Normal(0, 0.01²)`; initial level SD 10 °C.
- Initial seasonal orthonormal contrast SD 10 °C, with exchangeable seasonal effects.
- New 25-cell joint grid: 75 fits on Gallade and 75 on BIOBOT.
- Two parallel chains per screen fit, 1,000 warmup + 1,000 retained iterations each.
- Focused 30-year hindcasts (truncated at the available data), risk and 30-year forecasts.
- Neighbouring-cell stability checks, pointwise 95% intervals, paired-draw diagnostics and an automatic HTML review.
- The previous 100-fit `all` suite remains available separately; do not combine it with the focused grid unless intended.
- No copula, heavy-tailed hyperprior or leave-one-response-out jobs in the active plan.
- Four chains and longer budgets for the separate paper tier.

```bash
python -m pip install -e '.[plot,test]'
python -m research.seasonal.jobs --verify --tier screen
python -m research.seasonal.jobs --list --tier screen --batch all
```

Configuration checks compile all models and verify folds and matched marginal priors.
They do not establish posterior convergence. Screen outputs preserve numerical flags.

The source includes the bundled daily record, the `bucex` API, research runners,
Slurm/PBS launchers, prior simulations, tests and reusable figure/report builders.
See `RELEASE_NOTES.md` for changes and `RELEASE_VALIDATION.json` for release checks.
