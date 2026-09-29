# BUCEX 1.9.6

Bayesian structural time-series models for temperature averages and extremes.
This release implements the final pooled half-normal screening design.

Start with **[BUCEX-1.9.6-commands.md](BUCEX-1.9.6-commands.md)** for complete BIOBOT
and VSC instructions, settings, job counts and output locations.

- Reference innovation scales: `(0.01, 0.0002, 0.01)` for level, slope and season.
- Separate initial slopes: `Normal(0, 0.01²)`; initial level SD 10 °C.
- Initial seasonal orthonormal contrast SD 10 °C, with exchangeable seasonal effects.
- 100 screen fits, two parallel chains each, 1,000 warm-up + 1,000 retained iterations.
- HN sensitivity grid, private reference checks, 35-year hindcasts, risk and 30-year forecasts.
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
