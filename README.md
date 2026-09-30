# BUCEX 1.9.8.1

Bayesian structural time-series models for temperature averages and extremes.
This release compares pooled half-normal shrinkage, separate half-normal hierarchies,
and separate fixed Normal shrinkage priors with **no hyperpriors**.

See [BUCEX-1.9.8.1-commands.md](BUCEX-1.9.8.1-commands.md) for the complete screen commands.

- Level scales: **0.05, 0.1, 0.5**; slope scales: **0.001, 0.002, 0.005, 0.01**.
- The 3×4 grid fixes the seasonal innovation scale at 0.1. Two additional seasonal checks use 0.05 and 0.2 at the reference (level 0.1, slope 0.002).
- All 14 settings have pooled HN, separate HN and separate fixed-Normal full-record fits.
- Pooled and fixed-Normal fits have matched validation at five origins; separate HN validation covers the reference.
- Fixed fits learn the signed innovation coefficients and process variances. Only their Normal prior SDs are fixed.
- Equal calibration values match marginal coefficient variances across constructions, not their marginal prior distributions. Shared/private HN one-response priors match exactly.
- Screen: 2 parallel chains × (1,000 warmup + 2,000 retained iterations).
- Gallade: **390 individual fits in 100 array elements**. BIOBOT: **312 individual fits**. The host batches are disjoint.
- Full-record forecasts: 30 years, 50,000 predictive draws. Held-out forecasts: up to 30 years, 10,000 draws.
- Pointwise 95% trajectory intervals and 90/95/99% validation coverage; matched CRPS, PIT, risk, allocation and convergence diagnostics.
- Initial level and seasonal contrast SDs remain 10 °C; initial slope SD remains 0.01 °C per seasonal step.
- Conditional observation independence; exact GIG hyperparameter updates retained for HN hierarchies.

```bash
python -m research.seasonal.jobs --verify --tier screen
python -m research.seasonal.jobs --list --tier screen --batch sweetspot
```

The two focused launchers use `sweetspot_hpc` and `sweetspot_validation`.
The optional `sweetspot_fixed` batch contains only the 504 fixed-prior fits;
it overlaps the complete study, so use it only when intentionally running that subset.
The older general-purpose `all` batch is retained and is **not** the new grid.

Use a fresh output root. Screening is for numerical and sensitivity assessment;
short-run checks do not establish convergence. Release verification is recorded in
`RELEASE_VALIDATION.json`. No production jobs are submitted by downloading this release.
