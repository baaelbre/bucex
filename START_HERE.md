# Start with BUCEX 1.7.2

Extract the release and open a terminal in `bucex-1.7.2`, beside
`pyproject.toml`. Commands work in Windows PowerShell, Linux and macOS.

```text
python -m pip install -e ".[plot]"
python -c "import bucex; print(bucex.__version__, bucex.__file__)"
```

Version 1.7.2 preserves the 1.7.1 inference kernels and prior defaults. You do
**not** need new chains just to change figures. A change to observation scale,
dependence, priors or the training data does require a new fit.

## Your next scientific run

Keep your six existing constant-scale fits as the baseline. Check the monthly
scale extension first for TNm, with all location innovation priors unchanged:

```text
python -m research.serra.preflight --config research/serra/config/revision/monthly_scale.json
python -m research.serra.univariate --config research/serra/config/revision/monthly_scale.json --series TNm
```

This uses January 1892 to August 2026, normal FS priors, physical innovation-SD
prior medians (.01, .00005, .02), initial-slope SD .0025 per month, constant
unknown shape, and repeating monthly scales. The default starting budget is
four chains, 2,000 warmup and 2,000 retained draws per chain; it does not promise
convergence. ASIS and discrete selection are off.

Inspect `convergence.json`, initial-slope and process-SD traces, scale estimates,
`TNm_pit_by_month.csv/png`, `TNm_normal_score_by_month.png`, slopes and risks.
Smoothed PITs describe in-sample fit; use held-out predictions for calibration.
If satisfactory, run the other five:

```text
python -m research.serra.univariate --config research/serra/config/revision/monthly_scale.json --series TXm TXx TXn TNx TNn
```

## Matched joint fits

```text
python -m research.serra.preflight --config research/serra/config/revision/copula_monthly.json
python -m research.serra.copula --config research/serra/config/revision/copula_monthly.json --independence
python -m research.serra.copula --config research/serra/config/revision/copula_monthly.json
```

The two joint fits use the same marginal specification. The latter estimates a
constant residual Gaussian correlation matrix and updates margins jointly.
The stored state array is about 8.07 GB at this draw count; computation and
reporting need additional memory. Constant-scale alternatives are
`revision/constant_scale.json`, `revision/independence_constant.json` and
`revision/copula_constant.json`. Independent marginal analyses remain usable
if joint inference is not yet reliable.

## Figures from existing results, without refitting

For compact reports already supplied:

```text
python -m research.serra.figures --reports PATH_TO_SIX_SERIES_REPORT_ROOT
```

Or give six explicit series directories after `--reports`. A dated directory
contains coordinated manuscript PNGs and `figure_manifest.json`. Add
`--formats png pdf` for vector versions. Figure names match the rewritten
manuscript, including its two appendix panels. A missing trace CSV creates a
clearly marked placeholder; the other figures still render.

To obtain the newly exported compact parameter traces from a saved fit:

```text
python -m research.serra.report --fit PATH_TO_TXn/fit.bucex --format png --level 0.95
```

Use the newly printed report directory in the figure command. Re-reporting
also creates monthly diagnostics and includes initial-slope traces. It reads
the large archive but does not run MCMC or change the fitted model.

For a final figure export, add `--strict` so missing inputs stop execution.
See [figure recipes](docs/FIGURES.md) for selecting panels and changing style.

## Sensitivity

```text
python -m research.serra.sensitivity --config research/serra/config/revision/sensitivity_monthly.json --series TNm --variants normal_reference level_half level_double
```

Compare period warming, slopes, calibration and risks. Do not select a prior
just because a curve looks smooth. The [complete run guide](docs/PUBLICATION_RUNS.md)
covers fixed/evolving seasonality, shape bounds and priors, monthly-scale
priors, held-out comparisons, endpoint checks and recovery simulations.

Check execution locally with `revision/monthly_scale_smoke.json` and
`revision/copula_monthly_smoke.json` before scheduling expensive work. These
36-month, four-draw fits cannot provide scientific evidence.


## Validation

In report.py, posterior predictive observations are generated from the fitted model and compared to the original data, through PP and QQ plots. 
```text
python -m research.serra.validation --config research/serra/config/revision/validation.json
```
