# SERRA research with BUCEX 1.7.3

Start with [START_HERE](../../START_HERE.md). Complete sequential commands and
output interpretation are in [the publication run guide](../../docs/PUBLICATION_RUNS.md).
The workflow fits through **August 2026**, preserves six independent analyses,
and adds dependence through matched R=I/copula models.

Study declarations live in `models.py`; inference, diagnostics, predictive
calculations, style and manuscript panel construction live in BUCEX. JSON
configurations contain study choices. No conference-specific scripts are needed.

| Stage | Script | Principal revision configuration |
|---|---|---|
| Focused prior/posterior and forecast comparison | `prior_assessment` | `priors/pilot.json`; start with `--stage plan` |
| Resolve assumptions/resources | `preflight` | `revision/monthly_scale.json` |
| Constant-scale baseline | `univariate` | `revision/constant_scale.json` |
| Repeating monthly scales | `univariate` | `revision/monthly_scale.json` |
| Matched joint dependence | `copula` | `revision/copula_monthly.json`; add `--independence` for R=I |
| Prior sensitivity | `sensitivity` | `revision/sensitivity_monthly.json`, `revision/sensitivity_copula_monthly.json` |
| Fixed/evolving seasonality | `sensitivity` | `revision/structure.json` |
| Held-out assessment | `validate`, `model_comparison`, `compare` | `revision/model_comparison.json` |
| Actual July 2019 endpoint | `endpoint` | `revision/endpoint_monthly.json` |
| Controlled recovery | `simulate`, `joint_recovery` | `simulation/*`, `revision/joint_recovery_pilot.json` |
| Forecast widths | `forecast_check` | `revision/forecast_TXx_20y.json` |
| Historical source comparison | `univariate` | `revision/historical_monthly_scale.json` |
| Re-report saved draws | `report` | Saved run config or explicit report configuration |
| Manuscript figures | `figures` | `revision/figures.json` |

Use `python -m research.serra.SCRIPT --help`. `univariate`, `sensitivity` and
`model_comparison` accept `--series`. The copula's `--scale` controls marginal
scale; its `--structure` controls dependence seasonality. Existing configuration
paths are retained for compatibility; `revision/` is the documented paper workflow.

The fixed scale/shape baseline remains part of the paper. Monthly-scale fits
test a specific residual deficiency; they do not imply stochastic volatility.
Normal innovation priors remain the reference, with .01 level-innovation SD
prior median. No SSVS or ASIS is required for these runs.

The script creates a new timestamped result directory and prints it. Keep the
resolved configuration and `.bucex` fit locally. For initial review, share small
CSVs/JSONs, PNGs and `*_traces.csv.gz`. Reporting does not re-estimate the posterior.
A new likelihood, prior, data window or dependence model does.

## Focused innovation-prior work

`prior_assessment.py` orchestrates the existing fit/validation drivers. One
candidate list is used in both stages; `bucex.SensitivityReport` owns table and
figure comparison. `--stage sensitivity`, `--stage predictive` and `--stage
report` can run separately using the same saved assessment directory. Default
pilot: TNm, three normal-prior settings, four process workers, 500+500
iterations per chain and three historical five-year forecast blocks. No
simulation study is invoked. See [the new guide](../../docs/PRIOR_ASSESSMENT.md).
