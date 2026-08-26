# Uccle six-series analysis

## Recommended primary analysis

The six bundled monthly series are analysed as six independent univariate
models. This is the primary specification in bucex 1.5.2:

| Series | Description | Observation family | JSON | Example/engine |
|---|---|---|---|---|
| TXm | monthly mean daily maximum | Gaussian | `uccle_gaussian/01_txm.json` | 12 / exact FFBS |
| TNm | monthly mean daily minimum | Gaussian | `uccle_gaussian/02_tnm.json` | 12 / exact FFBS |
| TXx | monthly maximum daily maximum | GEV upper tail | `uccle/01_txx.json` | 09 / exact Laplace-MH |
| TXn | monthly minimum daily maximum | GEV lower tail | `uccle/02_txn.json` | 09 / exact Laplace-MH |
| TNx | monthly maximum daily minimum | GEV upper tail | `uccle/03_tnx.json` | 09 / exact Laplace-MH |
| TNn | monthly minimum daily minimum | GEV lower tail | `uccle/04_tnn.json` | 09 / exact Laplace-MH |

The bundled records run monthly from January 1892 through December 2022. A
`null` `data.end` selects the latest bundled observation. TXn and TNn are
negated internally so one numerically stable upper-tail GEV implementation can
be used; results, figures, predictions, and risk summaries are transformed
back to their original orientation.

## Shared structural model

All six series use a local-linear trend and 12-month dummy seasonality:

```text
mean/predictor_t = level_t + seasonal_t
level_t          = level_(t-1) + slope_(t-1) + level innovation
slope_t          = slope_(t-1) + slope innovation
seasonal_t       = sum-to-zero dummy-seasonal recursion + seasonal innovation
```

Structural SSVS averages over:

| Component | Candidate states |
|---|---|
| level | fixed, dynamic |
| slope | absent, fixed, dynamic |
| seasonality | fixed, dynamic |

The Gaussian mean models use

\[
Y_t\mid\mu_t,\sigma^2 \sim N(\mu_t,\sigma^2).
\]

The four extreme models use

\[
Y_t\mid\eta_t,\sigma,\xi \sim \operatorname{GEV}(\eta_t,\sigma,\xi),
\]

with stationary scale in the primary analysis. The two observation families
share the structural prior but not an observation model: fitting TXm/TNm as
GEV extremes would be scientifically wrong, while Gaussian FFBS gives exact
conditional state simulation for monthly means.

## Primary hyperparameters

| JSON field | Value | Interpretation |
|---|---:|---|
| `alpha_sd` | 3.2 | initial level SD in °C |
| `beta_mean` | 0 | initial/fixed slope mean |
| `beta_sd` | 0.0025 | slope SD in °C/month (0.30 °C/decade) |
| `seasonal_initial_sd` | 2.25 | initial seasonal-state SD in °C |
| `sigma2` | IG(2, 2) | Gaussian variance or GEV scale-squared prior |
| `xi_bounds` | (-0.5, 0.5) | GEV shape support; extremes only |
| `innovation_slab_sd.level` | 0.02 | level innovation slab SD |
| `innovation_slab_sd.trend` | 0.00005 | slope innovation slab SD |
| `innovation_slab_sd.season` | 0.02 | seasonal innovation slab SD |
| `level_dynamic_probability` | 0.50 | prior P(dynamic level) |
| `trend_probabilities` | (0.20, 0.40, 0.40) | absent, fixed, dynamic |
| `season_probabilities` | (0.00, 0.50, 0.50) | absent, fixed, dynamic |

The seasonal component cannot be absent because strong annual seasonality is
known before seeing these data. It may still be fixed or evolving. These are
the primary calibrated values, not values estimated from the attached result
sets. A sensitivity analysis should change one defensible block at a time.

## MCMC and inference defaults

Every production JSON requests 1,000 warmup iterations and 1,000 retained
draws **per chain**, with four independent chains. Thus each fit retains 4,000
posterior draws before any downstream subsampling.

- TXm/TNm use `engine: "ffbs"`; the Gaussian state update is exact.
- TXx/TXn/TNx/TNn use `engine: "laplace_mh"`; Laplace supplies a full-path
  proposal and an exact Metropolis-Hastings correction targets the GEV
  posterior.
- Four chains are independent and may execute as four one-core processes.
- Increase draws/warmup only if R-hat, ESS, Monte Carlo errors, or chain
  switching require it. Longer chains do not repair a misspecified prior.

## Run the primary analyses locally

From the repository root, either call Python directly (chains run inside one
process) or use the Bash wrappers (chains run as separate processes):

```bash
python examples/12_uccle_gaussian.py --config examples/config/uccle_gaussian/01_txm.json
python examples/12_uccle_gaussian.py --config examples/config/uccle_gaussian/02_tnm.json

python examples/09_uccle_laplace_mh.py --config examples/config/uccle/01_txx.json
python examples/09_uccle_laplace_mh.py --config examples/config/uccle/02_txn.json
python examples/09_uccle_laplace_mh.py --config examples/config/uccle/03_tnx.json
python examples/09_uccle_laplace_mh.py --config examples/config/uccle/04_tnn.json
```

Process-parallel examples:

```bash
bash bash_scripts/run_12_uccle_gaussian.sh examples/config/uccle_gaussian/01_txm.json 4
bash bash_scripts/run_09_uccle_laplace_mh.sh examples/config/uccle/01_txx.json 4
```

All settings come from the selected JSON. The final argument is only the
maximum number of concurrent chain processes and must be at least
`mcmc.chains` for a production file.

## Optional log-scale sensitivity for extremes

Example 11 fits `phi_t = log(sigma_t)` as stationary, linear, random walk, or
scale-model SSVS. There are 16 complete JSONs under
`examples/config/phi/uccle/`. Their location prior is exactly the primary
prior above; only the scale specification varies.

```bash
python examples/11_uccle_phi.py --config examples/config/phi/uccle/01_txx_linear.json
python examples/11_uccle_phi.py --config examples/config/phi/uccle/01_txx_rw.json
python examples/11_uccle_phi.py --config examples/config/phi/uccle/01_txx_ssvs.json
```

Use suffixes `_stationary`, `_linear`, `_rw`, or `_ssvs`, and prefixes
`01_txx`, `02_txn`, `03_tnx`, or `04_tnn`. TXm/TNm are Gaussian mean models
and do not use GEV log-scale selection in example 11.

The separate `uccle/*_narrow.json` files remain available as a **location-slab
sensitivity** for the four primary GEV analyses. They use slabs
`(0.01, 0.000025, 0.01)`. Do not call them primary, and do not mix their
conclusions with the scale-model sensitivity without stating that two prior
blocks changed.

## Outputs

Each fitted series writes:

```text
results/<example>/<run-id>__<signature>/
  run_config.json
  fits/<series>/combined.bucex
  tables/<series>/
    parameters.csv diagnostics.csv algorithm.csv
    trajectory.csv selection.csv models.csv switching.csv
    posterior_predictive.csv forecast.csv forecast_july.csv
    forecast_level.csv summary.json
  figures/<series>/
    trajectory.* trajectory_july.* posterior_predictive.*
    forecast.* forecast_july.* forecast_level.*
    level.* level_no_observations.* slope.* selection.*
    process_sd.* season.* seasonal_patterns.*
```

Gaussian fits add `gaussian.*`; GEV fits add `gev.*` and a finite endpoint
figure when applicable. Phi fits add phi/sigma paths and scale-model summaries.
Tables are always written; formats, DPI, diagnostics, interval probability,
predictive draws, focus month, forecast horizon/history, and selected seasonal-
pattern years are JSON fields. `season.*` follows each month through time;
`seasonal_patterns.*` instead compares the complete January--December pattern
for the years in `figures.seasonal_patterns.years` (1892 and 2022 by default).

## What to report

- component and joint structural probabilities, including switching by chain;
- model-averaged predictor, level, slope, and seasonal trajectories;
- posterior process scales and observation parameters;
- R-hat, ESS, Monte Carlo errors, and GEV state acceptance where applicable;
- posterior-predictive and forecast checks;
- finite GEV endpoints and time-indexed risk summaries where applicable;
- separately labelled location-slab and log-scale sensitivity results.

A return level at time `t` is conditional on parameters at that time; it is not
a timeless property of the whole 1892–2022 record.
