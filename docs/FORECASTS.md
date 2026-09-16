# Calendar forecasts, risk and diagnostic figures

The application data are monthly summaries, not daily observations. Full SERRA
configurations use the latest bundled month, August 2026 in this release.
`data.end: null` prevents an old protocol cutoff silently excluding new data.
The actual fitted dates are printed and saved in `data_window.json` and `run.json`.
A saved fit is immutable evidence about its original training record: report
generation cannot extend it to later observations. A new fit is required.

## Fit through the latest month

From the extracted source directory:

```bash
python -m pip install -e '.[test]'
python -m research.serra.univariate --config research/serra/config/independent_full.json --series TXm
```

Omit `--series TXm` to fit all six independently. The full default is four
chains, 2000 warmup and 2000 retained draws per chain, with ASIS. This is a
starting budget, not a convergence guarantee. Commands without `--config`
still use four-draw execution checks. Read the printed chain/draw/date counts.

For the old manuscript period, choose `independent_1892_2022.json` or
`copula_1892_2022.json`. When changing the actual source data, first update
`research/serra/config/prepare_uccle.json` and run `research.serra.prepare_uccle`.
Full fitting follows the refreshed monthly CSVs. Existing local saved JSON
configurations with `end: "2022-12-01"` continue to request that cutoff.

## Existing fit: more figures without more MCMC

```bash
python -m research.serra.report --fit PATH_TO_RUN/fit.bucex
python -m research.serra.report --fit PATH_TO_RUN/fit.bucex --months 1 7 8 --horizon 120
python -m research.serra.check_updates --fit PATH_TO_RUN/fit.bucex
```

`report` reads `config.json` beside the fit unless `--config` overrides it.
It writes a new directory, defaults to PNG at 150 dpi, and does not duplicate
the posterior archive. `--format pdf` is available. `check_updates` evaluates
later bundled observations from the saved forecast origin; it is a fixed-origin
multistep check, not rolling one-step validation. Fit mixing and record source
comparability still affect its interpretation. For rolling validation use
`research.serra.validate` with the intended full configuration.

## What to inspect

`NAME` is the saved series/channel name, such as `TXm`.

| File | Meaning and useful question |
|---|---|
| `NAME_slope.png`, `.csv` | Latent trend slope in °C/decade, with pointwise posterior intervals. Is change well identified? |
| `NAME_target_traces.png`, `NAME_target_mcmc.csv` | Mixing of endpoint slope and whole-record level change. |
| `NAME_parameter_traces.png` | Chain traces and autocorrelation of physical innovation SDs, baseline observation scale and shape where relevant. |
| `NAME_pit_qq_residuals.png` | Smoothed PIT histogram, normal-score Q-Q, residual time series and residual ACF. Descriptive in-sample checks. |
| `NAME_scale_mcmc.csv`, `NAME_scale_traces.png` | Mixing of monthly residual scales; baseline sigma alone can conceal poor mixing of seasonal effects. |
| `NAME_scale_by_month.png`, `.csv` | Monthly residual SD/GEV scale, using each month's last fitted occurrence; exact dates are in the table. |
| `NAME_forecast_observation_scale.png` | Future observation scale, including uncertainty and any specified linear/RW evolution. |
| `NAME_forecast_by_month.png` | All 12 calendar months separately, with recent observed history. Predictive intervals include observation noise. |
| `NAME_forecast_year.png`, `NAME_forecast_season.png` | Full-year and DJF/MAM/JJA/SON predictions, with the appropriate aggregation for the margin. |
| `NAME_forecast_monthly_risk.png`, `NAME_forecast_risk_by_month.png` | Future probability for the configured monthly threshold. Lines use mean probabilities. |
| `NAME_forecast_year_risk_curves.png`, `NAME_forecast_season_risk_curves.png` | Probability against a threshold for the aggregated quantity; avoids imposing a monthly-mean threshold on an annual mean. |
| `NAME_forecast_year_variance.csv` | For Gaussian means, decomposition into latent/parameter uncertainty and expected conditional observation variance. |
| `NAME_omitted_partial_year.csv`, `NAME_omitted_partial_season.csv` | Windows excluded because the forecast did not contain the full period. |
| `NAME_prediction_notes.json` | Actual dates, interval interpretation, temporal independence assumption and report provenance. |

More predictive draws reduce simulation noise; they do not repair a low MCMC
effective sample size. Resampling 500 posterior draws to make 2000 forecast
paths does not produce 2000 independent parameter draws. A split-chain R-hat
from one original chain cannot establish between-chain convergence.

## Public API

```python
import bucex as bx

fit = bx.load_fit("fit.bucex")
forecast = fit.forecast(120, draws=1000, seed=17)
forecast.summary(phase=7, level=.95)       # future Julys
forecast.risk_summary(25, phase=7)         # monthly event, original response tail
annual = forecast.aggregate()             # defaults from the margin family/tail
seasons = forecast.aggregate(frequency="season")
winter = forecast.aggregate(months=(12, 1, 2))
annual.summary(level=.95)
annual.risk_curve([15., 16., 17.])          # illustrative thresholds for an annual mean
annual.risk_summary(16., direction=">")
bx.save_prediction_report(fit, "figures", forecast=forecast, threshold=25)
```

For a multiseries forecast, pass `channel="TXm"` (or the required name) to
aggregation, risk and report functions. Draw indices are retained, so a
contemporaneous copula remains present in cross-channel aggregated ensembles.
The report exports marginal risk curves. A probability of a compound event
across annual summaries can be estimated jointly from aligned aggregated
observation draws; separate per-channel marginal probabilities must not be
multiplied when dependence is present.

## What is being aggregated?

For TXm/TNm, a year/season is the day-count-weighted average of the monthly
means. This equals the corresponding daily mean when monthly means include
all days. Leap February receives 29 days. Use `weighting="equal"` only when an
equal-month average is the intended estimand. If monthly summaries represent
incomplete sets of days, calendar weights alone do not reconstruct a daily
mean; prepare complete monthly blocks or explicitly resolve missingness first.

For TXx/TNx, use the maximum of monthly block maxima. For TXn/TNn, use the
minimum of monthly block minima, after returning to original temperature
orientation. The operation applies **inside each simulated posterior path**,
not to fitted locations or monthly interval endpoints. Aggregation preserves
shared parameters, temporal state uncertainty, and any contemporaneous copula.
Extremes with different monthly locations/scales need not aggregate to one
GEV family. Gaussian observations representing quantities other than monthly
means should use an explicit `reduction="max"`, `"min"` or `"mean"` as appropriate.

Only complete windows enter the default output. DJF 2027 means December 2026
through February 2027. For a fit ending August 2026, the first complete future
calendar year is 2027; the first complete season is SON 2026. A 12-month
September–August forecast contains no complete calendar year. Use a longer
horizon, or `include_partial=True` for an explicitly labelled partial window.
This release does not splice observed January–August into a partial-2026
forecast and call that a full-year forecast. Such a conditional current-year
prediction is a separate estimand and is not silently substituted.

## Observation scale and uncertainty

The reference model specifies

`log(sigma_t) = log(sigma_baseline) + d_month(t)`, with `sum(d_month)=0`.

Thus `scale_mode="constant"` plus `seasonal_scale=true` means a **repeating
seasonal scale**, not one common SD. The baseline is the geometric mean of
the 12 monthly scales. `sigma_baseline**2 ~ InverseGamma(2,2)`; orthonormal
zero-sum log-scale coordinates have independent `Normal(0,0.3**2)` priors.
The raw 12 constrained effects are consequently correlated, not 12 independent
Normal coefficients. These effects concern residual variability, separately
from changing seasonal means in the latent location process.

For Gaussian means, sigma is residual SD. For a GEV margin it is a scale
parameter, not generally the observation SD. Conditional observation noise,
future state innovations and parameter uncertainty all enter predictive draws.
`--scale linear` adds a log-scale slope; `--scale rw` adds a random-walk log
scale. The seasonal cycle remains unless explicitly disabled in the model.
The forecast simulates the new RW increments. Choosing among these models
requires held-out predictive comparison, not merely a wider fitted band.

Given a latent path and parameters, the variance of a Gaussian weighted mean
is `sum(w_t**2 * sigma_t**2)` under conditional temporal independence. Total
predictive variance additionally includes uncertainty in the weighted latent
mean. Neither the average sigma nor the average monthly interval width is the
annual predictive SD. Positive remaining residual serial correlation can make
the conditional independence variance too small. A cross-series copula alone
does **not** solve that problem.

## Risk bands

Gaussian aggregate-mean probabilities use the analytic conditional Gaussian
distribution. Extreme probabilities multiply monthly CDFs/survival functions
within each parameter/state draw and then average over draws. Observation
noise is integrated rather than estimated by a single zero/one event per draw.
For an upper extreme, the yearly exceedance probability is
`E[1 - product_t F_t(threshold | state, parameters)]`.
It is not `1 - product_t E[F_t(...)]`. For Gaussian means the event that the
annual mean exceeds a threshold is different from at least one monthly mean
exceeding that threshold.

Curves show mean probabilities. Shading gives quantiles of the conditional
risks across posterior parameters **and future latent paths**. This is neither
a Monte Carlo confidence interval nor a posterior interval for a risk with
future paths already integrated out. Obtaining the latter needs nested future
path integration for each parameter draw. In-sample smoothed PIT/QQ checks
likewise do not establish held-out calibration. These distinctions remain
explicit in the exported notes.
